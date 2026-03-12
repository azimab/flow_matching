import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from data import ActionStepDataset
from models import ActionVectorField


DEFAULTS = dict(
    checkpoint="checkpoints_action/action_checkpoint.pt",
    num_samples=8,
    n_ode_steps=50,
    max_rollout_steps=128,
    goal_threshold=0.05,
    snap_radius=0.01,
    condition_from_data=False,
    data_root=None,
    env_robot="EnvEmptyNoWait2D-RobotCompositeTwoPlanarDisk",
    device="cuda" if torch.cuda.is_available() else "cpu",
    out=None,
)


def load_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=None)
    for key, default in DEFAULTS.items():
        flag = f"--{key}"
        if isinstance(default, bool):
            parser.add_argument(flag, action="store_true", default=None)
        elif default is None:
            parser.add_argument(flag, type=str, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return SimpleNamespace(**cfg)


def sample_action(
    model: ActionVectorField,
    condition: torch.Tensor,
    n_ode_steps: int = 50,
) -> torch.Tensor:
    device = condition.device
    B = condition.shape[0]
    a = torch.randn(B, model.action_dim, device=device)
    dt = 1.0 / n_ode_steps

    with torch.no_grad():
        for i in range(n_ode_steps):
            t = torch.full((B,), i * dt, device=device)
            v = model(a, t, condition)
            a = a + v * dt
    return a


def clamp_to_goal(
    action: torch.Tensor,
    state: torch.Tensor,
    goal: torch.Tensor,
    n_disks: int,
    snap_radius: float = 0.01,
) -> torch.Tensor:
    pos_dim = n_disks * 2
    current_pos = state[:, :pos_dim]
    goal_pos = goal[:, :pos_dim]
    disp = goal_pos - current_pos

    out = action.clone()
    for k in range(n_disks):
        s = 2 * k
        e = s + 2
        d_k = disp[:, s:e]
        a_k = action[:, s:e]
        d_norm = d_k.norm(dim=-1)
        a_norm = a_k.norm(dim=-1).clamp(min=1e-8)

        close = d_norm < snap_radius
        out[close, s:e] = d_k[close]

        overshoot = (~close) & (a_norm > d_norm)
        if overshoot.any():
            scale = (d_norm[overshoot] / a_norm[overshoot]).unsqueeze(-1)
            out[overshoot, s:e] = a_k[overshoot] * scale

    return out


def rollout(
    model: ActionVectorField,
    start: torch.Tensor,
    goal: torch.Tensor,
    n_disks: int,
    n_ode_steps: int = 50,
    max_steps: int = 128,
    goal_threshold: float = 0.05,
    snap_radius: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = start.device
    B = start.shape[0]
    state_dim = start.shape[1]
    pos_dim = state_dim // 2

    state = start.clone()
    trajectory = [state.clone()]

    reached = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_steps):
        if reached.all():
            break

        goal_features = ActionStepDataset.compute_goal_features(state, goal, n_disks)
        c = torch.cat([state, goal, goal_features], dim=-1)

        action = sample_action(model, c, n_ode_steps=n_ode_steps)
        action = clamp_to_goal(action, state, goal, n_disks, snap_radius=snap_radius)

        action[reached] = 0.0

        next_pos = state[:, :pos_dim] + action
        next_vel = action
        state = torch.cat([next_pos, next_vel], dim=-1)
        trajectory.append(state.clone())

        dists = []
        for k in range(n_disks):
            dx = state[:, 2 * k] - goal[:, 2 * k]
            dy = state[:, 2 * k + 1] - goal[:, 2 * k + 1]
            dists.append((dx ** 2 + dy ** 2).sqrt())
        max_dist = torch.stack(dists, dim=-1).max(dim=-1).values
        reached = reached | (max_dist < goal_threshold)

    trajectory = torch.stack(trajectory, dim=1)
    return trajectory, reached


def compute_metrics(
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    reached: torch.Tensor,
    n_disks: int,
) -> dict:
    B, T, D = trajectory.shape
    pos_dim = D // 2

    goal_rate = reached.float().mean().item()

    path_lengths = torch.zeros(B, device=trajectory.device)
    for t in range(1, T):
        delta = trajectory[:, t, :pos_dim] - trajectory[:, t - 1, :pos_dim]
        path_lengths += delta.norm(dim=-1)

    straight_line = (trajectory[:, -1, :pos_dim] - trajectory[:, 0, :pos_dim]).norm(dim=-1)
    efficiency = (straight_line / path_lengths.clamp(min=1e-6)).mean().item()

    smoothness_vals = torch.zeros(B, device=trajectory.device)
    if T > 2:
        vel = trajectory[:, :, pos_dim:]
        accel = vel[:, 1:] - vel[:, :-1]
        smoothness_vals = accel.pow(2).mean(dim=(1, 2))
    smoothness = smoothness_vals.mean().item()

    return {
        "goal_reaching_rate": goal_rate,
        "path_efficiency": efficiency,
        "smoothness_msa": smoothness,
    }


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    ckpt_path = root / cfg.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    device = torch.device(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = ActionVectorField(
        action_dim=ckpt["action_dim"],
        condition_dim=ckpt["condition_dim"],
        time_embed_dim=ckpt["time_embed_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        activation=ckpt.get("activation", "silu"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_disks = ckpt["n_disks"]

    if cfg.condition_from_data:
        data_root = Path(cfg.data_root) if cfg.data_root else (root / "data_trajectories")
        ds = ActionStepDataset(data_root, env_robot=cfg.env_robot)
        state0, goal0, _, _ = ds[0]
        start = state0.unsqueeze(0).expand(cfg.num_samples, -1).to(device)
        goal = goal0.unsqueeze(0).expand(cfg.num_samples, -1).to(device)
        print("Using start/goal from first dataset sample.")
    else:
        state_dim = ckpt["state_dim"]
        start = torch.rand(cfg.num_samples, state_dim, device=device) * 2 - 1
        goal = torch.rand(cfg.num_samples, state_dim, device=device) * 2 - 1
        print("Using random start/goal.")

    print(f"Rolling out {cfg.num_samples} trajectories (max {cfg.max_rollout_steps} steps)...")

    trajectory, reached = rollout(
        model=model,
        start=start,
        goal=goal,
        n_disks=n_disks,
        n_ode_steps=cfg.n_ode_steps,
        max_steps=cfg.max_rollout_steps,
        goal_threshold=cfg.goal_threshold,
        snap_radius=cfg.snap_radius,
    )

    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Reached goal: {reached.sum().item()}/{cfg.num_samples}")

    metrics = compute_metrics(trajectory, goal, reached, n_disks)
    print(f"Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if cfg.out:
        out_path = root / cfg.out
        torch.save(trajectory.cpu(), out_path)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
