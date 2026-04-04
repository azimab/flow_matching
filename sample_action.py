import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from data import ActionStepDataset, DEFAULT_ENV_ROBOT
from models import ActionVectorField
from obstacles import (
    ObstacleScene,
    adjust_disk_action_slice,
    check_collisions,
    project_to_feasible,
)

# Obstacle sampling: bridge mixing count; noise resample period (0 = effectively never).
_ECI_N_MIX = 3
_ECI_RESAMPLE_STEP = 0

# Defaults only when a key is missing from YAML (see load_config).
_FALLBACK = {
    "checkpoint": "checkpoints_action/action_checkpoint.pt",
    "num_samples": 8,
    "n_ode_steps": 50,
    "max_rollout_steps": 128,
    "goal_threshold": 0.05,
    "condition_from_data": False,
    "data_root": None,
    "env_robot": DEFAULT_ENV_ROBOT,
    "out": None,
    "robot_radii": None,
    "obstacles_file": None,
    "obstacles": [],
}


def load_config() -> SimpleNamespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Roll out flow-matching policy (settings from YAML).")
    p.add_argument(
        "config",
        nargs="?",
        default=str(root / "configs" / "action_mlp.yaml"),
        help="YAML with rollout / obstacle keys",
    )
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    p.add_argument("--out", default=None, help="Override output .pt path")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        from_yaml = yaml.safe_load(f) or {}

    cfg = {**_FALLBACK, **from_yaml}
    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint
    if args.out is not None:
        cfg["out"] = args.out
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples
    if args.device is not None:
        cfg["device"] = args.device
    elif "device" not in cfg or cfg["device"] is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return SimpleNamespace(**cfg)


def sample_action(
    model: ActionVectorField,
    state: torch.Tensor,
    goal: torch.Tensor,
    n_disks: int,
    n_ode_steps: int,
    scene: ObstacleScene | None = None,
) -> torch.Tensor:
    device = state.device
    B = state.shape[0]
    pos_dim = n_disks * 2
    dt = 1.0 / n_ode_steps
    goal_features = ActionStepDataset.compute_goal_features(state, goal, n_disks)
    base_c = torch.cat([state, goal, goal_features], dim=-1)

    use_eci = scene is not None and len(scene.obstacles) > 0
    rs = _ECI_RESAMPLE_STEP
    if use_eci and rs == 0:
        rs = n_ode_steps * _ECI_N_MIX + 1

    action_full = torch.zeros(B, pos_dim, device=device)
    for k in range(n_disks):
        disk_oh = torch.zeros(B, n_disks, device=device)
        disk_oh[:, k] = 1.0
        condition = torch.cat([base_c, disk_oh], dim=-1)

        noise = torch.randn(B, model.action_dim, device=device)
        a = noise.clone()
        cnt = 0

        for i in range(n_ode_steps):
            t_scalar = i * dt
            t = torch.full((B,), t_scalar, device=device)

            if use_eci:
                for u in range(_ECI_N_MIX):
                    cnt += 1
                    if cnt % rs == 0:
                        noise = torch.randn(B, model.action_dim, device=device)
                    with torch.no_grad():
                        v = model(a, t, condition)
                    br = (1.0 - t).unsqueeze(-1)
                    a1 = a + v * br
                    a1 = adjust_disk_action_slice(
                        a1, action_full, k, n_disks, state, scene
                    )
                    if u < _ECI_N_MIX - 1:
                        a = a1 * t.unsqueeze(-1) + noise * (1.0 - t.unsqueeze(-1))
                    else:
                        t_next = t_scalar + dt
                        tn = torch.full((B,), t_next, device=device)
                        a = a1 * tn.unsqueeze(-1) + noise * (1.0 - tn.unsqueeze(-1))
            else:
                with torch.no_grad():
                    v = model(a, t, condition)
                a = a + v * dt

        action_full[:, 2 * k : 2 * k + 2] = a

    return action_full


def rollout(
    model: ActionVectorField,
    start: torch.Tensor,
    goal: torch.Tensor,
    n_disks: int,
    n_ode_steps: int = 50,
    max_steps: int = 128,
    goal_threshold: float = 0.05,
    scene: ObstacleScene | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = start.device
    B = start.shape[0]
    state_dim = start.shape[1]
    pos_dim = state_dim // 2

    state = start.clone()
    trajectory = [state.clone()]

    disk_reached = torch.zeros(B, n_disks, dtype=torch.bool, device=device)
    reached = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_steps):
        if reached.all():
            break

        action = sample_action(
            model, state, goal, n_disks=n_disks, n_ode_steps=n_ode_steps, scene=scene
        )

        if scene is not None and len(scene.obstacles) > 0:
            action = project_to_feasible(action, state, scene)

        if disk_reached.any():
            for k in range(n_disks):
                s = 2 * k
                e = s + 2
                action[disk_reached[:, k], s:e] = 0.0
        action[reached] = 0.0

        next_pos = state[:, :pos_dim] + action
        # Within goal_threshold, snap position to exact goal (exact star alignment;
        # avoids freezing short after marking reached with zero action).
        for k in range(n_disks):
            s, e = 2 * k, 2 * k + 2
            gk = goal[:, s:e]
            dk = (next_pos[:, s:e] - gk).norm(dim=-1)
            snap_m = dk < goal_threshold
            next_pos[snap_m, s:e] = gk[snap_m]
        next_vel = next_pos - state[:, :pos_dim]
        state = torch.cat([next_pos, next_vel], dim=-1)
        trajectory.append(state.clone())

        for k in range(n_disks):
            dx = state[:, 2 * k] - goal[:, 2 * k]
            dy = state[:, 2 * k + 1] - goal[:, 2 * k + 1]
            dist_k = (dx ** 2 + dy ** 2).sqrt()
            disk_reached[:, k] = disk_reached[:, k] | (dist_k < goal_threshold)
        reached = reached | disk_reached.all(dim=1)

    trajectory = torch.stack(trajectory, dim=1)
    return trajectory, reached


def compute_metrics(
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    reached: torch.Tensor,
    n_disks: int,
    scene: ObstacleScene | None = None,
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

    metrics = {
        "goal_reaching_rate": goal_rate,
        "path_efficiency": efficiency,
        "smoothness_msa": smoothness,
    }

    if scene is not None and len(scene.obstacles) > 0:
        positions = trajectory[:, :, :pos_dim]
        colliding = check_collisions(positions, scene)
        metrics["collision_rate"] = colliding.float().mean().item()
        metrics["collision_free_traj_rate"] = (
            (~colliding.any(dim=1)).float().mean().item()
        )

    return metrics


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
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_disks = ckpt["n_disks"]

    scene = None
    obstacles_cfg: list[dict] = []
    if cfg.obstacles_file:
        with open(cfg.obstacles_file) as f:
            obstacles_cfg = yaml.safe_load(f) or []
    elif getattr(cfg, "obstacles", None):
        obstacles_cfg = cfg.obstacles

    if obstacles_cfg:
        radii = cfg.robot_radii if cfg.robot_radii else [0.08] * n_disks
        scene = ObstacleScene.from_config(obstacles_cfg, radii).to(device)
        print(f"Loaded {len(scene.obstacles)} obstacles, robot_radii={scene.robot_radii}")
    else:
        print("No obstacles configured.")

    if cfg.condition_from_data:
        data_root = Path(cfg.data_root) if cfg.data_root else (root / "data_trajectories")
        ds = ActionStepDataset(data_root, env_robot=cfg.env_robot)
        state0, goal0, _, _, _ = ds[0]
        start = state0.unsqueeze(0).expand(cfg.num_samples, -1).to(device)
        goal = goal0.unsqueeze(0).expand(cfg.num_samples, -1).to(device)
        print("Using start/goal from first dataset sample.")
    else:
        state_dim = ckpt["state_dim"]
        start = torch.empty(cfg.num_samples, state_dim, device=device).uniform_(-1.0, 1.0)
        goal = torch.empty(cfg.num_samples, state_dim, device=device).uniform_(-1.0, 1.0)
        start.clamp_(-1.0, 1.0)
        goal.clamp_(-1.0, 1.0)

    print(f"Rolling out {cfg.num_samples} trajectories (max {cfg.max_rollout_steps} steps)...")

    trajectory, reached = rollout(
        model=model,
        start=start,
        goal=goal,
        n_disks=n_disks,
        n_ode_steps=cfg.n_ode_steps,
        max_steps=cfg.max_rollout_steps,
        goal_threshold=cfg.goal_threshold,
        scene=scene,
    )

    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Reached goal: {reached.sum().item()}/{cfg.num_samples}")

    metrics = compute_metrics(trajectory, goal, reached, n_disks, scene=scene)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if cfg.out:
        out_path = root / cfg.out
        torch.save(trajectory.cpu(), out_path)
        print(f"Saved to {out_path}")
        goal_path = out_path.with_name(out_path.stem + "_goal.pt")
        torch.save(goal.cpu(), goal_path)
        print(f"Saved conditioned goal to {goal_path}")


if __name__ == "__main__":
    main()
