"""Test script: run sampling with ECI obstacle projection and produce a plot.

Run from repo root: ./venv/bin/python test_obstacles_run.py
"""
import argparse
from pathlib import Path

import torch
import yaml

from models import ActionVectorField
from obstacles import ObstacleScene
from sample_action import rollout

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Install matplotlib: pip install matplotlib")

from visualize import plot_obstacles


def _disk_center_clear(
    xy: torch.Tensor,
    scene: ObstacleScene,
    robot_r: float,
    slack: float = 0.02,
) -> bool:
    for obs in scene.obstacles:
        dist = (xy - obs.center).norm()
        if dist < obs.radius + robot_r + slack:
            return False
    return True


def _inter_disk_clear(
    pos: torch.Tensor,
    n_disks: int,
    robot_radii: list[float],
    slack: float = 0.02,
) -> bool:
    for i in range(n_disks):
        for j in range(i + 1, n_disks):
            pi = pos[2 * i : 2 * i + 2]
            pj = pos[2 * j : 2 * j + 2]
            if (pi - pj).norm() < robot_radii[i] + robot_radii[j] + slack:
                return False
    return True


def _sample_feasible_positions(
    scene: ObstacleScene,
    robot_radii: list[float],
    n_disks: int,
    device: torch.device,
    box: float = 0.88,
    max_tries: int = 400,
) -> torch.Tensor | None:
    """Sample joint disk centers (pos_dim,) inside [-box, box]^2, clearance to obstacles and pairwise."""
    pos_dim = 2 * n_disks
    for _ in range(max_tries):
        pos = torch.empty(pos_dim, device=device).uniform_(-box, box)
        ok = True
        for k in range(n_disks):
            if not _disk_center_clear(pos[2 * k : 2 * k + 2], scene, robot_radii[k]):
                ok = False
                break
        if ok and _inter_disk_clear(pos, n_disks, robot_radii):
            return pos
    return None


def _pack_state(pos: torch.Tensor, state_dim: int) -> torch.Tensor:
    s = torch.zeros(state_dim, device=pos.device, dtype=pos.dtype)
    s[: pos.numel()] = pos
    return s


def _sample_starts_goals(
    num_samples: int,
    scene: ObstacleScene,
    robot_radii: list[float],
    n_disks: int,
    state_dim: int,
    device: torch.device,
    min_separation: float = 0.25,
    pos_jitter: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diverse feasible start/goal pairs with minimum disk-wise displacement."""
    pos_dim = 2 * n_disks
    starts: list[torch.Tensor] = []
    goals: list[torch.Tensor] = []

    for _ in range(num_samples * 50):
        if len(starts) >= num_samples:
            break
        sp = _sample_feasible_positions(scene, robot_radii, n_disks, device)
        gp = _sample_feasible_positions(scene, robot_radii, n_disks, device)
        if sp is None or gp is None:
            continue
        sep_ok = True
        for k in range(n_disks):
            if (sp[2 * k : 2 * k + 2] - gp[2 * k : 2 * k + 2]).norm() < min_separation:
                sep_ok = False
                break
        if not sep_ok:
            continue
        if pos_jitter > 0:
            sp = sp + torch.randn_like(sp) * pos_jitter
            sp = sp.clamp(-0.92, 0.92)
            if not all(
                _disk_center_clear(sp[2 * k : 2 * k + 2], scene, robot_radii[k])
                for k in range(n_disks)
            ):
                continue
            if not _inter_disk_clear(sp, n_disks, robot_radii):
                continue
        starts.append(sp)
        goals.append(gp)

    if len(starts) < num_samples:
        raise RuntimeError(
            f"Only found {len(starts)}/{num_samples} feasible start/goal pairs; "
            "try lowering min_separation or fewer/smaller obstacles."
        )

    S = torch.stack([_pack_state(s, state_dim) for s in starts])
    G = torch.stack([_pack_state(g, state_dim) for g in goals])
    return S, G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility (random if omitted)")
    parser.add_argument(
        "--goal_threshold",
        type=float,
        default=0.05,
        help="Mark disk as reached when within this distance of goal position",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of rollouts with different sampled start/goal pairs",
    )
    parser.add_argument(
        "--min_separation",
        type=float,
        default=0.28,
        help="Minimum start–goal distance per disk (encourages longer paths)",
    )
    parser.add_argument(
        "--start_jitter",
        type=float,
        default=0.04,
        help="Gaussian noise on start positions (0 = fixed sampled starts)",
    )
    parser.add_argument(
        "--obstacles_file",
        type=str,
        default=None,
        help="YAML obstacle list (default: configs/test_obstacles.yaml)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Using seed {args.seed}")

    root = Path(__file__).resolve().parent
    device = torch.device("cpu")

    ckpt_path = root / "checkpoints_action" / "action_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not ckpt.get("per_disk_actions", False):
        print("Checkpoint is not per-disk trained; run train_action.py first.")
        return

    model = ActionVectorField(
        action_dim=ckpt["action_dim"],
        condition_dim=ckpt["condition_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_disks = ckpt["n_disks"]
    state_dim = ckpt["state_dim"]

    obs_path = Path(args.obstacles_file) if args.obstacles_file else root / "configs" / "test_obstacles.yaml"
    with open(obs_path) as f:
        obs_cfg = yaml.safe_load(f)
    robot_radii = [0.08] * n_disks
    scene = ObstacleScene.from_config(obs_cfg, robot_radii).to(device)
    print(f"Loaded {len(scene.obstacles)} obstacles from {obs_path}")

    num_samples = args.num_samples

    start, goal = _sample_starts_goals(
        num_samples,
        scene,
        robot_radii,
        n_disks,
        state_dim,
        device,
        min_separation=args.min_separation,
        pos_jitter=args.start_jitter,
    )

    print(f"Running {num_samples} trajectories with ECI obstacle projection...")
    print(f"  goal_threshold={args.goal_threshold}")
    traj, reached = rollout(
        model=model,
        start=start,
        goal=goal,
        n_disks=n_disks,
        n_ode_steps=50,
        max_steps=128,
        goal_threshold=args.goal_threshold,
        scene=scene,
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plot_obstacles(scene, ax)

    traj_cpu = traj.cpu()
    goal_cpu = goal.cpu()
    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")
    for s in range(num_samples):
        c = cmap((s % 20) / 19.0)
        x = traj_cpu[s, :, 0].numpy()
        y = traj_cpu[s, :, 1].numpy()
        ax.plot(x, y, color=c, alpha=0.75, linewidth=1.4, label=f"sample {s}")
        ax.scatter(x[0], y[0], color=c, s=36, zorder=5, marker="o")
        ax.scatter(x[-1], y[-1], color=c, s=36, zorder=5, marker="s")
        gx = goal_cpu[s, 0].item()
        gy = goal_cpu[s, 1].item()
        ax.scatter(
            gx,
            gy,
            color=c,
            s=140,
            zorder=10,
            marker="*",
            edgecolors="black",
            linewidths=0.35,
        )

    ax.set_title("ECI sampling — varied starts/goals and obstacles")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path = root / "obstacle_test_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
