import argparse
from pathlib import Path

import torch
import yaml

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    plt = None
    np = None
    mpatches = None

from obstacles import ObstacleScene


def infer_n_disks_from_dim(D: int, robot_radii: list[float] | None = None) -> int:
    """Infer number of 2D disks from trajectory/state dimensionality.

    Active rollout saves full state as [positions, velocities] (4 * n_disks).
    Older data may be positions only (2 * n_disks). Prefer robot_radii when provided.
    """
    if robot_radii:
        return len(robot_radii)
    if D >= 8 and D % 4 == 0:
        return D // 4
    if D % 2 != 0:
        raise ValueError(f"Cannot infer n_disks from odd dimension D={D}")
    return D // 2


def infer_pos_dim(D: int, n_disks: int) -> int:
    if D == 4 * n_disks:
        return 2 * n_disks
    if D == 2 * n_disks:
        return D
    # Fallback for mixed/legacy tensors: keep first half as positions.
    return D // 2


def get_position_paths(traj, n_disks: int):
    """
    traj: (T, D), where D is either 2*n_disks (positions only) or
    4*n_disks ([positions, velocities]).
    Returns (n_disks, T, 2) — x,y path for each disk.
    """
    T, D = traj.shape
    pos_dim = infer_pos_dim(D, n_disks)
    pos = traj[:, :pos_dim]
    paths = []
    for k in range(n_disks):
        x = pos[:, 2 * k]
        y = pos[:, 2 * k + 1]
        paths.append(torch.stack([x, y], dim=-1))
    return torch.stack(paths, dim=0)


def plot_trajectories(trajs, ax=None, colors=None, alpha=0.7, linestyle="-", linewidth=1.5,
                      label_prefix=None, n_disks: int | None = None):
    """
    trajs: (num_samples, T, D). Plot each sample's disk paths on ax.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    trajs = torch.as_tensor(trajs)
    if trajs.dim() == 2:
        trajs = trajs.unsqueeze(0)
    n_samples, _, D = trajs.shape
    if n_disks is None:
        n_disks = infer_n_disks_from_dim(D)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_samples, n_disks)))
    for s in range(trajs.shape[0]):
        paths = get_position_paths(trajs[s], n_disks=n_disks)  # (n_disks, T, 2)
        for k in range(paths.shape[0]):
            x = paths[k, :, 0].numpy()
            y = paths[k, :, 1].numpy()
            c = colors[k % len(colors)]
            lbl = f"{label_prefix} disk {k}" if label_prefix and s == 0 else None
            ax.plot(x, y, color=c, alpha=alpha, linewidth=linewidth, linestyle=linestyle,
                    label=lbl)
            ax.scatter(x[0], y[0], color=c, s=40, zorder=5, marker="o")
            ax.scatter(x[-1], y[-1], color=c, s=40, zorder=5, marker="s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_goal_markers(goal, ax, n_disks):
    """Draw goal positions as large star markers."""
    for k in range(n_disks):
        gx = goal[2 * k].item()
        gy = goal[2 * k + 1].item()
        ax.scatter(gx, gy, color="red", s=200, zorder=10, marker="*",
                   label=f"goal disk {k}" if k == 0 else None)


def plot_obstacles(
    scene: ObstacleScene,
    ax,
    obstacle_color="#444444",
    obstacle_alpha=0.5,
    robot_radii_alpha=0.12,
    show_inflated: bool = False,
):
    """Draw circle obstacles as filled patches.

    If ``show_inflated`` is True, also draws robot-inflated boundaries (Minkowski sum)
    as dashed outlines.
    """
    for i, obs in enumerate(scene.obstacles):
        cx, cy = obs.center[0].item(), obs.center[1].item()
        circle = plt.Circle((cx, cy), obs.radius, color=obstacle_color,
                             alpha=obstacle_alpha, zorder=2,
                             label="obstacle" if i == 0 else None)
        ax.add_patch(circle)
        if show_inflated and scene.robot_radii:
            inflated_r = obs.radius + max(scene.robot_radii)
            inflated = plt.Circle(
                (cx, cy),
                inflated_r,
                fill=False,
                linestyle="--",
                linewidth=1.0,
                color=obstacle_color,
                alpha=robot_radii_alpha,
                zorder=1,
                label="inflated obstacle" if i == 0 else None,
            )
            ax.add_patch(inflated)


def main():
    parser = argparse.ArgumentParser(description="Visualize sampled trajectories")
    parser.add_argument("trajectories", type=str, help="Path to .pt file with shape (num_samples, T, D)")
    parser.add_argument("--max_samples", type=int, default=None, help="Plot at most this many samples (default: all). Use 1 for one trajectory = 2 dots + 2 squares for two disks")
    parser.add_argument("--out", type=str, default="samples_plot.png", help="Output figure path")
    parser.add_argument("--ref_trajectory", type=str, default=None, help="Overlay one reference (original) trajectory from dataset; pass data root path (e.g. data_trajectories)")
    parser.add_argument("--env_robot", type=str, default="EnvEmptyNoWait2D-RobotCompositeTwoPlanarDisk", help="env_robot subdirectory for reference trajectories")
    parser.add_argument("--goal", type=str, default=None, help="Path to .pt goal tensor (state_dim,) to show goal markers")
    parser.add_argument("--obstacles_file", type=str, default=None, help="YAML file with obstacle list [{center: [x,y], radius: r}, ...]")
    parser.add_argument("--robot_radii", nargs="+", type=float, default=None, help="Robot disk radii (for inflated boundaries when --show_inflated_obstacles)")
    parser.add_argument(
        "--show_inflated_obstacles",
        action="store_true",
        help="Draw dashed Minkowski-inflated obstacle outlines",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive figure")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    path = root / args.trajectories
    if not path.exists():
        print(f"File not found: {path}")
        return

    trajs = torch.load(path, weights_only=True)
    if isinstance(trajs, torch.Tensor):
        trajs = trajs.cpu()
    else:
        trajs = torch.tensor(trajs)
    if args.max_samples is not None and trajs.shape[0] > args.max_samples:
        trajs = trajs[: args.max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if args.ref_trajectory:
        from data.action_dataset import _find_sample_dirs
        data_root = Path(args.ref_trajectory)
        if not data_root.exists():
            data_root = root / "data_trajectories"
        if data_root.exists():
            candidates = [
                (e, s) for e, s in _find_sample_dirs(data_root) if e == args.env_robot
            ]
            if candidates:
                env_robot, sample_id = candidates[0]
                ref = torch.load(data_root / env_robot / sample_id / "trajs-free.pt", weights_only=True)
                if ref.dim() == 3 and ref.shape[0] == 1:
                    ref = ref.squeeze(0)
                ref = ref.unsqueeze(0).cpu()  # (1, T, D)
                n_disks_ref = infer_n_disks_from_dim(ref.shape[2])
                plot_trajectories(ref, ax=ax, colors=["black"] * n_disks_ref, alpha=0.7,
                                  linestyle="--", linewidth=2, label_prefix="ref",
                                  n_disks=n_disks_ref)
                ax.set_title("Reference (dashed) vs sampled (solid)")
            else:
                print("No reference trajectories found; skipping overlay.")
        else:
            print("Ref data root not found; skipping reference overlay.")
    else:
        ax.set_title("Sampled trajectories (dots=start, squares=goal)")

    # Draw obstacles if provided
    scene = None
    if args.obstacles_file:
        obs_path = Path(args.obstacles_file)
        if not obs_path.is_absolute():
            obs_path = root / obs_path
        with open(obs_path) as f:
            obs_cfg = yaml.safe_load(f) or []
        n_disks = infer_n_disks_from_dim(
            trajs.shape[-1], robot_radii=args.robot_radii
        )
        radii = args.robot_radii if args.robot_radii else [0.08] * n_disks
        scene = ObstacleScene.from_config(obs_cfg, radii)
        plot_obstacles(scene, ax, show_inflated=args.show_inflated_obstacles)

    n_disks = infer_n_disks_from_dim(
        trajs.shape[-1], robot_radii=(scene.robot_radii if scene is not None else None)
    )
    plot_trajectories(trajs, ax=ax, alpha=0.8, label_prefix="sampled", n_disks=n_disks)

    if args.goal:
        goal_path = Path(args.goal)
        if not goal_path.is_absolute():
            goal_path = root / goal_path
        goal = torch.load(goal_path, weights_only=True).cpu()
        plot_goal_markers(goal, ax, n_disks)

    if args.ref_trajectory or args.goal or scene is not None:
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = root / args.out
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
