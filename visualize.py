import argparse
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


def get_position_paths(traj):
    """
    traj: (T, D). D = 2 * n_disks (pos + vel); positions are traj[:, :D/2].
    Returns (n_disks, T, 2) — x,y path for each disk.
    """
    T, D = traj.shape
    state_dim = D // 2  # positions only
    n_disks = state_dim // 2
    paths = []
    for k in range(n_disks):
        x = traj[:, 2 * k]
        y = traj[:, 2 * k + 1]
        paths.append(torch.stack([x, y], dim=-1))
    return torch.stack(paths, dim=0)


def plot_trajectories(trajs, ax=None, colors=None, alpha=0.7, linestyle="-", linewidth=1.5,
                      label_prefix=None):
    """
    trajs: (num_samples, T, D). Plot each sample's disk paths on ax.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    trajs = torch.as_tensor(trajs)
    if trajs.dim() == 2:
        trajs = trajs.unsqueeze(0)
    n_samples, T, D = trajs.shape
    n_disks = (D // 2) // 2
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_samples, n_disks)))
    for s in range(trajs.shape[0]):
        paths = get_position_paths(trajs[s])  # (n_disks, T, 2)
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


def main():
    parser = argparse.ArgumentParser(description="Visualize sampled trajectories")
    parser.add_argument("trajectories", type=str, help="Path to .pt file with shape (num_samples, T, D)")
    parser.add_argument("--max_samples", type=int, default=None, help="Plot at most this many samples (default: all). Use 1 for one trajectory = 2 dots + 2 squares for two disks")
    parser.add_argument("--out", type=str, default="samples_plot.png", help="Output figure path")
    parser.add_argument("--ref_trajectory", type=str, default=None, help="Overlay one reference (original) trajectory from dataset; pass data root path (e.g. data_trajectories)")
    parser.add_argument("--env_robot", type=str, default="EnvEmptyNoWait2D-RobotCompositeTwoPlanarDisk", help="env_robot subdirectory for reference trajectories")
    parser.add_argument("--goal", type=str, default=None, help="Path to .pt goal tensor (state_dim,) to show goal markers")
    parser.add_argument("--show", action="store_true", help="Show interactive figure")
    args = parser.parse_args()

    if plt is None or np is None:
        print("Install matplotlib and numpy: pip install matplotlib numpy")
        return

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
                n_disks_ref = (ref.shape[2] // 2) // 2
                plot_trajectories(ref, ax=ax, colors=["black"] * n_disks_ref, alpha=0.7,
                                  linestyle="--", linewidth=2, label_prefix="ref")
                ax.set_title("Reference (dashed) vs sampled (solid)")
            else:
                print("No reference trajectories found; skipping overlay.")
        else:
            print("Ref data root not found; skipping reference overlay.")
    else:
        ax.set_title("Sampled trajectories (dots=start, squares=goal)")

    plot_trajectories(trajs, ax=ax, alpha=0.8, label_prefix="sampled")

    if args.goal:
        goal = torch.load(root / args.goal, weights_only=True).cpu()
        D = trajs.shape[-1]
        n_disks = (D // 2) // 2
        plot_goal_markers(goal, ax, n_disks)

    if args.ref_trajectory or args.goal:
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
