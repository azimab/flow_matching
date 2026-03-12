"""
Unrolls trajectories into (state_t, goal, action_t, goal_features).
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

DEFAULT_ENV_ROBOT = "EnvEmptyNoWait2D-RobotCompositeTwoPlanarDisk"


def _find_sample_dirs(root: Path):
    if not root.exists():
        return
    for env_robot in root.iterdir():
        if not env_robot.is_dir():
            continue
        for sample_dir in env_robot.iterdir():
            if not sample_dir.is_dir():
                continue
            if (sample_dir / "trajs-free.pt").exists():
                yield env_robot.name, sample_dir.name


class ActionStepDataset(Dataset):
    """
    Returns (state, goal, action, goal_features) per timestep.
    """
    def __init__(
        self,
        data_root: str | Path,
        env_robot: str = DEFAULT_ENV_ROBOT,
    ):
        self.root = Path(data_root)

        all_candidates = list(_find_sample_dirs(self.root))
        if not all_candidates:
            raise FileNotFoundError(f"No trajs-free.pt found under {self.root}")

        traj_samples = [(e, s) for e, s in all_candidates if e == env_robot]
        

        self._index: list[tuple[str, str, int]] = []
        self._traj_cache: dict[tuple[str, str], torch.Tensor] = {}

        for env_robot, sample_id in traj_samples:
            traj = self._load_trajectory(env_robot, sample_id)
            self._traj_cache[(env_robot, sample_id)] = traj
            T = traj.shape[0]
            for t in range(T - 1):
                self._index.append((env_robot, sample_id, t))

        first_traj = next(iter(self._traj_cache.values()))
        D = first_traj.shape[1]
        self.state_dim = D
        self.pos_dim = D // 2
        self.action_dim = self.pos_dim
        self.n_disks = self.pos_dim // 2

    def _load_trajectory(self, env_robot: str, sample_id: str) -> torch.Tensor:
        path = self.root / env_robot / sample_id / "trajs-free.pt"
        data = torch.load(path, weights_only=True)
        if data.dim() == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        return data

    @staticmethod
    def compute_goal_features(
        state: torch.Tensor, goal: torch.Tensor, n_disks: int
    ) -> torch.Tensor:
        pos_dim = n_disks * 2
        current_pos = state[..., :pos_dim]
        goal_pos = goal[..., :pos_dim]
        direction = goal_pos - current_pos

        distances = []
        for k in range(n_disks):
            dx = direction[..., 2 * k]
            dy = direction[..., 2 * k + 1]
            distances.append((dx ** 2 + dy ** 2).sqrt())
        dist = torch.stack(distances, dim=-1)

        return torch.cat([direction, dist], dim=-1)

    @property
    def condition_dim(self) -> int:
        return self.state_dim + self.state_dim + self.pos_dim + self.n_disks

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        env_robot, sample_id, t = self._index[idx]
        traj = self._traj_cache[(env_robot, sample_id)]
        state = traj[t]
        goal = traj[-1]
        action = traj[t + 1, :self.pos_dim] - traj[t, :self.pos_dim]
        goal_features = self.compute_goal_features(state, goal, self.n_disks)
        return state, goal, action, goal_features
