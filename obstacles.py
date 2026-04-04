from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CircleObstacle:
    center: torch.Tensor
    radius: float

    def to(self, device: torch.device) -> CircleObstacle:
        return CircleObstacle(center=self.center.to(device), radius=self.radius)


@dataclass
class ObstacleScene:
    obstacles: list[CircleObstacle] = field(default_factory=list)
    robot_radii: list[float] = field(default_factory=list)

    def to(self, device: torch.device) -> ObstacleScene:
        return ObstacleScene(
            obstacles=[o.to(device) for o in self.obstacles],
            robot_radii=list(self.robot_radii),
        )

    @property
    def n_disks(self) -> int:
        return len(self.robot_radii)

    @staticmethod
    def from_config(
        obstacles_cfg: list[dict],
        robot_radii: list[float],
    ) -> ObstacleScene:
        obs = []
        for o in obstacles_cfg:
            center = torch.tensor(o["center"], dtype=torch.float32)
            obs.append(CircleObstacle(center=center, radius=float(o["radius"])))
        return ObstacleScene(obstacles=obs, robot_radii=robot_radii)


def project_to_feasible(
    action: torch.Tensor,
    state: torch.Tensor,
    scene: ObstacleScene,
) -> torch.Tensor:
    n_disks = scene.n_disks
    pos_dim = n_disks * 2
    out = action.clone()
    next_pos = state[:, :pos_dim] + out

    for k in range(n_disks):
        r_disk = scene.robot_radii[k]
        for obs in scene.obstacles:
            disk_pos = next_pos[:, 2 * k : 2 * k + 2]
            diff = disk_pos - obs.center
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            min_dist = obs.radius + r_disk
            penetrating = (dist < min_dist).squeeze(-1)
            if penetrating.any():
                normal = diff[penetrating] / dist[penetrating]
                correction = normal * (min_dist - dist[penetrating])
                out[penetrating, 2 * k : 2 * k + 2] += correction
                next_pos = state[:, :pos_dim] + out

    for i in range(n_disks):
        for j in range(i + 1, n_disks):
            pi = next_pos[:, 2 * i : 2 * i + 2]
            pj = next_pos[:, 2 * j : 2 * j + 2]
            diff = pi - pj
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            min_dist = scene.robot_radii[i] + scene.robot_radii[j]
            penetrating = (dist < min_dist).squeeze(-1)
            if penetrating.any():
                normal = diff[penetrating] / dist[penetrating]
                overlap = min_dist - dist[penetrating]
                half_correction = normal * (overlap * 0.5)
                out[penetrating, 2 * i : 2 * i + 2] += half_correction
                out[penetrating, 2 * j : 2 * j + 2] -= half_correction
                next_pos = state[:, :pos_dim] + out

    return out


def adjust_disk_action_slice(
    a_slice: torch.Tensor,
    action_full_prefix: torch.Tensor,
    disk_k: int,
    n_disks: int,
    state: torch.Tensor,
    scene: ObstacleScene,
) -> torch.Tensor:
    
    B, two = a_slice.shape
    assert two == 2
    pos_dim = n_disks * 2
    device, dtype = a_slice.device, a_slice.dtype
    full = torch.zeros(B, pos_dim, device=device, dtype=dtype)
    full[:, : 2 * disk_k] = action_full_prefix[:, : 2 * disk_k].to(dtype)
    full[:, 2 * disk_k : 2 * disk_k + 2] = a_slice
    full = project_to_feasible(full, state, scene)
    return full[:, 2 * disk_k : 2 * disk_k + 2]


def check_collisions(
    positions: torch.Tensor,
    scene: ObstacleScene,
) -> torch.Tensor:
    squeezed = positions.dim() == 2
    if squeezed:
        positions = positions.unsqueeze(1)
    B, T, _ = positions.shape
    n_disks = scene.n_disks

    colliding = torch.zeros(B, T, dtype=torch.bool, device=positions.device)

    for k in range(n_disks):
        disk_pos = positions[:, :, 2 * k : 2 * k + 2]
        r_disk = scene.robot_radii[k]
        for obs in scene.obstacles:
            d = (disk_pos - obs.center).norm(dim=-1) - obs.radius - r_disk
            colliding = colliding | (d < 0)

    # Inter-disk collisions
    for i in range(n_disks):
        for j in range(i + 1, n_disks):
            pi = positions[:, :, 2 * i : 2 * i + 2]
            pj = positions[:, :, 2 * j : 2 * j + 2]
            d = (pi - pj).norm(dim=-1) - scene.robot_radii[i] - scene.robot_radii[j]
            colliding = colliding | (d < 0)

    if squeezed:
        return colliding.squeeze(1)
    return colliding



