import math
from typing import List, Union

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ActionVectorField(nn.Module):
    def __init__(
        self,
        action_dim: int,
        condition_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: Union[List[int], tuple] = (512, 512, 512),
        activation: str = "silu",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim

        act = nn.SiLU if activation.lower() == "silu" else nn.GELU
        in_dim = action_dim + time_embed_dim + condition_dim
        dims = [in_dim] + list(hidden_dims) + [action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act())
        self.mlp = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.squeeze(-1)
        t = t.to(a.dtype)
        t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
        inp = torch.cat([a, t_embed, c], dim=-1)
        return self.mlp(inp)
