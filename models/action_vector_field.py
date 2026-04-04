import math

import torch
import torch.nn as nn

_TIME_EMBED_DIM = 64
_HIDDEN_DIMS = (512, 512, 512)


class ActionVectorField(nn.Module):
    def __init__(self, action_dim: int, condition_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.condition_dim = condition_dim

        in_dim = action_dim + _TIME_EMBED_DIM + condition_dim
        dims = [in_dim] + list(_HIDDEN_DIMS) + [action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.squeeze(-1)
        t = t.to(a.dtype)
        half = _TIME_EMBED_DIM // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        args = t * freqs.unsqueeze(0)
        t_embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        inp = torch.cat([a, t_embed, c], dim=-1)
        return self.mlp(inp)
