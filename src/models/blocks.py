"""Low-level computation blocks and the shared trunk.

SwiGLU+Residual:
    z_V = W_V h + b_V
    z_G = W_G h + b_G
    m   = z_V ⊙ SiLU(z_G)
    h'  = h + W_proj m

MLPBlock is the equal-parameter control for the block ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUBlock(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.w_v = nn.Linear(dim, hidden)
        self.w_g = nn.Linear(dim, hidden)
        self.proj = nn.Linear(hidden, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.proj(self.w_v(h) * F.silu(self.w_g(h)))


class MLPBlock(nn.Module):
    """Equal-parameter MLP+Residual control (block ablation)."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(torch.tanh(self.fc1(h)))


class SiLUMLPBlock(nn.Module):
    """Ungated, SiLU-activated equal-parameter MLP+Residual (ablation separating the gating factor from the activation factor)."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(F.silu(self.fc1(h)))


BLOCKS = {"swiglu": SwiGLUBlock, "mlp": MLPBlock, "silu_mlp": SiLUMLPBlock}


class FeatureTrunk(nn.Module):
    """input -> embed -> blocks x depth -> heads; output is a packed tensor (B, sum head_ranks).

    Param-Net and State-Net share this trunk structure (shared backbone, one head
    per slot: one head for V, one head per lower-triangular slot of L).

    head_ranks: rank of each head (= truncated SVE rank of that output channel), a
    capacity-allocation hyperparameter. The entries of L are usually simpler than V (five
    of six systems have M = I), so R_L <= R_V is the physically sensible default; see split().
    """

    def __init__(self, input_dim: int, width: int, depth: int,
                 head_ranks: list[int], block: str = "swiglu"):
        super().__init__()
        self.head_ranks = list(head_ranks)
        self.embed = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            [BLOCKS[block](width) for _ in range(depth)])
        self.heads = nn.Linear(width, sum(self.head_ranks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.heads(h)

    def split(self, packed: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(packed, self.head_ranks, dim=-1))
