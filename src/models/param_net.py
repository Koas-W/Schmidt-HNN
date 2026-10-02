"""Param-Net: mu -> {u_k(mu)}.

Degenerate case mu_dim = 0 (e.g. the simple pendulum): u becomes a learnable constant vector
(nn.Parameter of length sum head_ranks) and the separable product reduces to a learnable linear
combination of State-Net features; this is a declared design decision.
"""
import torch
import torch.nn as nn

from .blocks import FeatureTrunk


class ParamNet(FeatureTrunk):
    def __init__(self, mu_dim: int, width: int, depth: int,
                 head_ranks: list[int], block: str = "swiglu"):
        assert mu_dim > 0, "use ConstParamNet for mu_dim=0"
        super().__init__(mu_dim, width, depth, head_ranks, block)


class ConstParamNet(nn.Module):
    """Degenerate Param-Net for parameter-free systems: learnable constants, packed shape (sum head_ranks,)."""

    def __init__(self, head_ranks: list[int]):
        super().__init__()
        self.head_ranks = list(head_ranks)
        total = sum(self.head_ranks)
        # Initialized to 1. When the parameter factor is constant its scale is absorbed by the coordinate network, so 1 loses no generality;
        # random init at small rank can give channel multipliers near 0 (u_1=0.007 observed on one seed), suppressing that channel's effective learning rate.
        self.u = nn.Parameter(torch.ones(total))

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.u.unsqueeze(0).expand(mu.shape[0], -1)

    def split(self, packed: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(packed, self.head_ranks, dim=-1))


def build_param_net(mu_dim: int, width: int, depth: int,
                    head_ranks: list[int],
                    block: str = "swiglu") -> nn.Module:
    if mu_dim == 0:
        return ConstParamNet(head_ranks)
    return ParamNet(mu_dim, width, depth, head_ranks, block)
