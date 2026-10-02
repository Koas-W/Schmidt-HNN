"""Ablation arm: Schmidt-HNN-concat, differing from Schmidt-HNN only in how mu is injected.

The "Cholesky + concat + SwiGLU" cell of the 2x2(x block) factor design:
- same as Schmidt-HNN: SwiGLU+Residual trunk, V/L multi-head, softplus on the diagonal of L after the output,
  H = 1/2||L^T p||^2 + V, parameter matching +-5%;
- only difference: no Param-Net x State-Net separable product; a single trunk takes concat(q, mu).

The gap between Schmidt-HNN and this model = net contribution of separable product injection (the isolating control for the headline claim).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BLOCKS
from .hamiltonian_base import HamiltonianModel, kinetic_from_factor


class PSHNNConcat(HamiltonianModel):

    def __init__(self, n_dof: int, mu_dim: int, width: int,
                 depth: int = 2, block: str = "swiglu",
                 learn_kinetic: bool = True, mass_param: str = "inverse"):
        super().__init__()
        self.n_dof, self.mu_dim = n_dof, mu_dim
        self.learn_kinetic = learn_kinetic
        self.mass_param = mass_param
        n_L = n_dof * (n_dof + 1) // 2 if learn_kinetic else 0
        self.embed = nn.Linear(n_dof + mu_dim, width)
        self.blocks = nn.ModuleList([BLOCKS[block](width)
                                     for _ in range(depth)])
        self.heads = nn.Linear(width, 1 + n_L)
        tril = torch.tril_indices(n_dof, n_dof)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])
        self.register_buffer("diag_mask", tril[0] == tril[1])

    def V_and_L(self, q: torch.Tensor,
                mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(torch.cat([q, mu], dim=-1))
        for blk in self.blocks:
            h = blk(h)
        s = self.heads(h)
        V = s[:, 0]
        slots = s[:, 1:]
        slots = torch.where(self.diag_mask, F.softplus(slots), slots)
        L = q.new_zeros(q.shape[0], self.n_dof, self.n_dof)
        L[:, self.tril_rows, self.tril_cols] = slots
        return V, L

    def hamiltonian(self, q, p, mu):
        if not self.learn_kinetic:
            h = self.embed(torch.cat([q, mu], dim=-1))
            for blk in self.blocks:
                h = blk(h)
            return 0.5 * (p ** 2).sum(dim=-1) + self.heads(h)[:, 0]
        V, L = self.V_and_L(q, mu)
        return kinetic_from_factor(L, p, self.mass_param) + V
