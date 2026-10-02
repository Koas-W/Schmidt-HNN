"""Ablation arm: Sep-HNN, separable product injection but without the Cholesky kinetic structure.

The "no kinetic structure x separable injection x SwiGLU" cell of the factor cube:
- same as Schmidt-HNN: Param-Net(mu) -> {u_k}, SwiGLU+Residual trunk, channel-wise product
  and sum, common protocol and parameter matching (+-5%);
- different: the State-Net takes the whole state (q,p) and a single head outputs state factors {w_k},
  H_hat = sum_k u_k(mu) * w_k(q,p); no V/L decomposition, no positive-definite quadratic kinetic term guaranteed.

Comparison logic: Schmidt-HNN - Sep-HNN = net contribution of the Cholesky structure under separable injection;
Sep-HNN - AHNN ~ contribution of the injection scheme without kinetic structure (the block factor is
mixed in; pshnn_concat gives the exact isolation).
"""
import torch

from .blocks import FeatureTrunk
from .hamiltonian_base import HamiltonianModel
from .param_net import build_param_net


class SepHNN(HamiltonianModel):

    def __init__(self, n_dof: int, mu_dim: int, width: int,
                 depth: int = 2, R: int = 8, block: str = "swiglu",
                 learn_kinetic: bool = True):
        super().__init__()
        self.n_dof, self.mu_dim, self.R = n_dof, mu_dim, R
        self.learn_kinetic = learn_kinetic
        # Single head (factors of H_hat directly); the State-Net input is the full state (q,p).
        # learn_kinetic=False (known-kinetic wrapper): the State-Net takes only q and 1/2|p|^2 is added to H_hat
        state_dim = 2 * n_dof if learn_kinetic else n_dof
        self.state_net = FeatureTrunk(state_dim, width, depth,
                                      head_ranks=[R], block=block)
        self.param_net = build_param_net(mu_dim, width, depth,
                                         head_ranks=[R], block=block)

    def hamiltonian(self, q, p, mu):
        u = self.param_net(mu)                          # (B,R)
        if not self.learn_kinetic:
            return 0.5 * (p ** 2).sum(dim=-1) + (u * self.state_net(q)).sum(dim=-1)
        w = self.state_net(torch.cat([q, p], dim=-1))  # (B,R)
        return (u * w).sum(dim=-1)
