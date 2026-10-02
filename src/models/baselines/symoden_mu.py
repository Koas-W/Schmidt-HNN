"""SymODEN-mu: faithful adaptation of the literature baseline (Zhong, Dey, Chakraborty, ICLR 2020 + mu concat).

Points kept faithful to the official code (Physics-aware-AI/Symplectic-ODENet):
- M^-1(q,mu) comes from a PSD net: the network outputs a lower-triangular L and M^-1 = L L^T + 0.1*I
  (diagonal positivity via the +0.1I offset, as in the PSD class of nn_models.py; not the softplus scheme used here);
- V(q,mu) is a separate MLP; both nets are 3 layers (2 hidden) tanh + orthogonal init (as in the MLP class);
- autograd of H_hat builds the symplectic vector field with the same sign convention.

Declared deviations (protocol uniformity first):
- the paper trains by trajectory matching through torchdiffeq odeint; all models here use derivative matching;
- the paper has no mu: here mu is concatenated with q as input to both nets (the mu-concat adaptation).

Factor placement: Cholesky(+0.1I variant) x concat x tanh-MLP, i.e. it differs from Schmidt-HNN in two factors
(injection scheme + block type); pshnn_concat isolates a single factor.
"""
import torch
import torch.nn as nn

from ..hamiltonian_base import HamiltonianModel


def _mlp3(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Official SymODEN MLP: 3 linear layers, tanh, orthogonal init."""
    layers = [nn.Linear(in_dim, hidden), nn.Tanh(),
              nn.Linear(hidden, hidden), nn.Tanh(),
              nn.Linear(hidden, out_dim)]
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


class SymODENmu(HamiltonianModel):

    EPS_DIAG = 0.1  # positive-definite offset of the official PSD class

    def __init__(self, n_dof: int, mu_dim: int, width: int, depth: int = 2):
        super().__init__()
        assert depth == 2, "faithful replication of the official 3-layer (2 hidden) structure"
        self.n_dof, self.mu_dim = n_dof, mu_dim
        n_L = n_dof * (n_dof + 1) // 2
        self.M_net = _mlp3(n_dof + mu_dim, width, n_L)
        self.V_net = _mlp3(n_dof + mu_dim, width, 1)
        tril = torch.tril_indices(n_dof, n_dof)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])

    def minv(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """M^-1 = L L^T + 0.1*I, returns (B,n,n)."""
        slots = self.M_net(torch.cat([q, mu], dim=-1))
        L = q.new_zeros(q.shape[0], self.n_dof, self.n_dof)
        L[:, self.tril_rows, self.tril_cols] = slots
        eye = torch.eye(self.n_dof, dtype=q.dtype, device=q.device)
        return L @ L.transpose(-1, -2) + self.EPS_DIAG * eye

    def hamiltonian(self, q, p, mu):
        Minv = self.minv(q, mu)
        T = 0.5 * torch.einsum("bi,bij,bj->b", p, Minv, p)
        V = self.V_net(torch.cat([q, mu], dim=-1)).squeeze(-1)
        return T + V
