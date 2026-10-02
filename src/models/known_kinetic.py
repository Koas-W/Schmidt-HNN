"""Known-kinetic wrappers.

The scientific question here is learning the potential: all parameters sit on the stiffness/load side and
the mass matrix is M = I for every mu. This knowledge must be given to all compared models alike; otherwise
only the Schmidt-HNN would receive the M = I hint and the injection comparison would be unclean. Two wrappers:

- KnownKineticHNN: any (q,mu) -> scalar potential network N is assembled as H_hat = 1/2|p|^2 + N(q,mu);
  the vector field still comes from the base-class autograd (dH_hat/dp = p holds exactly);
- KnownKineticODE: direct-regression control, q_dot = p fixed, only p_dot = f(q,mu) is regressed
  (the true p_dot = -dV/dq depends only on (q,mu), so the input excludes p: same information as the potential net).
"""
import torch
import torch.nn as nn

from .hamiltonian_base import HamiltonianModel


class TanhPotentialNet(nn.Module):
    """MLP taking the concatenation [q; mu] and outputting a scalar potential. Default tanh (as in AHNN);
    act="silu" is the "modern general-purpose MLP potential" baseline (unbounded smooth activation; SwiGLU is Swish gating,
    so silu_hnn differs from Schmidt-HNN only in concat vs separable product and additive vs multiplicative primitive)."""

    ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU}

    def __init__(self, n_dof: int, mu_dim: int, width: int, depth: int = 2, act: str = "tanh"):
        super().__init__()
        A = self.ACTS[act]
        layers: list[nn.Module] = [nn.Linear(n_dof + mu_dim, width), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), A()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, mu], dim=-1)).squeeze(-1)


class KnownKineticHNN(HamiltonianModel):
    """H_hat = 1/2|p|^2 + N(q, mu)."""

    def __init__(self, n_dof: int, potential: nn.Module):
        super().__init__()
        self.n_dof = n_dof
        self.potential = potential

    def hamiltonian(self, q, p, mu):
        return 0.5 * (p ** 2).sum(dim=-1) + self.potential(q, mu)


class KnownKineticODE(nn.Module):
    """q_dot = p fixed, p_dot = MLP(q, mu) regressed directly (no Hamiltonian structure)."""

    def __init__(self, n_dof: int, mu_dim: int, width: int, depth: int = 2):
        super().__init__()
        self.n_dof, self.mu_dim = n_dof, mu_dim
        layers: list[nn.Module] = [nn.Linear(n_dof + mu_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, n_dof))
        self.net = nn.Sequential(*layers)

    def time_derivative(self, x: torch.Tensor, mu: torch.Tensor,
                        create_graph: bool = False) -> torch.Tensor:
        n = self.n_dof
        q, p = x[:, :n], x[:, n:]
        return torch.cat([p, self.net(torch.cat([q, mu], dim=-1))], dim=-1)
