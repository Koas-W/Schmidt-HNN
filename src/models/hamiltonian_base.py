"""Common base class for Hamiltonian-type models: symplectic vector field from a scalar H_hat via autograd.

Unified model interface (training and evaluation depend only on it):
    time_derivative(x, mu, create_graph) : (B,2n),(B,m) -> (B,2n)
where x = (q||p), output = (q_dot_hat||p_dot_hat), q_dot_hat = dH_hat/dp, p_dot_hat = -dH_hat/dq.
"""
import torch
import torch.nn as nn


class HamiltonianModel(nn.Module):
    """Subclasses only implement hamiltonian(q, p, mu) -> (B,)."""

    n_dof: int

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor,
                    mu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def time_derivative(self, x: torch.Tensor, mu: torch.Tensor,
                        create_graph: bool = False) -> torch.Tensor:
        n = self.n_dof
        with torch.enable_grad():
            q = x[:, :n].detach().requires_grad_(True)
            p = x[:, n:].detach().requires_grad_(True)
            H = self.hamiltonian(q, p, mu)
            gq, gp = torch.autograd.grad(H.sum(), (q, p),
                                         create_graph=create_graph)
        if not create_graph:
            gq, gp = gq.detach(), gp.detach()
        return torch.cat([gp, -gq], dim=-1)


def kinetic_from_factor(L: torch.Tensor, p: torch.Tensor, mass_param: str = "inverse") -> torch.Tensor:
    """Assemble the kinetic energy from a lower-triangular factor L (switch for which mass object is represented):
    "inverse": M^-1 = L L^T (inherited from SymODEN), T = 1/2||L^T p||^2;
    "direct":  M = L L^T (M assembled additively over bodies/elements, entries are parameter monomials times functions of q),
               T = 1/2 p^T M^-1 p = 1/2||L^-1 p||^2, obtained by triangular solve."""
    if mass_param == "inverse":
        LTp = torch.einsum("bij,bi->bj", L, p)
        return 0.5 * (LTp ** 2).sum(dim=-1)
    if mass_param == "direct":
        y = torch.linalg.solve_triangular(L, p.unsqueeze(-1), upper=False).squeeze(-1)
        return 0.5 * (y ** 2).sum(dim=-1)
    raise ValueError(f"unknown mass_param={mass_param!r}")
