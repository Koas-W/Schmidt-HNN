"""Adaptable HNN baseline (Han, Glaz, Haile, Lai, Phys. Rev. Research 3, 023156, 2021).

The parameter mu enters the first hidden layer together with (q,p) through a dedicated input channel
(concat injection); a single network regresses the scalar H_hat and the vector field is built by autograd
(checked against the paper: 2 hidden tanh layers, derivative-matching loss).

With mu_dim=0 it reduces to the original HNN (Greydanus 2019). Central comparison of this work: concat injection
vs the separable product of Schmidt-HNN (the two ends of the injection ablation axis).
"""
import torch
import torch.nn as nn

from ..hamiltonian_base import HamiltonianModel


class AHNN(HamiltonianModel):

    ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU}

    def __init__(self, n_dof: int, mu_dim: int, width: int, depth: int = 2, act: str = "tanh"):
        super().__init__()
        self.n_dof, self.mu_dim = n_dof, mu_dim
        layers: list[nn.Module] = [nn.Linear(2 * n_dof + mu_dim, width),
                                   self.ACTS[act]()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), self.ACTS[act]()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor,
                    mu: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, p, mu], dim=-1)).squeeze(-1)
