"""MLP-ODE baseline: (q,p,mu) -> MLP -> (q_dot_hat, p_dot_hat), direct regression of the vector field, no structural prior.

Network style follows the HNN/AHNN literature convention: tanh activation, 2 hidden layers (width is tuned, not depth).
"""
import torch
import torch.nn as nn


class MLPODE(nn.Module):

    def __init__(self, n_dof: int, mu_dim: int, width: int, depth: int = 2):
        super().__init__()
        self.n_dof, self.mu_dim = n_dof, mu_dim
        layers: list[nn.Module] = [nn.Linear(2 * n_dof + mu_dim, width),
                                   nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2 * n_dof))
        self.net = nn.Sequential(*layers)

    def time_derivative(self, x: torch.Tensor, mu: torch.Tensor,
                        create_graph: bool = False) -> torch.Tensor:
        return self.net(torch.cat([x, mu], dim=-1))
