"""Simple pendulum (dimensionless): H = ½ p² + (1 − cos q).

No free parameter after nondimensionalization (mu_dim = 0); serves as the degenerate case
of the pipeline and as a sanity-check system, not used in the μ-generalization experiments.
"""
import torch

from .base import HamiltonianSystem, register


@register
class Pendulum(HamiltonianSystem):
    name = "pendulum"
    dim = 1
    mu_dim = 0

    def hamiltonian(self, q, p, mu=None):
        return 0.5 * p[:, 0] ** 2 + (1.0 - torch.cos(q[:, 0]))

    def grad_q(self, q, p, mu=None):
        return torch.sin(q)

    def grad_p(self, q, p, mu=None):
        return p

    def equilibrium(self):
        return torch.zeros(1), torch.zeros(1)

    def default_mu_ranges(self):
        return []

    def phase_space_box(self):
        # max energy inside the box (1−cos 1.8) + ½·1.0² ≈ 1.73 < 2 (separatrix),
        # so all initial conditions lie in the oscillation region and never cross the homoclinic orbit.
        return [(-1.8, 1.8), (-1.0, 1.0)]
