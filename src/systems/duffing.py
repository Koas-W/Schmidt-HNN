"""Duffing oscillator (dimensionless, hardening spring): H = ½p² + ½q² + (μ/4)q⁴, μ>0.

For μ>0 the potential is globally confining (no escape, no saddle), so no energy window
is needed; the sampling box is the only constraint.
"""
import torch

from .base import HamiltonianSystem, register


@register
class Duffing(HamiltonianSystem):
    name = "duffing"
    dim = 1
    mu_dim = 1  # μ = quartic coefficient

    def hamiltonian(self, q, p, mu):
        return (0.5 * p[:, 0] ** 2 + 0.5 * q[:, 0] ** 2
                + 0.25 * mu[:, 0] * q[:, 0] ** 4)

    def grad_q(self, q, p, mu):
        return (q[:, 0] + mu[:, 0] * q[:, 0] ** 3).unsqueeze(-1)

    def grad_p(self, q, p, mu):
        return p

    def equilibrium(self):
        return torch.zeros(1), torch.zeros(1)

    def default_mu_ranges(self):
        # after the ±30% range-width extrap extension: [0.44, 2.36], strictly positive (hardening)
        return [(0.8, 2.0)]

    def phase_space_box(self):
        return [(-1.2, 1.2), (-1.2, 1.2)]
