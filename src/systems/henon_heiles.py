"""Hénon-Heiles (parametrized): H = ½|p|² + ½|q|² + λ(q₁²q₂ − ⅓q₂³).

Scaling relation H(q,p;λ) = λ⁻²H₁(λq,λp) gives the escape energy E_esc(λ) = 1/(6λ²) and
the saddle radius |q| = 1/λ.

The potential is unbounded below outside the confined region. Permanent confinement =
energy window (E ≤ 0.75·E_esc) AND start point inside the central well (ic_valid:
|λq| ≤ 0.9): escape requires crossing a saddle with V_saddle ≥ E_esc > H, so energy
conservation keeps the trajectory in the well.
"""
import torch

from .base import HamiltonianSystem, register


@register
class HenonHeiles(HamiltonianSystem):
    name = "henon_heiles"
    dim = 2
    mu_dim = 1  # μ = (λ,)

    ETA_CAP = 0.75
    ETA_FLOOR = 0.05
    WELL_RADIUS = 0.9  # fraction of the saddle radius 1/λ (compared as λ|q|), with margin

    def hamiltonian(self, q, p, mu):
        q1, q2 = q[:, 0], q[:, 1]
        lam = mu[:, 0]
        T = 0.5 * (p ** 2).sum(dim=-1)
        V = 0.5 * (q ** 2).sum(dim=-1) + lam * (q1 ** 2 * q2 - q2 ** 3 / 3.0)
        return T + V

    def grad_q(self, q, p, mu):
        q1, q2 = q[:, 0], q[:, 1]
        lam = mu[:, 0]
        g1 = q1 + 2.0 * lam * q1 * q2
        g2 = q2 + lam * (q1 ** 2 - q2 ** 2)
        return torch.stack([g1, g2], dim=-1)

    def grad_p(self, q, p, mu):
        return p

    def equilibrium(self):
        return torch.zeros(2), torch.zeros(2)

    def default_mu_ranges(self):
        # after the ±30% range-width extrap extension λ ∈ [0.68, 1.32]; E_esc = 1/(6λ²)
        return [(0.8, 1.2)]

    def phase_space_box(self):
        # proposal box: covers the well radius 0.9/0.68≈1.32 at λ=0.68; p is clipped by the energy window
        return [(-1.35, 1.35), (-1.35, 1.35), (-0.75, 0.75), (-0.75, 0.75)]

    def energy_window(self, mu):
        e_esc = 1.0 / (6.0 * mu[:, 0] ** 2)
        e_max = self.ETA_CAP * e_esc
        return self.ETA_FLOOR * e_max, e_max

    def ic_valid(self, q, p, mu):
        return mu[:, 0] * q.norm(dim=-1) <= self.WELL_RADIUS
