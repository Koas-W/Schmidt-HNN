"""Double pendulum (dimensionless): representative system with M depending on (q;μ).

    H(q,p;α,β) = ½ pᵀ M̃⁻¹(q;α,β) p + (1−cos q₁) + αβ(1−cos q₂)

    M̃ = [ 1      αβ c ]    c = cos(q₁−q₂), α = m₂/(m₁+m₂) ∈ (0,1), β = l₂/l₁
         [ αβ c   αβ²  ]

det M̃ = αβ²(1−αc²), strictly positive for 0<α<1, so M̃ is always symmetric positive definite;
the condition number degrades as α→1 and |c|→1. The default μ range (α ≤ 0.65, ≤ 0.74 after
the extrap extension) guarantees 1−αc² ≥ 0.26.
"""
import torch

from .base import HamiltonianSystem, register


@register
class DoublePendulum(HamiltonianSystem):
    name = "double_pendulum"
    dim = 2
    mu_dim = 2  # μ = (α, β)

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _unpack(q, p, mu):
        return (q[:, 0], q[:, 1], p[:, 0], p[:, 1], mu[:, 0], mu[:, 1])

    def mass_matrix(self, q, mu):
        """M̃(q;α,β), returns (B,2,2). Used for tests and condition-number checks."""
        alpha, beta = mu[:, 0], mu[:, 1]
        c = torch.cos(q[:, 0] - q[:, 1])
        ab_c = alpha * beta * c
        row1 = torch.stack([torch.ones_like(ab_c), ab_c], dim=-1)
        row2 = torch.stack([ab_c, alpha * beta**2], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def minv(self, q, mu):
        """Analytic inverse M̃⁻¹ = N/d, returns (A11, A12, A22), each (B,)."""
        alpha, beta = mu[:, 0], mu[:, 1]
        c = torch.cos(q[:, 0] - q[:, 1])
        d = alpha * beta**2 * (1.0 - alpha * c**2)
        A11 = alpha * beta**2 / d
        A12 = -alpha * beta * c / d
        A22 = 1.0 / d
        return A11, A12, A22

    # ---- interface -------------------------------------------------------
    def hamiltonian(self, q, p, mu):
        q1, q2, p1, p2, alpha, beta = self._unpack(q, p, mu)
        A11, A12, A22 = self.minv(q, mu)
        T = 0.5 * (A11 * p1**2 + 2.0 * A12 * p1 * p2 + A22 * p2**2)
        V = (1.0 - torch.cos(q1)) + alpha * beta * (1.0 - torch.cos(q2))
        return T + V

    def grad_p(self, q, p, mu):
        p1, p2 = p[:, 0], p[:, 1]
        A11, A12, A22 = self.minv(q, mu)
        return torch.stack([A11 * p1 + A12 * p2, A12 * p1 + A22 * p2], dim=-1)

    def grad_q(self, q, p, mu):
        # ∂T/∂q_k = ½ pᵀ(dA/dc)p · ∂c/∂q_k, with
        # ½ pᵀ(dA/dc)p = (−αβ p₁p₂ d + α²β² c · pᵀNp) / d²
        q1, q2, p1, p2, alpha, beta = self._unpack(q, p, mu)
        delta = q1 - q2
        c, s = torch.cos(delta), torch.sin(delta)
        d = alpha * beta**2 * (1.0 - alpha * c**2)
        pNp = alpha * beta**2 * p1**2 - 2.0 * alpha * beta * c * p1 * p2 + p2**2
        half_p_dAdc_p = (-alpha * beta * p1 * p2 * d
                         + alpha**2 * beta**2 * c * pNp) / d**2
        gq1 = half_p_dAdc_p * (-s) + torch.sin(q1)
        gq2 = half_p_dAdc_p * s + alpha * beta * torch.sin(q2)
        return torch.stack([gq1, gq2], dim=-1)

    def equilibrium(self):
        return torch.zeros(2), torch.zeros(2)

    def default_mu_ranges(self):
        # train tier; the data pipeline extends extrap outward by 30% of the range width (α stays < 0.74 < 1)
        return [(0.35, 0.65), (0.7, 1.3)]

    def phase_space_box(self):
        # the sampling box is only a proposal; the real constraint is energy_window rejection sampling
        return [(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]

    # oscillation-region energy fraction and zero-signal floor (system attributes, not repeated in config)
    ETA_CAP = 0.75
    ETA_FLOOR = 0.05

    def energy_window(self, mu):
        """E_max = η·min(2, 2αβ): total energy below the rotation barrier of each angle,
        so energy conservation keeps q₁, q₂ oscillating forever (|q|<π, compact phase space).
        This matches the bounded-domain L² assumption and avoids numerical error
        amplification in the rotating chaotic regime (with plain box sampling q₂ drifted to −23 rad).
        E_min excludes near-equilibrium zero-signal trajectories.
        """
        barrier = 2.0 * torch.minimum(torch.ones_like(mu[:, 0]),
                                      mu[:, 0] * mu[:, 1])
        e_max = self.ETA_CAP * barrier
        return self.ETA_FLOOR * e_max, e_max
