"""Hénon-Heiles systems following the AHNN protocol (Han et al. 2021, arXiv:2102.13235).

Kept separate from the project's own henon_heiles system (λ∈[0.8,1.2], energy window
0.75·E_esc); these match the AHNN paper protocol and do not affect the other experiments.

Two systems:
- ``henon_heiles_ahnn``: single-parameter flagship (paper Sec. III, Eq. 3). Hamiltonian and
  gradients identical to HenonHeiles; only (a) the training grid λ∈{0.2,0.4,0.6,0.8} (the paper's
  "four green pentagons", default_mu_ranges=[(0.2,0.8)] + n_train_per_dim=4 -> linspace) and
  (b) the energy convention differ: the paper fixes the working energy E=1/6 and requires the
  initial "energy below the escape threshold", so E_cap = min(1/6, η·E_esc(λ)) with η=0.95
  (safety margin near the threshold to avoid marginal escape). For λ≤~0.9 the working energy
  1/6 < E_esc and E_cap=1/6 (full coverage of the {V<1/6} evaluation region); as λ→1,
  E_esc→1/6 and E_cap falls back to 0.95·E_esc.

- ``henon_heiles_2p``: asymmetric two-parameter version (paper Sec. IV.B, Eq. 7)
      H = ½(p₁²+p₂²) + ½(q₁²+q₂²) + λ₁ q₁²q₂ − (λ₂/3) q₂³
  λ₁=λ₂ reduces to the symmetric HH (Eq. 3). The paper determines the escape threshold
  numerically; here the saddle points are analytic:
    ∂V/∂q₁ = q₁(1 + 2λ₁q₂) = 0,  ∂V/∂q₂ = q₂ + λ₁q₁² − λ₂q₂² = 0
    - saddle A: (0, 1/λ₂), V_A = 1/(6λ₂²), |q|=1/λ₂
    - saddles B±: (±q₁*, −1/(2λ₁)), q₁*² = 1/(2λ₁²)+λ₂/(4λ₁³),
              V_B = 1/(8λ₁²)+λ₂/(24λ₁³), |q| = √(q₁*²+1/(4λ₁²))
  Escape threshold E_esc = min(V_A, V_B); inscribed-sphere radius of the central well
  d_min = min(|q|_A, |q|_B). Symmetric reduction λ₁=λ₂=λ: V_A=V_B=1/(6λ²), d_min=1/λ
  (consistent with the single-parameter system; see tests).
"""
import torch

from .base import HamiltonianSystem, register
from .henon_heiles import HenonHeiles

E_WORK = 1.0 / 6.0  # AHNN fixed working energy


@register
class HenonHeilesAHNN(HenonHeiles):
    name = "henon_heiles_ahnn"

    ETA_CAP = 0.95    # safety margin below the escape threshold (covers the E=1/6 evaluation region; falls back as λ→1)
    ETA_FLOOR = 0.05  # excludes near-equilibrium zero-signal trajectories (same convention as the other systems)

    def default_mu_ranges(self):
        # λ ∈ {0.2, 0.4, 0.6, 0.8} (n_train_per_dim=4 -> linspace(0.2,0.8,4))
        return [(0.2, 0.8)]

    def energy_window(self, mu):
        e_esc = 1.0 / (6.0 * mu[:, 0] ** 2)
        e_cap = torch.minimum(torch.full_like(e_esc, E_WORK),
                              self.ETA_CAP * e_esc)
        return self.ETA_FLOOR * e_cap, e_cap

    def eval_domain_threshold(self, mu):
        """Threshold of the ⟨ΔV⟩ evaluation domain {V_real<threshold} (paper: V<1/6; the escape threshold when lower)."""
        e_esc = 1.0 / (6.0 * mu[:, 0] ** 2)
        return torch.minimum(torch.full_like(e_esc, E_WORK), e_esc)

    def working_energy(self):
        """Fixed working energy E=1/6 (common scale of the robust normalized error over the whole domain)."""
        return E_WORK


@register
class HenonHeiles2P(HamiltonianSystem):
    """Asymmetric two-parameter Hénon-Heiles (paper Eq. 7), μ = (λ₁, λ₂)."""

    name = "henon_heiles_2p"
    dim = 2
    mu_dim = 2

    ETA_CAP = 0.95
    ETA_FLOOR = 0.05
    WELL_MARGIN = 0.9  # margin on the central-well inscribed-sphere radius (same 0.9 as the single-parameter system)

    def hamiltonian(self, q, p, mu):
        q1, q2 = q[:, 0], q[:, 1]
        l1, l2 = mu[:, 0], mu[:, 1]
        T = 0.5 * (p ** 2).sum(dim=-1)
        V = 0.5 * (q ** 2).sum(dim=-1) + l1 * q1 ** 2 * q2 - (l2 / 3.0) * q2 ** 3
        return T + V

    def grad_q(self, q, p, mu):
        q1, q2 = q[:, 0], q[:, 1]
        l1, l2 = mu[:, 0], mu[:, 1]
        g1 = q1 + 2.0 * l1 * q1 * q2
        g2 = q2 + l1 * q1 ** 2 - l2 * q2 ** 2
        return torch.stack([g1, g2], dim=-1)

    def grad_p(self, q, p, mu):
        return p

    def equilibrium(self):
        return torch.zeros(2), torch.zeros(2)

    def default_mu_ranges(self):
        # 4×4 grid: λ₁,λ₂ ∈ {0.2,0.4,0.6,0.8} (n_train_per_dim=4, Cartesian product over both dims)
        return [(0.2, 0.8), (0.2, 0.8)]

    def phase_space_box(self):
        # covers the E=1/6 reachable region (½|q|²≲1/6 gives |q|≲0.58; the cubic term stretches the q₂<0 lobe to ~1)
        return [(-1.35, 1.35), (-1.35, 1.35), (-0.75, 0.75), (-0.75, 0.75)]

    def escape(self, mu):
        """Analytic escape threshold E_esc and nearest-saddle distance d_min, each (B,)."""
        l1, l2 = mu[:, 0], mu[:, 1]
        v_a = 1.0 / (6.0 * l2 ** 2)               # saddle A height
        d_a = 1.0 / l2                            # saddle A distance
        v_b = 1.0 / (8.0 * l1 ** 2) + l2 / (24.0 * l1 ** 3)   # saddle B height
        q1sq = 1.0 / (2.0 * l1 ** 2) + l2 / (4.0 * l1 ** 3)
        d_b = torch.sqrt(q1sq + 1.0 / (4.0 * l1 ** 2))        # saddle B distance
        return torch.minimum(v_a, v_b), torch.minimum(d_a, d_b)

    def energy_window(self, mu):
        e_esc, _ = self.escape(mu)
        e_cap = torch.minimum(torch.full_like(e_esc, E_WORK),
                              self.ETA_CAP * e_esc)
        return self.ETA_FLOOR * e_cap, e_cap

    def eval_domain_threshold(self, mu):
        """⟨ΔV⟩ evaluation-domain threshold = min(1/6, E_esc(λ₁,λ₂)) (paper: use the escape threshold when it is below 1/6)."""
        e_esc, _ = self.escape(mu)
        return torch.minimum(torch.full_like(e_esc, E_WORK), e_esc)

    def working_energy(self):
        """Fixed working energy E=1/6 (common scale of the robust normalized error over the whole domain)."""
        return E_WORK

    def ic_valid(self, q, p, mu):
        _, d_min = self.escape(mu)
        return q.norm(dim=-1) <= self.WELL_MARGIN * d_min
