"""Acceptance gates for the clamped-clamped beam system: modal basis, gradients, positive definiteness, condensation, envelope theorem.

All float64. The modal-count convergence and integrator checks belong to the data generation stage and are not in this file.
"""
import math
import unittest

import torch

from src.systems import get_system
from src.systems.clamped_beam import (N_CR, SQRT12, ClampedBeamBase,
                                      beta_roots)

torch.set_default_dtype(torch.float64)

# textbook values (first 6 roots of cos(beta) cosh(beta) = 1 for the clamped-clamped beam)
BETA_REF = [4.730041, 7.853205, 10.995608, 14.137165, 17.278760, 20.420352]


def _mu_box(system, n, seed=0, expand=0.0):
    """Sample uniformly inside default_mu_ranges (optionally widened outward by expand times the range width)."""
    g = torch.Generator().manual_seed(seed)
    mr = torch.tensor(system.default_mu_ranges())
    lo, hi = mr[:, 0], mr[:, 1]
    w = hi - lo
    lo, hi = lo - expand * w, hi + expand * w
    return lo + (hi - lo) * torch.rand(n, system.mu_dim, generator=g)


class TestT1Modes(unittest.TestCase):
    """Modal basis: correctness and numerical stability."""

    def test_beta_matches_textbook(self):
        b = beta_roots(6)
        for i, ref in enumerate(BETA_REF):
            self.assertAlmostEqual(float(b[i]), ref, places=5,
                                   msg=f"β_{i+1} disagrees with the textbook value")

    def test_beta_asymptotic(self):
        b = beta_roots(12)
        for i in range(6, 12):
            self.assertAlmostEqual(float(b[i]), (2 * (i + 1) + 1) * math.pi / 2,
                                   places=6, msg="higher roots should approach (2i+1)pi/2")

    def test_orthonormality(self):
        """int phi_i phi_j = delta_ij in the numerically stable form including high modes (criterion <1e-10)."""
        for n_total in (8, 16, 24):
            s = ClampedBeamBase(n_total // 2, n_total // 2)
            err = float((s.norm - torch.eye(n_total)).abs().max())
            self.assertLess(err, 1e-10, f"N={n_total} orthogonality error {err:.2e}")

    def test_f_closed_form(self):
        """f_i = beta_i^-4 [phi_i'''(1) - phi_i'''(0)] (closed-form identity)."""
        s = ClampedBeamBase(4, 4)
        beta = s.beta
        x = torch.tensor([0.0, 1.0])
        # phi''' taken directly from the third-order term of the stable form
        from src.systems.clamped_beam import _sigma_and_one_minus
        sg, omv = _sigma_and_one_minus(beta)
        b = beta.unsqueeze(1)
        bx = b * x.unsqueeze(0)
        ep, em = torch.exp(bx), torch.exp(-bx)
        c, sn = torch.cos(bx), torch.sin(bx)
        d3 = b ** 3 * (0.5 * omv.unsqueeze(1) * ep - 0.5 * (1 + sg.unsqueeze(1)) * em
                       - sn - sg.unsqueeze(1) * c)
        f_closed = (d3[:, 1] - d3[:, 0]) / beta ** 4
        self.assertLess(float((f_closed - s.f).abs().max()), 1e-10)

    def test_parity_structure(self):
        """Cross-parity G_ij and f_i of antisymmetric modes vanish exactly (determines the initial-condition sampling)."""
        s = ClampedBeamBase(4, 4)
        n = s.n_total
        cross = max(abs(float(s.G[i, j])) for i in range(n) for j in range(n)
                    if (i % 2) != (j % 2))
        self.assertLess(cross, 1e-10, "G must be block-diagonal by parity")
        anti = max(abs(float(s.f[i])) for i in range(1, n, 2))
        self.assertLess(anti, 1e-12, "f_i of antisymmetric modes must vanish exactly")
        self.assertGreater(abs(float(s.f[0])), 0.1, "f_i of symmetric modes must not vanish")

    def test_buckling_load(self):
        """Single-mode Galerkin buckling estimate approaches 4 pi^2 from above."""
        s = ClampedBeamBase(4, 4)
        est = float(s.k_diag[0] / s.G[0, 0])
        self.assertGreater(est, N_CR)
        self.assertLess(est, 1.05 * N_CR, f"single-mode estimate {est:.2f} deviates too much from 4 pi^2")


class TestT2Gradients(unittest.TestCase):
    """Analytic gradients match autograd (project discipline atol=1e-6)."""

    def test_grad_matches_autograd(self):
        s = get_system("clamped_beam")
        mu = _mu_box(s, 64, seed=3)
        box = torch.tensor(s.phase_space_box())
        g = torch.Generator().manual_seed(4)
        x = box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(64, 2 * s.dim,
                                                             generator=g)
        q = x[:, :s.dim].clone().requires_grad_(True)
        p = x[:, s.dim:].clone().requires_grad_(True)
        H = s.hamiltonian(q, p, mu)
        gq, gp = torch.autograd.grad(H.sum(), (q, p))
        ga = s.grad_q(q.detach(), p.detach(), mu)
        gpa = s.grad_p(q.detach(), p.detach(), mu)
        rel = float(((gq - ga).abs() / (ga.abs() + 1e-8)).max())
        self.assertLess(rel, 1e-6, f"grad_q relative error {rel:.2e}")
        self.assertLess(float((gp - gpa).abs().max()), 1e-10)

    def test_kinetic_is_identity(self):
        """All three parameters enter the stiffness/load side => the mass matrix is the identity for every mu."""
        s = get_system("clamped_beam")
        mu = _mu_box(s, 16, seed=5)
        p = torch.randn(16, s.dim)
        q = torch.zeros(16, s.dim)
        self.assertTrue(torch.allclose(s.grad_p(q, p, mu), p))


class TestT3PositiveDefinite(unittest.TestCase):
    """Hessian positive definite with margin over the parameter box (including the 30% extrapolation range) x amplitude window."""

    def test_margin_over_extrapolated_box(self):
        s = ClampedBeamBase(4, 4)
        mu = _mu_box(s, 256, seed=7, expand=0.3)
        g = torch.Generator().manual_seed(8)
        # cover the top of the amplitude window: first-mode amplitude at 1.2x the value corresponding to peak|w|/h = 3
        scale = 3.0 * SQRT12 / float(s.phi_grid[0].abs().max()) * 1.2
        a = (torch.rand(256, 4, generator=g) * 2 - 1) * torch.tensor(
            [scale, scale / 3, scale / 6, scale / 10])
        b = s.condense(a, mu)
        H = s.hess_V_full(torch.cat([a, b], -1), mu)
        ev = torch.linalg.eigvalsh(H)
        self.assertGreater(float(ev.min()), 0.0, "Hessian loses positive definiteness on the extrapolation range")


class TestT4Condensation(unittest.TestCase):
    """Static condensation: convergence and uniqueness."""

    def test_residual_and_uniqueness(self):
        s = ClampedBeamBase(4, 4)
        mu = _mu_box(s, 64, seed=11)
        g = torch.Generator().manual_seed(12)
        a = torch.randn(64, 4, generator=g) * torch.tensor([3.0, 1.0, .5, .3])
        b_ref = s.condense(a, mu)
        r = s.grad_V_full(torch.cat([a, b_ref], -1), mu)[:, 4:]
        rel = float((r.abs() / (s.k_diag[4:] * (1 + b_ref.abs()))).max())
        self.assertLess(rel, 1e-12, f"Newton relative residual {rel:.2e} too large")
        for seed in (13, 14, 15):
            gg = torch.Generator().manual_seed(seed)
            b0 = torch.randn(64, 4, generator=gg) * 0.5
            b = s.condense(a, mu, b0=b0)
            self.assertLess(float((b - b_ref).abs().max()), 1e-9,
                            "different initial guesses converge to different b* (condensation not unique)")


class TestT7Envelope(unittest.TestCase):
    """Envelope theorem: dV_eff/da = dV_full/da at b=b*."""

    def test_vs_autograd_through_condensation(self):
        """Backpropagate through the condensation (Newton converged => equals the implicit-function derivative)."""
        s = ClampedBeamBase(4, 4)
        mu = _mu_box(s, 32, seed=17)
        g = torch.Generator().manual_seed(18)
        a = (torch.randn(32, 4, generator=g)
             * torch.tensor([3.0, 1.0, .5, .3])).requires_grad_(True)
        V = s.V_eff(a, mu)
        (ga,) = torch.autograd.grad(V.sum(), (a,))
        env = s.grad_q(a.detach(), torch.zeros_like(a.detach()), mu)
        rel = float(((ga - env).abs() / (env.abs() + 1e-8)).max())
        self.assertLess(rel, 1e-8, f"envelope theorem disagrees with autograd, relative error {rel:.2e}")

    def test_vs_central_difference(self):
        """Central-difference cross-check (limited by finite-difference rounding; criterion relaxed to relative 1e-6)."""
        s = ClampedBeamBase(4, 4)
        mu = _mu_box(s, 16, seed=19)
        g = torch.Generator().manual_seed(20)
        a = torch.randn(16, 4, generator=g) * torch.tensor([3.0, 1.0, .5, .3])
        env = s.grad_q(a, torch.zeros_like(a), mu)
        eps = 1e-5
        fd = torch.zeros_like(a)
        for i in range(4):
            e = torch.zeros_like(a)
            e[:, i] = eps
            fd[:, i] = (s.V_eff(a + e, mu) - s.V_eff(a - e, mu)) / (2 * eps)
        rel = float(((fd - env).abs() / (env.abs() + 1.0)).max())
        self.assertLess(rel, 1e-6, f"central-difference relative error {rel:.2e}")


class TestPhysicsSanity(unittest.TestCase):
    """Physical consistency: limits of g, hardening trend, static equilibrium."""

    def test_g_limits(self):
        """k_tilde -> 0: no membrane-force hardening; k_tilde -> inf: immovable ends (g -> 1)."""
        s = ClampedBeamBase(4, 4)
        mu = torch.tensor([[-6.0, 0.0, 0.5], [6.0, 0.0, 0.5]])
        gg, _, _ = s.decode_mu(mu)
        self.assertLess(float(gg[0]), 1e-5)
        self.assertGreater(float(gg[1]), 1 - 1e-5)

    def test_hardening_monotone(self):
        """Restoring force of the first mode grows superlinearly with amplitude (hardening), and stiff supports are harder than soft ones."""
        s = ClampedBeamBase(4, 4)
        amps = torch.tensor([0.5, 1.0, 2.0, 4.0]).unsqueeze(-1)
        a = torch.cat([amps, torch.zeros(4, 3)], -1)
        # mu3=0: remove the static pressure, otherwise the constant static load dominates the restoring force at small amplitude and the ratio is meaningless
        for m1, expect_hard in ((1.0, True), (-3.0, False)):
            mu = torch.tensor([[m1, 0.0, 0.0]]).repeat(4, 1)
            fq = s.grad_q(a, torch.zeros_like(a), mu)[:, 0]
            ratio = float(fq[-1] / fq[0] / 8.0)   # equals 1 in the linear case
            if expect_hard:
                self.assertGreater(ratio, 1.5, "stiff supports should show clear hardening")
            else:
                self.assertLess(ratio, 1.05, "soft supports should be close to linear")

    def test_static_equilibrium_is_stationary(self):
        s = get_system("clamped_beam")
        mu = _mu_box(s, 32, seed=21)
        a_s = s.static_equilibrium(mu)
        r = s.grad_q(a_s, torch.zeros_like(a_s), mu)
        self.assertLess(float(r.abs().max()), 1e-10)

    def test_sample_ic_hits_target_peak(self):
        """sample_ic (two-stage scaling): the measured short-window total peak lies in [lo_eff, AMP_HI] and approaches the stratified targets."""
        s = get_system("clamped_beam")
        box = torch.tensor(s.phase_space_box())
        from src.data.trajectories import sample_initial_conditions
        for mu in (torch.tensor([0.0, 0.0, 0.35]), torch.tensor([1.0, -0.5, 0.5])):
            g = torch.Generator().manual_seed(3)
            y0 = sample_initial_conditions(s, mu, 8, box[:, 0], box[:, 1], g)
            mr = mu.expand(8, -1)
            lo_eff, _, _ = s.amplitude_bounds(mu[None])
            pk = s._short_window_peak(y0[:, :s.dim], y0[:, s.dim:], mr)
            self.assertTrue(bool((pk >= float(lo_eff) - 0.05).all()), "peak below the usable lower bound")
            self.assertTrue(bool((pk <= s.AMP_HI + 1e-6).all()), "peak above the upper bound")
            # 8 stratified targets cover 8 bins of [lo_eff, AMP_HI - margin]: the measured peak range should be >= half the interval width
            self.assertGreater(float(pk.max() - pk.min()),
                               0.5 * (s.AMP_HI - s.IC_TARGET_MARGIN - float(lo_eff)))

    def test_energy_window_ordered(self):
        s = get_system("clamped_beam")
        mu = _mu_box(s, 64, seed=22)
        lo, hi = s.energy_window(mu)
        self.assertTrue(bool((lo < hi).all()), "energy window lower bound must be strictly below the upper bound")


if __name__ == "__main__":
    unittest.main()
