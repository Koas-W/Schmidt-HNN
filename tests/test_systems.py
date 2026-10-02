"""Mandatory system tests: autograd gradients match the analytic gradients at random points with atol=1e-6.

All in float64: a 1e-6 tolerance is meaningless in float32.
"""
import unittest

import torch

from src.systems import available_systems, get_system

torch.set_default_dtype(torch.float64)

B = 256  # number of random test points


def sample_points(system, seed=0):
    """Sample a batch of (q, p, mu) uniformly inside phase_space_box x default_mu_ranges."""
    g = torch.Generator().manual_seed(seed)
    box = torch.tensor(system.phase_space_box())  # (2n, 2)
    lo, hi = box[:, 0], box[:, 1]
    x = lo + (hi - lo) * torch.rand(B, 2 * system.dim, generator=g)
    q, p = x[:, : system.dim], x[:, system.dim:]
    if system.mu_dim:
        mr = torch.tensor(system.default_mu_ranges())  # (m, 2)
        mu = mr[:, 0] + (mr[:, 1] - mr[:, 0]) * torch.rand(B, system.mu_dim, generator=g)
    else:
        mu = torch.zeros(B, 0)
    return q, p, mu


class TestAnalyticGradients(unittest.TestCase):
    """autograd dH/dq, dH/dp vs analytic grad_q, grad_p."""

    def _check(self, name):
        system = get_system(name)
        q, p, mu = sample_points(system)
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = system.hamiltonian(q, p, mu)
        gq_auto, gp_auto = torch.autograd.grad(H.sum(), (q, p))
        gq_ana = system.grad_q(q.detach(), p.detach(), mu)
        gp_ana = system.grad_p(q.detach(), p.detach(), mu)
        self.assertTrue(torch.allclose(gq_auto, gq_ana, atol=1e-6),
                        f"{name}: analytic dH/dq disagrees with autograd, "
                        f"max|Δ|={(gq_auto - gq_ana).abs().max():.3e}")
        self.assertTrue(torch.allclose(gp_auto, gp_ana, atol=1e-6),
                        f"{name}: analytic dH/dp disagrees with autograd, "
                        f"max|Δ|={(gp_auto - gp_ana).abs().max():.3e}")

    def test_all_registered_systems(self):
        for name in available_systems():
            with self.subTest(system=name):
                self._check(name)


class TestConventions(unittest.TestCase):
    """Sign conventions and basic physical properties."""

    def test_sign_convention_pendulum(self):
        # at q>0, p=0: qdot=0 and pdot=-sin q<0 (restoring force points to the equilibrium)
        system = get_system("pendulum")
        q = torch.tensor([[0.5]])
        p = torch.zeros(1, 1)
        mu = torch.zeros(1, 0)
        qdot, pdot = system.vector_field(q, p, mu)
        self.assertAlmostEqual(qdot.item(), 0.0)
        self.assertLess(pdot.item(), 0.0)

    def test_potential_zero_at_equilibrium(self):
        for name in available_systems():
            with self.subTest(system=name):
                system = get_system(name)
                q_eq, p_eq = system.equilibrium()
                if system.mu_dim:
                    mr = torch.tensor(system.default_mu_ranges())
                    mu = mr.mean(dim=1).unsqueeze(0)
                else:
                    mu = torch.zeros(1, 0)
                H_eq = system.hamiltonian(q_eq.unsqueeze(0), p_eq.unsqueeze(0), mu)
                self.assertAlmostEqual(H_eq.item(), 0.0, places=12,
                                       msg=f"{name}: V(q_eq) ≠ 0")


class TestDoublePendulumMass(unittest.TestCase):
    """Double pendulum: 2x2 analytic inverse vs torch.linalg.inv; positive definiteness."""

    def setUp(self):
        self.system = get_system("double_pendulum")
        self.q, self.p, self.mu = sample_points(self.system, seed=1)

    def test_analytic_inverse(self):
        M = self.system.mass_matrix(self.q, self.mu)
        A11, A12, A22 = self.system.minv(self.q, self.mu)
        A = torch.stack([torch.stack([A11, A12], -1),
                         torch.stack([A12, A22], -1)], -2)
        self.assertTrue(torch.allclose(A, torch.linalg.inv(M), atol=1e-10))

    def test_positive_definite(self):
        M = self.system.mass_matrix(self.q, self.mu)
        eigvals = torch.linalg.eigvalsh(M)
        self.assertGreater(eigvals.min().item(), 0.0)
        # condition number bounded on the sampling box + default mu ranges (docstring states 1-alpha*c^2 >= 0.26)
        cond = eigvals[:, 1] / eigvals[:, 0]
        self.assertLess(cond.max().item(), 50.0)


class TestHenonHeiles2P(unittest.TestCase):
    """Asymmetric two-parameter HH (Eq. 7): symmetric reduction of the escape threshold + analytic saddles with grad V = 0."""

    def setUp(self):
        self.sys = get_system("henon_heiles_2p")

    def test_symmetric_reduction(self):
        # lam1=lam2=lam => E_esc=1/(6 lam^2), d_min=1/lam (reduces to the one-parameter formula)
        lam = torch.tensor([0.2, 0.5, 0.8, 1.0])
        mu = torch.stack([lam, lam], dim=-1)
        e_esc, d_min = self.sys.escape(mu)
        self.assertTrue(torch.allclose(e_esc, 1.0 / (6.0 * lam ** 2), atol=1e-12))
        self.assertTrue(torch.allclose(d_min, 1.0 / lam, atol=1e-12))

    def test_saddles_are_critical(self):
        # dV/dq = 0 at the analytic saddles A and B (checked with analytic grad_q; independent of p)
        l1 = torch.tensor([0.3, 0.6, 0.9])
        l2 = torch.tensor([0.8, 0.4, 1.0])
        mu = torch.stack([l1, l2], dim=-1)
        p = torch.zeros(3, 2)
        q_a = torch.stack([torch.zeros_like(l2), 1.0 / l2], dim=-1)
        self.assertTrue(torch.allclose(
            self.sys.grad_q(q_a, p, mu), torch.zeros(3, 2), atol=1e-12))
        q1 = torch.sqrt(1.0 / (2 * l1 ** 2) + l2 / (4 * l1 ** 3))
        q_b = torch.stack([q1, -1.0 / (2 * l1)], dim=-1)
        self.assertTrue(torch.allclose(
            self.sys.grad_q(q_b, p, mu), torch.zeros(3, 2), atol=1e-12))

    def test_saddle_energy_matches_escape(self):
        # the energy of H at the saddle (p=0) must equal the saddle height returned by escape()
        l1 = torch.tensor([0.3, 0.7]); l2 = torch.tensor([0.6, 0.5])
        mu = torch.stack([l1, l2], dim=-1)
        p = torch.zeros(2, 2)
        q_a = torch.stack([torch.zeros_like(l2), 1.0 / l2], dim=-1)
        v_a = self.sys.hamiltonian(q_a, p, mu)
        self.assertTrue(torch.allclose(v_a, 1.0 / (6.0 * l2 ** 2), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
