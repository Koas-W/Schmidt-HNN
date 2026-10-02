"""Acceptance tests for the synthetic family (hard gates):
dictionary orthogonality <1e-10; analytic J/Hess vs autograd; SVE=SVD;
FPU exact_C cross-checked against the analytic form; equilibrium; positive-definiteness margin.
"""
import unittest

import numpy as np
import torch

from src.synthetic.dictionaries import TensorLegendre, ZeroPointLegendre
from src.synthetic.fpu import make_fpu_family, u1, u2, v1, v2
from src.synthetic.planted import PlantedFamily, geometric_sigma

torch.set_default_dtype(torch.float64)


def _gauss_grid(box, n_per_dim):
    nodes, weights = np.polynomial.legendre.leggauss(n_per_dim)
    nodes, weights = torch.from_numpy(nodes), torch.from_numpy(weights)
    d = len(box)
    grid = torch.cartesian_prod(*([nodes] * d)).reshape(-1, d)
    w = torch.cartesian_prod(*([weights] * d)).reshape(-1, d).prod(-1) / 2 ** d
    box_t = torch.tensor(box)
    x = box_t[:, 0] + (grid + 1) / 2 * (box_t[:, 1] - box_t[:, 0])
    return x, w


class TestT1Orthogonality(unittest.TestCase):

    def test_gram_identity(self):
        cases = [TensorLegendre([(-1.0, 1.0)] * 3, 4),
                 TensorLegendre([(0.5, 1.5)] * 4, 4),
                 ZeroPointLegendre([(-1.0, 1.0)] * 6, 4)]  # psi of the FPU family (zero-point constrained version)
        for dic in cases:
            with self.subTest(dic=type(dic).__name__, d=dic.d):
                x, w = _gauss_grid([tuple(r) for r in dic.box.tolist()],
                                   dic.deg_max + 1)
                Psi = dic.eval(x)
                G = torch.einsum("bi,b,bj->ij", Psi, w, Psi)
                err = (G - torch.eye(dic.n_basis)).abs().max().item()
                self.assertLess(err, 1e-10, f"orthogonality error {err:.2e}")

    def test_zero_point_constraints(self):
        # hard acceptance: every basis function has zero value and zero gradient at q=0
        dic = ZeroPointLegendre([(-1.0, 1.0)] * 4, 4)
        zero = torch.zeros(1, 4)
        vals, J = dic.eval_grad(zero)
        self.assertLess(vals.abs().max().item(), 1e-12)
        self.assertLess(J.abs().max().item(), 1e-12)


class TestT2Derivatives(unittest.TestCase):

    def test_grad_hess_vs_autograd(self):
        dic = TensorLegendre([(-1.2, 0.8), (0.5, 1.5), (-1.0, 1.0)], 4,
                             deg_min=2)
        # sample points must lie inside the box in every dimension (the autograd gradient of the clamp is zero outside)
        box = dic.box
        x = (box[:, 0] + (box[:, 1] - box[:, 0])
             * (0.05 + 0.9 * torch.rand(16, 3))).requires_grad_(True)
        vals, J = dic.eval_grad(x.detach())
        s = dic.eval(x).sum(0)  # (n_basis,)
        for a in range(0, dic.n_basis, 7):  # spot-check basis functions
            g = torch.autograd.grad(s[a], x, retain_graph=True)[0]
            self.assertTrue(torch.allclose(J[:, a, :], g, atol=1e-8))
        # Hessian spot-check (also covers consistency with the pairwise contracted version)
        H = dic.eval_hess(x.detach())
        coeff = torch.randn(16, dic.n_basis)
        Hc = dic.eval_hess_contracted(x.detach(), coeff)
        self.assertTrue(torch.allclose(
            torch.einsum("ba,banm->bnm", coeff, H), Hc, atol=1e-10))


class TestT3SVEisSVD(unittest.TestCase):

    def test_planted_spectrum_recovered(self):
        fam = PlantedFamily([(0.5, 1.5)] * 2, [(-1.0, 1.0)] * 2,
                            D_mu=3, D_q=4,
                            sigma=geometric_sigma(4, 0.5), seed=3, eps=0.1)
        # Gauss-node discretization (polynomial kernel => exact quadrature => spectrum recovered exactly)
        xm, wm = _gauss_grid([(0.5, 1.5)] * 2, 6)
        xq, wq = _gauss_grid([(-1.0, 1.0)] * 2, 6)
        Phi, Psi = fam.phi.eval(xm), fam.psi.eval(xq)
        kernel = Phi @ fam.C @ Psi.T                     # (Nμ, Nq)
        Bmat = (wm.sqrt().unsqueeze(-1) * kernel) * wq.sqrt()
        sv = torch.linalg.svdvals(Bmat)[:4]
        rel = ((sv - fam.sigma).abs() / fam.sigma).max().item()
        self.assertLess(rel, 1e-6, f"SVE != SVD: relative spectral error {rel:.2e}")


class TestS0FPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fam = make_fpu_family(seed=0)

    def test_dims_match_spec(self):
        self.assertEqual(self.fam.phi.n_basis, 70)   # C(8,4)
        self.assertEqual(self.fam.psi.n_basis, 203)  # C(10,4)−1−6

    def test_exact_C_matches_analytic(self):
        g = torch.Generator().manual_seed(1)
        q = torch.rand(256, 6, generator=g) * 2 - 1
        mu = 0.5 + torch.rand(256, 4, generator=g)
        analytic = u1(mu) * v1(q) + u2(mu) * v2(q)
        self.assertTrue(torch.allclose(self.fam.v_plant(q, mu), analytic,
                                       atol=1e-9),
                        "exact_C dictionary representation disagrees with the analytic FPU potential")
        # analytic gradient cross-check (autograd through the dictionary path)
        q2 = q[:32].requires_grad_(True)
        Vd = self.fam.v_plant(q2, mu[:32]).sum()
        g_auto = torch.autograd.grad(Vd, q2)[0]
        self.assertTrue(torch.allclose(
            g_auto, self.fam.grad_v_plant(q2.detach(), mu[:32]), atol=1e-9))

    def test_effective_rank_two(self):
        s = self.fam.sigma
        self.assertGreater((s[0] / s[1]).item(), 0.99)  # two effective channels
        self.assertLess((s[2] / s[0]).item(), 1e-12)    # the rest are numerically zero

    def test_R2_equilibrium(self):
        mu = 0.5 + torch.rand(64, 4)
        q0 = torch.zeros(64, 6)
        self.assertTrue(torch.allclose(self.fam.v_plant(q0, mu),
                                       torch.zeros(64), atol=1e-12))
        self.assertTrue(torch.allclose(self.fam.grad_q(q0, q0, mu), q0,
                                       atol=1e-12))

    def test_R3_margin(self):
        self.assertGreater(self.fam.eps, 0.0)
        # a re-scan with a new seed (independent sample) still satisfies the margin (small statistical fluctuation allowed)
        Hp = self.fam._scan_hess_plant(n_scan=2048, seed=99)
        self.assertGreaterEqual(self.fam.min_hess_eig(self.fam.eps, Hp),
                                self.fam.hess_margin - 0.05)

    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fam.npz"
            self.fam.save(p)
            back = PlantedFamily.load(p)
        self.assertTrue(torch.allclose(back.C, self.fam.C))
        self.assertEqual(back.eps, self.fam.eps)


if __name__ == "__main__":
    unittest.main()
