"""Mandatory model tests:
- Schmidt-HNN rank-R representability smoke test: with R large enough it fits a hand-built low-rank separable Hamiltonian;
- autograd vector field vs central differences of the Hamiltonian (PSHNN/AHNN);
- parameter counts of all models aligned within +-5% (fairness mechanism);
- lower-triangular structure and positive diagonal of the Schmidt-HNN L factor.
"""
import unittest

import torch

from src.models.baselines.ahnn import AHNN
from src.models.factory import MODEL_NAMES, build_model
from src.models.pshnn import PSHNN

torch.set_default_dtype(torch.float64)

TARGET = 40_000


class TestParamAlignment(unittest.TestCase):

    def test_three_models_within_5pct(self):
        for n_dof, mu_dim in [(1, 0), (2, 2)]:  # pendulum / double pendulum shapes
            for name in MODEL_NAMES:
                with self.subTest(system=(n_dof, mu_dim), model=name):
                    _, info = build_model(name, n_dof, mu_dim, {}, TARGET)
                    rel = abs(info["n_params"] - TARGET) / TARGET
                    self.assertLessEqual(
                        rel, 0.05,
                        f"{name}: {info['n_params']} vs {TARGET}")


class TestFiniteDifference(unittest.TestCase):
    """time_derivative (autograd) must agree with central differences of H_hat."""

    def _check(self, model, n_dof, mu_dim, seed=0):
        g = torch.Generator().manual_seed(seed)
        B = 32
        x = torch.rand(B, 2 * n_dof, generator=g) * 2 - 1
        mu = torch.rand(B, mu_dim, generator=g) + 0.5
        td = model.time_derivative(x, mu)
        eps = 1e-6
        grads = []
        for j in range(2 * n_dof):
            e = torch.zeros_like(x)
            e[:, j] = eps
            with torch.no_grad():
                Hp = model.hamiltonian((x + e)[:, :n_dof], (x + e)[:, n_dof:], mu)
                Hm = model.hamiltonian((x - e)[:, :n_dof], (x - e)[:, n_dof:], mu)
            grads.append((Hp - Hm) / (2 * eps))
        dH = torch.stack(grads, dim=-1)             # (B, 2n) = (∂H/∂q, ∂H/∂p)
        expected = torch.cat([dH[:, n_dof:], -dH[:, :n_dof]], dim=-1)
        self.assertTrue(torch.allclose(td, expected, atol=1e-6),
                        f"max|Δ|={(td - expected).abs().max():.3e}")

    def test_pshnn(self):
        self._check(PSHNN(2, 2, width=24, depth=2, R=4), 2, 2)

    def test_ahnn(self):
        self._check(AHNN(2, 2, width=64, depth=2), 2, 2)

    def test_pshnn_concat(self):
        from src.models.pshnn_concat import PSHNNConcat
        self._check(PSHNNConcat(2, 2, width=24, depth=2), 2, 2)

    def test_symoden_mu(self):
        from src.models.baselines.symoden_mu import SymODENmu
        self._check(SymODENmu(2, 2, width=48, depth=2), 2, 2)

    def test_sep_hnn(self):
        from src.models.sep_hnn import SepHNN
        self._check(SepHNN(2, 2, width=24, depth=2, R=4), 2, 2)

    def test_pshnn_mlp_block(self):
        self._check(PSHNN(2, 2, width=24, depth=2, R=4, block="mlp"), 2, 2)


class TestSymODENmuStructure(unittest.TestCase):

    def test_minv_positive_definite(self):
        from src.models.baselines.symoden_mu import SymODENmu
        torch.manual_seed(0)
        model = SymODENmu(3, 2, width=32)
        q, mu = torch.randn(64, 3), torch.rand(64, 2)
        eigvals = torch.linalg.eigvalsh(model.minv(q, mu))
        # L L^T >= 0 plus 0.1*I => smallest eigenvalue >= 0.1 (positive-definiteness mechanism of the official PSD class)
        self.assertGreaterEqual(eigvals.min().item(), 0.1 - 1e-10)


class TestPSHNNStructure(unittest.TestCase):

    def test_L_lower_triangular_positive_diag(self):
        model = PSHNN(3, 2, width=16, depth=1, R=4)
        q = torch.randn(8, 3)
        mu = torch.rand(8, 2)
        _, L = model.V_and_L(q, mu)
        upper = torch.triu(L, diagonal=1)
        self.assertTrue(torch.all(upper == 0))
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        self.assertTrue(torch.all(diag > 0))


class TestHeadRankAllocation(unittest.TestCase):
    """R_V/R_L capacity split: the default is equivalent to the previous version; R_L<R_V reduces the parameter count as expected."""

    def test_default_equivalent_param_count(self):
        a = PSHNN(2, 2, width=32, depth=2, R=8)
        b = PSHNN(2, 2, width=32, depth=2, R=8, R_L=8)
        na = sum(p.numel() for p in a.parameters())
        nb = sum(p.numel() for p in b.parameters())
        self.assertEqual(na, nb)

    def test_rl_smaller_reduces_heads(self):
        big = PSHNN(2, 2, width=32, depth=2, R=8, R_L=8)
        small = PSHNN(2, 2, width=32, depth=2, R=8, R_L=2)
        n_big = sum(p.numel() for p in big.parameters())
        n_small = sum(p.numel() for p in small.parameters())
        # n_L=3 L heads, each rank 8->2; the head layers of State-Net and Param-Net each save 3*6*(32+1)
        self.assertEqual(n_big - n_small, 2 * 3 * 6 * 33)

    def test_fd_with_split_ranks(self):
        model = PSHNN(2, 2, width=24, depth=2, R=8, R_L=2)
        TestFiniteDifference()._check(model, 2, 2)

    def test_bias_head_both_modes(self):
        # bias head: fd check in both modes + parameter count larger by exactly one head (width+1)
        for lk in (True, False):
            with self.subTest(learn_kinetic=lk):
                base = PSHNN(2, 2, width=24, depth=2, R=4,
                             learn_kinetic=lk, bias_head=False)
                bh = PSHNN(2, 2, width=24, depth=2, R=4,
                           learn_kinetic=lk, bias_head=True)
                n0 = sum(p.numel() for p in base.parameters())
                n1 = sum(p.numel() for p in bh.parameters())
                self.assertEqual(n1 - n0, 24 + 1)
                TestFiniteDifference()._check(bh, 2, 2)


class TestRepresentability(unittest.TestCase):
    """Rank-R representability smoke test: fit H* = p^2/2 + mu q^4 (rank-1 separable + constant L)."""

    def test_pshnn_fits_separable_target(self):
        torch.manual_seed(0)
        B = 4096
        q = torch.rand(B, 1) * 2 - 1
        p = torch.rand(B, 1) * 2 - 1
        mu = torch.rand(B, 1) + 0.5
        x = torch.cat([q, p], dim=-1)
        dx = torch.cat([p, -4.0 * mu * q**3], dim=-1)   # (q̇, ṗ)

        model = PSHNN(1, 1, width=32, depth=2, R=4)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        loss = None
        for step in range(1500):
            opt.zero_grad(set_to_none=True)
            pred = model.time_derivative(x, mu, create_graph=True)
            loss = torch.nn.functional.mse_loss(pred, dx)
            loss.backward()
            opt.step()
            if loss.item() < 5e-4:
                break
        self.assertLess(loss.item(), 1e-3,
                        f"Schmidt-HNN failed to fit the rank-1 separable target, final loss={loss.item():.2e}")


if __name__ == "__main__":
    unittest.main()
