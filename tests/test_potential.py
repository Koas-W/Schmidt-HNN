"""AHNN potential metric tests: <DeltaV> ~ 0 on analytic ground-truth systems; the Taylor fit recovers the 6 true terms exactly."""
import unittest

import torch

from src.systems import get_system
from src.analysis.potential import deltaV_curve, taylor_coeffs

torch.set_default_dtype(torch.float64)


class TestDeltaV(unittest.TestCase):
    """Using the system's own H as the "perfect model", <DeltaV> must be 0 (V_pred equals V_real pointwise)."""

    def test_zero_on_true_hh1p(self):
        s = get_system("henon_heiles_ahnn")
        mu = torch.tensor([[0.2], [0.4], [0.6], [0.8], [1.0]])
        dv = deltaV_curve(s.hamiltonian, s, mu)
        self.assertLess(dv.abs().max().item(), 1e-9,
                        f"ground-truth system: <DeltaV> should be ~0, got {dv.tolist()}")

    def test_zero_on_true_hh2p(self):
        s = get_system("henon_heiles_2p")
        mu = torch.tensor([[0.3, 0.5], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
        dv = deltaV_curve(s.hamiltonian, s, mu)
        self.assertLess(dv.abs().max().item(), 1e-9,
                        f"ground-truth 2P system: <DeltaV> should be ~0, got {dv.tolist()}")

    def test_nonzero_on_perturbed(self):
        # inject a wrong quadratic term into the true H => <DeltaV> clearly >0 (the metric does measure the error)
        s = get_system("henon_heiles_ahnn")
        bad = lambda q, p, mu: s.hamiltonian(q, p, mu) + 0.1 * q[:, 0] ** 2
        mu = torch.tensor([[0.4]])
        self.assertGreater(deltaV_curve(bad, s, mu).item(), 1e-3)


class TestTaylor(unittest.TestCase):
    """Eq. 5: fit of the 35 monomials up to degree 3; recovers the 6 nonzero coefficients of the true HH exactly."""

    def _coeff_map(self, coeffs_row, exps):
        return {e: coeffs_row[i].item() for i, e in enumerate(exps)}

    def test_recover_hh1p(self):
        s = get_system("henon_heiles_ahnn")
        lam = [0.2, 0.5, 0.8]
        mu = torch.tensor([[x] for x in lam])
        coeffs, exps = taylor_coeffs(s.hamiltonian, mu)
        self.assertEqual(len(exps), 35)
        for r, l in enumerate(lam):
            c = self._coeff_map(coeffs[r], exps)
            for e in [(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)]:
                self.assertAlmostEqual(c[e], 0.5, places=5, msg=f"λ={l} {e}")
            self.assertAlmostEqual(c[(2, 1, 0, 0)], l, places=5)      # q1^2 q2 -> lambda
            self.assertAlmostEqual(c[(0, 3, 0, 0)], -l / 3, places=5)  # q2^3 -> -lambda/3
            # the remaining 29 terms should be ~0
            spurious = max(abs(v) for e, v in c.items()
                           if e not in [(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0),
                                        (0, 0, 0, 2), (2, 1, 0, 0), (0, 3, 0, 0)])
            self.assertLess(spurious, 1e-5, f"λ={l} spurious terms too large {spurious:.2e}")

    def test_recover_hh2p(self):
        s = get_system("henon_heiles_2p")
        mu = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        coeffs, exps = taylor_coeffs(s.hamiltonian, mu)
        for r, (l1, l2) in enumerate([(0.3, 0.7), (0.6, 0.4)]):
            c = self._coeff_map(coeffs[r], exps)
            self.assertAlmostEqual(c[(2, 1, 0, 0)], l1, places=5)       # q1^2 q2 -> lambda1
            self.assertAlmostEqual(c[(0, 3, 0, 0)], -l2 / 3, places=5)  # q2^3 -> -lambda2/3


if __name__ == "__main__":
    unittest.main()
