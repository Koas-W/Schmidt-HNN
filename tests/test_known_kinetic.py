"""Known-kinetic wrapper: parameter alignment still holds for all models; qdot == p holds exactly."""
import unittest

import torch

from src.models.factory import MODEL_NAMES, build_model

torch.set_default_dtype(torch.float64)
CFG = {"target_params": 40000, "known_kinetic": True,
       "pshnn": {"R": 8, "depth": 2, "block": "swiglu", "bias_head": True},
       "pshnn_mlp": {"R": 8, "depth": 2, "bias_head": True},
       "ahnn": {"depth": 2}, "mlp_ode": {"depth": 2}}


class TestKnownKinetic(unittest.TestCase):
    def test_params_aligned_and_qdot_is_p(self):
        n, m, B = 4, 3, 32
        g = torch.Generator().manual_seed(0)
        x = torch.randn(B, 2 * n, generator=g)
        mu = torch.randn(B, m, generator=g)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model, info = build_model(name, n, m, CFG, CFG["target_params"])
                self.assertLess(abs(info["n_params"] - 40000) / 40000, 0.05)
                model = model.double()
                dx = model.time_derivative(x, mu, create_graph=False)
                self.assertEqual(dx.shape, (B, 2 * n))
                if name != "symoden_mu":      # the symoden kinetic structure is declared redundant here and is not wrapped
                    self.assertTrue(torch.allclose(dx[:, :n], x[:, n:], atol=1e-12),
                                    f"{name}: qdot must equal p exactly")

    def test_off_by_default(self):
        """With the switch off, ahnn/mlp_ode remain the original baselines (qdot != p in general)."""
        cfg = {"target_params": 40000, "ahnn": {"depth": 2}, "mlp_ode": {"depth": 2}}
        x = torch.randn(8, 8); mu = torch.randn(8, 3)
        for name in ("ahnn", "mlp_ode"):
            model, _ = build_model(name, 4, 3, cfg, 40000)
            dx = model.double().time_derivative(x, mu)
            self.assertFalse(torch.allclose(dx[:, :4], x[:, 4:]))


if __name__ == "__main__":
    unittest.main()
