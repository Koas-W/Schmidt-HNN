"""Hard gates for the Muon combined optimizer: orthogonalization correctness, eager/graph consistency, capturable tensor lr."""
import unittest

import torch

from src.models.factory import build_model
from src.train.muon import MuonAdamW, newton_schulz_orth
from src.train.trainer import evaluate_loss, train_model

CFG = {"target_params": 40000,
       "pshnn": {"R": 2, "R_L": 1, "depth": 2, "block": "swiglu", "bias_head": True}}


class TestNewtonSchulz(unittest.TestCase):
    def test_singular_values_flattened(self):
        """5 NS steps pull the singular values into ~[0.7,1.3] (convergence band of the Jordan coefficients); ill-scaled columns within condition number 1e2 should be flattened too.
        Note: NS cannot recover singular values below ~1e-3 relative magnitude (known property), so a 1e6 span is not tested."""
        g = torch.Generator().manual_seed(0)
        for shape, span in [((56, 56), 0), ((56, 56), 1), ((56, 5), 1), ((5, 56), 1)]:
            G = torch.randn(*shape, generator=g) * torch.logspace(-span, span, shape[1])
            O = newton_schulz_orth(G.double(), steps=5)
            s = torch.linalg.svdvals(O)
            frac = float(((s > 0.6) & (s < 1.3)).float().mean())   # steady-state band of the Jordan coefficients is about [0.68, 1.13]; the smallest singular value of a square matrix is close to 0 anyway
            self.assertGreater(frac, 0.85, f"{shape} span=1e{span}: flattened fraction {frac:.2f}, s={s}")
            U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
            P = U @ Vh
            cos = (O * P).sum() / (O.norm() * P.norm())
            self.assertGreater(float(cos), 0.95, f"{shape}: cosine to the polar-decomposition direction {float(cos):.3f}")


class TestMuonAdamW(unittest.TestCase):
    def test_partition_and_step_moves_both_groups(self):
        torch.manual_seed(0)
        model = build_model("pshnn", 8, 4, CFG, 40000)[0]
        opt = MuonAdamW(model.named_parameters(), lr=1e-3)
        n_mat = sum(1 for p in model.parameters() if p.ndim == 2)
        self.assertEqual(len(opt.mat), n_mat)
        x = torch.randn(64, 16); mu = torch.rand(64, 4)
        loss = model.time_derivative(x, mu, create_graph=True).pow(2).mean()
        loss.backward()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        opt.step()
        moved_mat = any(not torch.equal(before[k], v) for k, v in model.state_dict().items() if v.ndim == 2)
        moved_vec = any(not torch.equal(before[k], v) for k, v in model.state_dict().items() if v.ndim == 1)
        self.assertTrue(moved_mat and moved_vec)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graph_matches_eager(self):
        device = torch.device("cuda")
        g = torch.Generator().manual_seed(0)
        N = 16384
        x = torch.randn(N, 16, generator=g); mu = torch.rand(N, 4, generator=g)
        W = torch.randn(16, 16, generator=g) * 0.1
        tv = {"x": x, "mu": mu, "dx": x @ W}
        vv = {"x": x[:2048], "mu": mu[:2048], "dx": tv["dx"][:2048]}
        base = {"lr": 1e-3, "batch_size": 8192, "max_steps": 600, "eval_every": 100,
                "patience": 50, "weight_decay": 1e-4, "optimizer": "muon",
                "lr_schedule": "cosine", "lr_min": 1e-5}
        vals = {}
        for label, extra in [("eager", {}), ("graph", {"cuda_graph": True})]:
            torch.manual_seed(0)
            model = build_model("pshnn", 8, 4, CFG, 40000)[0]
            state, _ = train_model(model, tv, vv, {**base, **extra}, seed=0, device=device)
            model.load_state_dict(state)
            vals[label] = evaluate_loss(model, vv, device)
        rel = abs(vals["eager"] - vals["graph"]) / vals["eager"]
        self.assertLess(rel, 0.15, f"muon: final val loss of graphed vs eager differs too much: {vals}")


if __name__ == "__main__":
    unittest.main()
