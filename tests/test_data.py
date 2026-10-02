"""Data pipeline tests: properties of the three-tier mu split, reproducible generation, training isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import (eval_views, generate_dataset, load_dataset,
                              training_views)
from src.data.mu_split import make_mu_split

torch.set_default_dtype(torch.float64)

RANGES = [(0.35, 0.65), (0.7, 1.3)]


def tiny_config(system: str) -> dict:
    if system == "double_pendulum":
        data = dict(dt=0.005, t_span=1.0, sample_every=20, noise_std=0.0,
                    seed=7, n_traj=dict(train=3, val=1, test=2),
                    mu_split=dict(n_train_per_dim=2, extrap_frac=0.3,
                                  n_extrap_side=1))
    else:  # pendulum / duffing / henon_heiles
        data = dict(dt=0.01, t_span=2.0, sample_every=10, noise_std=0.0,
                    seed=7, n_traj=dict(train=4, val=2, test=2),
                    mu_split=dict(n_train_per_dim=2, extrap_frac=0.3,
                                  n_extrap_side=1))
    return {"system": system, "data": data}


class TestMuSplit(unittest.TestCase):

    def setUp(self):
        self.split = make_mu_split(RANGES, n_train_per_dim=4,
                                   extrap_frac=0.3, n_extrap_side=2)

    def test_shapes(self):
        self.assertEqual(self.split["train"].shape, (16, 2))   # 4×4
        self.assertEqual(self.split["interp"].shape, (9, 2))   # 3x3 midpoints
        self.assertEqual(self.split["extrap"].shape, (8, 2))   # 2 dims x 2 sides x 2 points

    def test_interp_inside_hull_and_unseen(self):
        lo = torch.tensor([r[0] for r in RANGES])
        hi = torch.tensor([r[1] for r in RANGES])
        interp = self.split["interp"]
        self.assertTrue(((interp > lo) & (interp < hi)).all())
        # must not coincide with any train grid point
        d = (interp.unsqueeze(1) - self.split["train"].unsqueeze(0)).norm(dim=-1)
        self.assertGreater(d.min().item(), 1e-8)

    def test_extrap_outside_hull(self):
        lo = torch.tensor([r[0] for r in RANGES])
        hi = torch.tensor([r[1] for r in RANGES])
        outside = ((self.split["extrap"] < lo) |
                   (self.split["extrap"] > hi)).any(dim=-1)
        self.assertTrue(outside.all())

    def test_deterministic(self):
        again = make_mu_split(RANGES, n_train_per_dim=4,
                              extrap_frac=0.3, n_extrap_side=2)
        for k in ("train", "interp", "extrap"):
            self.assertTrue(torch.equal(self.split[k], again[k]))

    def test_mu_dim_zero(self):
        split = make_mu_split([])
        self.assertEqual(split["train"].shape, (1, 0))
        self.assertEqual(split["interp"].shape[0], 0)
        self.assertEqual(split["extrap"].shape[0], 0)


class TestDatasetGeneration(unittest.TestCase):

    def _roundtrip(self, system):
        cfg = tiny_config(system)
        with tempfile.TemporaryDirectory() as tmp:
            first = generate_dataset(cfg, tmp)
            data, manifest = load_dataset(tmp)
            again = generate_dataset(cfg, Path(tmp) / "again")
        return cfg, first, again, data, manifest

    def test_shapes_isolation_reproducibility(self):
        for system in ("pendulum", "double_pendulum", "duffing",
                       "henon_heiles"):
            with self.subTest(system=system):
                cfg, first, again, data, manifest = self._roundtrip(system)
                roles = cfg["data"]["n_traj"]
                n_tot = sum(roles.values())
                T = manifest["n_points_per_traj"]
                n_mu = len(manifest["tiers"]["train"])

                self.assertEqual(tuple(data["traj_train"].shape[:3]),
                                 (n_mu, n_tot, T))
                # reproducible: regenerating from the same config is bitwise identical
                for k in first:
                    np.testing.assert_array_equal(first[k], again[k])
                # training isolation: training views contain only train/val-role points
                tv = training_views(data, manifest)
                n_dim2 = data["traj_train"].shape[-1]
                self.assertEqual(tv["train"]["x"].shape,
                                 (n_mu * roles["train"] * T, n_dim2))
                self.assertEqual(tv["val"]["x"].shape,
                                 (n_mu * roles["val"] * T, n_dim2))
                # evaluation views: the train tier exposes only the test role
                ev = eval_views(data, manifest)
                self.assertEqual(ev["train"]["traj"].shape[1], roles["test"])
                if system == "pendulum":  # the only system without mu
                    self.assertNotIn("interp", ev)
                else:
                    self.assertIn("interp", ev)
                    self.assertIn("extrap", ev)

    def test_energy_window_and_libration(self):
        # double pendulum: IC energies lie inside energy_window; |q| < pi along the whole trajectory
        # (libration regime; energy conservation forbids crossing the barrier, which validates the rejection sampling)
        from src.systems import get_system
        cfg, _, _, data, manifest = self._roundtrip("double_pendulum")
        system = get_system("double_pendulum")
        for tier in ("train", "interp", "extrap"):
            traj, mu = data[f"traj_{tier}"], data[f"mu_{tier}"]
            n_mu, n_traj = traj.shape[:2]
            ic = traj[:, :, 0].reshape(n_mu * n_traj, -1)
            mu_rep = mu.repeat_interleave(n_traj, dim=0)
            H0 = system.hamiltonian(ic[:, :2], ic[:, 2:], mu_rep)
            e_min, e_max = system.energy_window(mu_rep)
            self.assertTrue((H0 >= e_min - 1e-12).all() and
                            (H0 <= e_max + 1e-12).all(),
                            f"{tier}: IC energy outside the energy window")
            self.assertLess(traj[..., :2].abs().max().item(), torch.pi,
                            f"{tier}: trajectory angle left (-pi, pi), entered the rotation regime")

    def test_henon_heiles_confinement(self):
        # HH: energy window + start inside the well => long integration never leaves the saddle radius (confinement check)
        import torch as T
        from src.systems import get_system
        from src.data.trajectories import sample_initial_conditions
        from src.data.integrators import rollout_sampled, system_field
        system = get_system("henon_heiles")
        g = T.Generator().manual_seed(3)
        box = T.tensor(system.phase_space_box(), dtype=T.float64)
        lo, hi = box[:, 0], box[:, 1]
        # cover both ends of lambda, including extrap
        for lam in (0.68, 1.0, 1.32):
            mu_row = T.tensor([lam], dtype=T.float64)
            y0 = sample_initial_conditions(system, mu_row, 16, lo, hi, g)
            mu = mu_row.expand(16, -1)
            with T.no_grad():
                traj = rollout_sampled(system_field(system, mu), y0,
                                       0.01, 5000, 50)  # t=50
            r = (traj[..., :2].norm(dim=-1) * lam).max().item()
            self.assertLess(r, 1.0,
                            f"λ={lam}: trajectory left the saddle radius (lambda|q|={r:.3f})")

    def test_derivative_consistency(self):
        # stored derivatives = analytic vector field evaluated at the stored states
        from src.systems import get_system
        cfg, _, _, data, manifest = self._roundtrip("double_pendulum")
        system = get_system("double_pendulum")
        traj, deriv = data["traj_train"], data["deriv_train"]
        mu = data["mu_train"]
        n_mu, n_traj, T, D = traj.shape
        n = system.dim
        flat = traj.reshape(-1, D)
        mu_rep = mu.repeat_interleave(n_traj * T, dim=0)
        qdot, pdot = system.vector_field(flat[:, :n], flat[:, n:], mu_rep)
        expected = torch.cat([qdot, pdot], -1).reshape(n_mu, n_traj, T, D)
        self.assertTrue(torch.allclose(expected, deriv, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
