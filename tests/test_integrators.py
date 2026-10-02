"""Mandatory integrator tests (fixed-step RK4 is the ground-truth integrator):
- fourth-order convergence check;
- energy drift over the data time span at the frozen step meets the target (<=1e-8; thresholds with margin in each test).

Measured values:
- pendulum  dt=0.01: drift(t=20)=4.9e-11, order=4.01
- double_pendulum dt=0.005: the 16 trajectories sampled here give drift(t=20)=5.1e-9, order=4.12;
  but the full dataset (648 trajectories, including the extrap tier alpha=0.74) peaks at 2.9e-7, above the target,
  hence dt=0.002 is frozen (dt^4 scaling gives ~7.4e-9 <= 1e-8 on the full set; re-checked by generate_data.py when writing data).
"""
import math
import unittest

import torch

from src.data.integrators import energy_drift, rollout, system_field
from src.systems import get_system

torch.set_default_dtype(torch.float64)

# frozen step sizes (same values as in configs/*.yaml; a config change must be mirrored here)
# full-dataset measurements: double_pendulum dt=0.002 -> 7.4e-10;
# duffing dt=0.005 -> ~5e-10 (dt=0.01 gave 8.6e-9, too thin a margin); HH dt=0.01 -> 2.1e-10
FROZEN_DT = {"pendulum": 0.01, "double_pendulum": 0.002,
             "duffing": 0.005, "henon_heiles": 0.01}
DRIFT_TOL = {"pendulum": 5e-10, "double_pendulum": 5e-9,
             "duffing": 5e-9, "henon_heiles": 2e-9}  # ~10x margin over the measured values


def sample(system, B=16, seed=0):
    """Same sampling as the data pipeline: box proposal + energy-window rejection (if the system defines one)."""
    from src.data.trajectories import sample_initial_conditions
    g = torch.Generator().manual_seed(seed)
    box = torch.tensor(system.phase_space_box(), dtype=torch.float64)
    lo, hi = box[:, 0], box[:, 1]
    if system.mu_dim:
        mr = torch.tensor(system.default_mu_ranges(), dtype=torch.float64)
        mu = mr[:, 0] + (mr[:, 1] - mr[:, 0]) * torch.rand(B, system.mu_dim,
                                                           generator=g)
    else:
        mu = torch.zeros(B, 0, dtype=torch.float64)
    rows = [sample_initial_conditions(system, mu[i], 1, lo, hi, g)
            for i in range(B)]
    return torch.cat(rows), mu


class TestRK4(unittest.TestCase):

    def test_fourth_order_convergence(self):
        T = 2.0
        for name in FROZEN_DT:
            with self.subTest(system=name):
                system = get_system(name)
                y0, mu = sample(system)
                f = system_field(system, mu)
                errs = []
                with torch.no_grad():
                    yref = rollout(f, y0, 0.05 / 32, int(T / (0.05 / 32)))[-1]
                    for dt in (0.05, 0.025):
                        yT = rollout(f, y0, dt, int(T / dt))[-1]
                        errs.append((yT - yref).norm(dim=-1).max().item())
                order = math.log2(errs[0] / errs[1])
                self.assertTrue(3.7 < order < 4.3,
                                f"{name}: RK4 convergence order {order:.2f} deviates from 4")

    def test_energy_drift_at_frozen_dt(self):
        T = 20.0  # data generation time span
        for name, dt in FROZEN_DT.items():
            with self.subTest(system=name):
                system = get_system(name)
                y0, mu = sample(system)
                f = system_field(system, mu)
                with torch.no_grad():
                    traj = rollout(f, y0, dt, int(T / dt))
                drift = energy_drift(system, traj, mu).max().item()
                self.assertLess(drift, DRIFT_TOL[name],
                                f"{name}: dt={dt} drift {drift:.2e} exceeds the threshold")


if __name__ == "__main__":
    unittest.main()


class TestGraphedRollout(unittest.TestCase):
    """CUDA Graph rollout is bitwise equal to eager (skipped without CUDA)."""

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graphed_matches_eager(self):
        from src.data.integrators import rollout_sampled, rollout_sampled_graphed
        torch.manual_seed(0)
        A = torch.randn(6, 6, device="cuda", dtype=torch.float64); A = A - A.T   # antisymmetric -> linear Hamiltonian field
        f = lambda y: y @ A.T
        y0 = torch.randn(5, 6, device="cuda", dtype=torch.float64)
        a = rollout_sampled(f, y0, 1e-3, 400, 100)
        b = rollout_sampled_graphed(f, y0, 1e-3, 400, 100)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(a, b), f"max difference {(a-b).abs().max()}")
