#!/usr/bin/env python3
"""RK4 step-size convergence and energy-drift acceptance check.

At the 4 corners + center of the parameter box (5 μ points) x several initial conditions within the energy window per point,
integrate over t_span for a set of dt values; report the maximum relative energy drift and the drift ratio between adjacent dt
(RK4 is fourth order: halving the step should reduce the drift by about 16x), and the largest dt satisfying <= 1e-8.

Usage: python3 scripts/beam_dt_sweep.py [--dts 2e-3 1e-3 5e-4 2.5e-4] [--t-span 10] [--n-ic 6]
Output: results/beam_checks/dt_sweep_t6.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.integrators import energy_drift, rollout_sampled, system_field  # noqa: E402
from src.data.trajectories import sample_initial_conditions  # noqa: E402
from src.systems import get_system  # noqa: E402

torch.set_default_dtype(torch.float64)
TARGET = 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dts", type=float, nargs="*",
                    default=[2e-3, 1e-3, 5e-4, 2.5e-4])
    ap.add_argument("--t-span", type=float, default=10.0)
    ap.add_argument("--n-ic", type=int, default=6)
    ap.add_argument("--system", default="clamped_beam")
    args = ap.parse_args()

    system = get_system(args.system)
    mr = system.default_mu_ranges()
    lo = [r[0] for r in mr]
    hi = [r[1] for r in mr]
    mid = [(a + b) / 2 for a, b in mr]
    mus = torch.tensor([lo, hi, [lo[0], hi[1], hi[2]], [hi[0], lo[1], lo[2]], mid],
                       dtype=torch.float64)
    box = torch.tensor(system.phase_space_box(), dtype=torch.float64)
    g = torch.Generator().manual_seed(2024)
    y0 = torch.cat([sample_initial_conditions(system, mus[i], args.n_ic,
                                              box[:, 0], box[:, 1], g)
                    for i in range(mus.shape[0])])
    mu_rep = mus.repeat_interleave(args.n_ic, dim=0)
    f = system_field(system, mu_rep)

    rows, prev = [], None
    print(f"[{args.system}] t_span={args.t_span}, μ points={mus.shape[0]}, ICs per point={args.n_ic}")
    for dt in args.dts:
        n_steps = int(round(args.t_span / dt))
        sample_every = max(1, n_steps // 200)
        with torch.no_grad():
            traj = rollout_sampled(f, y0, dt, n_steps, sample_every)   # (T+1,B,2n)
        drift = energy_drift(system, traj, mu_rep).max().item()
        ratio = (prev / drift) if prev else float("nan")
        ok = drift <= TARGET
        print(f"  dt={dt:.2e}  max energy drift={drift:.3e}  ratio vs prev={ratio:5.1f}  "
              f"{'✓ ≤1e-8' if ok else '✗'}", flush=True)
        rows.append({"dt": dt, "max_rel_energy_drift": f"{drift:.4e}",
                     "ratio_vs_prev": f"{ratio:.2f}", "pass": int(ok)})
        prev = drift
    out = Path("results/beam_checks"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "dt_sweep_t6.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    passing = [r["dt"] for r in rows if r["pass"]]
    print(f"largest dt satisfying <=1e-8 = {max(passing) if passing else None}; written {out/'dt_sweep_t6.csv'}")


if __name__ == "__main__":
    main()
