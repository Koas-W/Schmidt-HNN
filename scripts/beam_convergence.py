#!/usr/bin/env python3
"""Master/slave mode-count convergence test + Euler-Bernoulli (EB) fidelity report.

For each candidate n, compare the production model (n master, n slave) with the verification model with doubled total DOF (n master, 3n slave):
  static:  midpoint deflection w_s(0.5)/h of the compressed static equilibrium;
  dynamic: first backbone frequency ω₁ at the upper amplitude bound A/h=3 (started from rest, cycles counted by zero crossings).
Both are checked at the 8 corners + center of the parameter box (9 points), criterion relative error < 1e-3; the smallest n passing both is chosen.
Also reported: (n,n) vs (n+2,n+2) (convergence of the master modes themselves, reference only, not a gate) and the EB fidelity ε_i=2.06(β_i r/L)².

Usage: python3 scripts/beam_convergence.py [--n-list 2 3 4 5 6] [--dt 2e-4] [--t-end 3.0]
Output: results/beam_checks/convergence_t5.csv
"""
import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.integrators import rollout_sampled, system_field  # noqa: E402
from src.systems.clamped_beam import ClampedBeamBase  # noqa: E402

torch.set_default_dtype(torch.float64)
TOL = 1e-3
L_OVER_H = 50.0
AMP = 3.0


def mu_corners_and_center(system) -> torch.Tensor:
    mr = system.default_mu_ranges()
    pts = list(itertools.product(*[(lo, hi) for lo, hi in mr]))
    pts.append(tuple((lo + hi) / 2 for lo, hi in mr))
    return torch.tensor(pts, dtype=torch.float64)


def static_mid_deflection(system, mu) -> torch.Tensor:
    a_s = system.static_equilibrium(mu)
    b_s = system.condense(a_s, mu)
    q_full = torch.cat([a_s, b_s], dim=-1)
    return system.deflection(q_full, torch.tensor([0.5]))[:, 0]


def backbone_omega1(system, mu, dt: float, t_end: float) -> torch.Tensor:
    """Integrate from rest (peak deflection = AMP·h) and estimate ω₁ from the mean period between rising zero crossings of a₁(t)."""
    c, a_s = system.peak_coeff(mu, AMP)
    q0 = a_s.clone()
    q0[:, 0] += c
    y0 = torch.cat([q0, torch.zeros_like(q0)], dim=-1)
    n_steps = int(round(t_end / dt))
    f = system_field(system, mu)
    with torch.no_grad():
        y = rollout_sampled(f, y0, dt, n_steps, 1)          # (T+1, B, 2n)
    a1 = y[:, :, 0] - a_s[:, 0].unsqueeze(0)                # relative to the static equilibrium
    a1 = a1 - a1.mean(dim=0, keepdim=True)
    t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    omegas, omegas_fft = [], []
    T1 = a1.shape[0]
    win = torch.hann_window(T1, periodic=False, dtype=torch.float64)
    freqs = torch.fft.rfftfreq(T1, d=dt)
    for j in range(mu.shape[0]):
        s = a1[:, j]
        up = torch.nonzero((s[:-1] < 0) & (s[1:] >= 0)).flatten()
        tc = t[up] - s[up] * dt / (s[up + 1] - s[up])   # zero-crossing times by linear interpolation
        if tc.numel() < 3:
            omegas.append(float("nan"))
        else:
            period = float((tc[-1] - tc[0]) / (tc.numel() - 1))
            omegas.append(2 * math.pi / period)
        # FFT main peak + parabolic interpolation on the log spectrum (cross-check; more robust against quasi-periodic beating)
        spec = torch.fft.rfft(s * win).abs()
        k = int(spec[1:].argmax()) + 1
        if 1 <= k < spec.numel() - 1:
            l0, l1, l2 = (torch.log(spec[k - 1:k + 2] + 1e-300)).tolist()
            delta = 0.5 * (l0 - l2) / (l0 - 2 * l1 + l2)
        else:
            delta = 0.0
        omegas_fft.append(2 * math.pi * float(freqs[k] + delta * (freqs[1] - freqs[0])))
    return torch.tensor(omegas), torch.tensor(omegas_fft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="*", default=[3, 4, 5, 6])
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--t-end", type=float, default=8.0)
    ap.add_argument("--out", default="results/beam_checks/convergence_t5.csv")
    args = ap.parse_args()

    out_dir = Path("results/beam_checks")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    chosen = None
    for n in args.n_list:
        t0 = time.time()
        prod = ClampedBeamBase(n, n)          # production model N=2n
        ver = ClampedBeamBase(n, 3 * n)       # verification model N=4n
        ref = ClampedBeamBase(n + 2, n + 2)   # master modes +2 (reference)
        mu = mu_corners_and_center(prod)
        ws = {k: static_mid_deflection(s, mu) for k, s in
              (("prod", prod), ("ver", ver), ("ref", ref))}
        both = {k: backbone_omega1(s, mu, args.dt, args.t_end) for k, s in
                (("prod", prod), ("ver", ver), ("ref", ref))}
        om = {k: v[0] for k, v in both.items()}
        om_fft = {k: v[1] for k, v in both.items()}
        e_om_fft = ((om_fft["prod"] - om_fft["ver"]).abs() / om_fft["ver"].abs()).max().item()
        xchk = ((om["prod"] - om_fft["prod"]).abs() / om["prod"].abs()).max().item()
        e_ws = ((ws["prod"] - ws["ver"]).abs() / ws["ver"].abs()).max().item()
        e_om = ((om["prod"] - om["ver"]).abs() / om["ver"].abs()).max().item()
        r_ws = ((ws["prod"] - ws["ref"]).abs() / ws["ref"].abs()).max().item()
        r_om = ((om["prod"] - om["ref"]).abs() / om["ref"].abs()).max().item()
        ok = (e_ws < TOL) and (e_om < TOL)
        eps_n = 2.06 * (prod.beta[n - 1].item() * 0.2887 / L_OVER_H) ** 2
        print(f"n={n}: doubled slaves -> static deflection {e_ws:.2e}  backbone ω₁ {e_om:.2e} (FFT {e_om_fft:.2e}; estimator mismatch {xchk:.1e})  "
              f"[{'PASS' if ok else 'fail'}] | master +2 -> {r_ws:.2e} / {r_om:.2e} | "
              f"ε_n(EB,L/h=50)={eps_n*100:.2f}% | ω₁(A/h=3)/β₁² range "
              f"{(om['prod']/prod.beta[0]**2).min():.2f}–{(om['prod']/prod.beta[0]**2).max():.2f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        rows.append({"n": n, "N_prod": 2 * n, "N_ver": 4 * n,
                     "err_static_double_slaves": f"{e_ws:.4e}",
                     "err_omega1_double_slaves": f"{e_om:.4e}",
                     "err_omega1_double_slaves_fft": f"{e_om_fft:.4e}",
                     "omega1_estimator_crosscheck": f"{xchk:.4e}",
                     "err_static_master_plus2": f"{r_ws:.4e}",
                     "err_omega1_master_plus2": f"{r_om:.4e}",
                     "pass": int(ok), "eps_EB_n": f"{eps_n:.4e}",
                     "omega1_ratio_min": f"{(om['prod']/prod.beta[0]**2).min():.4f}",
                     "omega1_ratio_max": f"{(om['prod']/prod.beta[0]**2).max():.4f}"})
        if ok and chosen is None:
            chosen = n
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsmallest n passing criterion {TOL:g} (doubled slaves) = {chosen}; written {args.out}")


if __name__ == "__main__":
    main()
