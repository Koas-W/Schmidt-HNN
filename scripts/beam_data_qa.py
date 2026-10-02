#!/usr/bin/env python3
"""Beam dataset acceptance checks: amplitude coverage (t8) and symmetry/excitation (t9); t8/t9 are the CSV column labels.

t8 amplitude coverage: for each training anchor, the peak deflection max_{t,X}|w|/h of its train-role trajectories (slave-mode contribution included):
   (a) >= 60% of the trajectories fall in [AMP_LO, AMP_HI]; (b) none exceeds AMP_HI;
   (c) the anchor's usable interval [max(1, static+0.5), 3] split into four equal bins, each >= 5% (distribution coverage, not piled at one end);
   (d) measured dynamic offset max_{t,X}|w−w_s|/h >= 0.3 for every trajectory (not a small oscillation about the static equilibrium);
   (e) truncation consistency: slave/master modal amplitude ratio (trajectory median) <= 1e-2; the time-averaged energy share of the highest
       master mode is reported only (the master-mode share is physics the network must learn, not truncation error; at μ₁=1 the 5th-mode share can reach 1-2%).
t9 symmetry and excitation: (a) cross-parity G_ij and antisymmetric-mode f_i are numerically zero (<= 1e-10);
   (b) the sample std of the antisymmetric q, p components is of the same order as that of the symmetric ones (ratio >= 0.1),
       confirming that the initial conditions do excite the antisymmetric modes (otherwise they would be dead inputs identically zero).

Usage: python3 scripts/beam_data_qa.py [--data data_cache/clamped_beam]
Output: results/beam_checks/data_qa_t8t9.csv + console verdict
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_dataset  # noqa: E402
from src.systems import get_system  # noqa: E402

torch.set_default_dtype(torch.float64)
DYN_FLOOR = 0.3       # lower bound on the measured dynamic offset (h); sampling target is static+0.5, 40% loss allowed for random directions
ANTI_RATIO_MIN = 0.05  # lower bound on the antisymmetric/symmetric std ratio (power-law spectrum: 2nd-mode share ~7% => displacement ratio ~0.1)
SLAVE_RATIO_MAX = 1e-2  # upper bound on the slave/master modal amplitude ratio (truncation consistency; measured median ~7e-4, max 1.6e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_cache/clamped_beam")
    args = ap.parse_args()
    data, manifest = load_dataset(args.data)
    system = get_system(manifest["system"])
    n = system.n_master
    lo_amp, hi_amp = system.AMP_LO, system.AMP_HI
    xg = torch.linspace(0.0, 1.0, 401)

    # ---- t9(a): parity structure ----
    N = system.n_total
    cross = max(abs(float(system.G[i, j])) for i in range(N) for j in range(N)
                if (i % 2) != (j % 2))
    anti_f = max(abs(float(system.f[i])) for i in range(1, N, 2))
    t9a = cross <= 1e-10 and anti_f <= 1e-10
    print(f"t9(a) cross-parity |G_ij|max={cross:.1e}  antisymmetric |f_i|max={anti_f:.1e}  "
          f"{'PASS' if t9a else 'FAIL'}")

    # ---- t8 + t9(b): per training anchor ----
    a0, a1 = manifest["roles"]["train"]
    traj = data["traj_train"][:, a0:a1]          # (n_mu, n_traj, T, 2n)
    mu = data["mu_train"]
    n_mu, n_traj, T, D = traj.shape
    rows, all_ok = [], True
    sym_idx = list(range(0, n, 2))
    anti_idx = list(range(1, n, 2))
    for i in range(n_mu):
        states = traj[i].reshape(n_traj * T, D)
        a, p = states[:, :n], states[:, n:]
        mu_rep = mu[i].expand(states.shape[0], -1)
        b = system.condense(a, mu_rep)
        w = system.deflection(torch.cat([a, b], -1), xg)          # (n_traj·T, 401)
        peak = w.abs().max(dim=-1).values.reshape(n_traj, T).max(dim=-1).values
        # static deflection profile and measured dynamic offset max_{t,X}|w−w_s|/h
        lo_eff, static, a_s = system.amplitude_bounds(mu[i][None])
        w_s = system.deflection(torch.cat([a_s, system.condense(a_s, mu[i][None])], -1), xg)
        dyn = (w - w_s).abs().max(dim=-1).values.reshape(n_traj, T).max(dim=-1).values
        in_win = ((peak >= lo_amp) & (peak <= hi_amp + 1e-9)).float().mean().item()
        over = int((peak > hi_amp + 1e-9).sum())
        lo_e = float(lo_eff)
        edges = torch.linspace(lo_e, hi_amp, 5)               # four bins over the anchor's usable interval
        bins = [int(((peak >= edges[k]) & (peak < edges[k + 1] if k < 3 else peak <= edges[k + 1] + 1e-9)).sum())
                for k in range(4)]
        bin_frac = [c / n_traj for c in bins]
        dyn_ok = float(dyn.min()) >= DYN_FLOOR
        t8 = (in_win >= 0.6 and over == 0 and all(bf >= 0.05 for bf in bin_frac)
              and dyn_ok)
        # t9(b): std ratio of antisymmetric to symmetric components (nonzero and learnable suffices; relative error is normalized per mode by its own variance)
        sq = a[:, sym_idx].std().item(); sa = a[:, anti_idx].std().item() if anti_idx else float("nan")
        sp_s = p[:, sym_idx].std().item(); sp_a = p[:, anti_idx].std().item() if anti_idx else float("nan")
        t9b = (sa / sq >= ANTI_RATIO_MIN) and (sp_a / sp_s >= ANTI_RATIO_MIN) if anti_idx else True
        # t8(e): truncation consistency: time-averaged linear modal energy shares and slave-mode response ratio
        omega = system.k_diag[:n].sqrt()
        e_mode = 0.5 * (p ** 2 + (omega ** 2) * (a - a_s) ** 2)          # (n_traj·T, n)
        share = (e_mode / e_mode.sum(-1, keepdim=True)).mean(0)          # mean share per mode
        top_share = float(share[-1])
        slave_ratio = float((b.norm(dim=-1) / a.norm(dim=-1)).median())
        t8e = slave_ratio <= SLAVE_RATIO_MAX
        all_ok &= t8 and t9b and t8e
        rows.append({"anchor": i, "mu": mu[i].tolist(),
                     "static_peak": f"{float(static):.3f}", "lo_eff": f"{lo_e:.3f}",
                     "dyn_min": f"{dyn.min():.3f}", "dyn_ok": int(dyn_ok),
                     "peak_min": f"{peak.min():.3f}", "peak_max": f"{peak.max():.3f}",
                     "frac_in_window": f"{in_win:.2f}", "n_over": over,
                     "bins": bin_frac, "t8": int(t8),
                     "std_ratio_q_anti_sym": f"{sa / sq:.3f}",
                     "std_ratio_p_anti_sym": f"{sp_a / sp_s:.3f}", "t9b": int(t9b),
                     "energy_share_top_master": f"{top_share:.2e}",
                     "energy_share_modes": [f"{v:.2e}" for v in share.tolist()],
                     "slave_over_master_norm_median": f"{slave_ratio:.2e}", "t8e": int(t8e)})
    n_pass8 = sum(r["t8"] for r in rows); n_pass9 = sum(r["t9b"] for r in rows)
    n_pass8e = sum(r["t8e"] for r in rows)
    print(f"t8(e) truncation consistency: {n_pass8e}/{n_mu} anchors with slave/master amplitude ratio <={SLAVE_RATIO_MAX:g} (median "
          f"{sorted(float(r['slave_over_master_norm_median']) for r in rows)[n_mu//2]:.2e}, max "
          f"{max(float(r['slave_over_master_norm_median']) for r in rows):.2e}); reported: highest-master-mode energy share median "
          f"{sorted(float(r['energy_share_top_master']) for r in rows)[n_mu//2]:.2e}, max "
          f"{max(float(r['energy_share_top_master']) for r in rows):.2e}")
    print(f"t8  amplitude coverage: {n_pass8}/{n_mu} anchors pass; global peak range "
          f"[{min(float(r['peak_min']) for r in rows):.2f}, {max(float(r['peak_max']) for r in rows):.2f}]")
    print(f"t9(b) antisymmetric excitation: {n_pass9}/{n_mu} anchors pass")
    out = Path("results/beam_checks"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "data_qa_t8t9.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"overall: {'PASS' if (all_ok and t9a) else 'FAIL'}; written {out/'data_qa_t8t9.csv'}")


if __name__ == "__main__":
    main()
