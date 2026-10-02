#!/usr/bin/env python3
"""Midpoint-deflection trajectory error (mechanical definition, primary metric instead of full-state L2) + number of vibration cycles in the window.

For results/<exp>/<seed>/{model}.pt: starting from the true initial state, integrate the learned vector field with the same RK4
(dt, sample_every) as the ground truth up to t_end (default = data time span 5.0) and compare the midpoint deflection w(L/2,t)/h
(master modes combined through the mode shapes φ_i(1/2), same for the truth; slave modes excluded; float64). Per-trajectory statistics:
  rms_h   = RMS_t |ŵ−w| / h                      (thickness units)
  max_h   = max_t |ŵ−w| / h
  rel_amp = RMS_t(ŵ−w) / RMS_t(w − w̄)            (fraction of the dynamic amplitude, dimensionless)
  VPT     = first time |ŵ−w| exceeds 0.1·(dynamic peak), in true-vibration cycles (valid prediction time)
  cycles  = number of vibration cycles of the true w(t) in the window (zero crossings of (w−w̄) / 2)
Across trajectories: median + [25%,75%] (robust). Usage:
  python3 scripts/beam_midpoint_error.py --exp beam_main --seeds 0 1 [--t-end 5.0]
Output: results/<exp>/midpoint_error.csv (long table) + console table
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import eval_views, load_dataset  # noqa: E402
from src.data.integrators import rollout_sampled, rollout_sampled_graphed  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402
from src.systems.clamped_beam import SQRT12, mode_shapes  # noqa: E402


def model_field(model, mu):
    def f(y):
        return model.time_derivative(y, mu, create_graph=False)
    return f


def cycles_of(w):
    """(T,) true midpoint deflection -> number of cycles (zero crossings after mean removal / 2)."""
    s = torch.sign(w - w.mean())
    return float((s[1:] * s[:-1] < 0).sum()) / 2.0


def q25_50_75(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    return [float(torch.quantile(v, q)) for q in (0.25, 0.5, 0.75)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--t-end", type=float, default=None, help="default = data time span t_span")
    ap.add_argument("--device", default="cpu", help="evaluation runs on CPU so it does not compete with training for the GPU")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--sindy", nargs="*", default=[], help="sparse-regression coefficient files (coef_*.pt), evaluated with the same definitions; if given, only these are evaluated")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ecfg = yaml.safe_load((Path("configs") / f"{args.exp}.yaml").read_text())
    dcfg = yaml.safe_load(Path(ecfg["data_config"]).read_text())
    system = get_system(dcfg["system"]); n = system.dim
    dt, se = dcfg["data"]["dt"], dcfg["data"]["sample_every"]
    t_end = args.t_end or dcfg["data"]["t_span"]
    n_steps = int(round(t_end / dt)); assert n_steps % se == 0
    n_roll = ecfg["eval"]["n_rollout"] if "eval" in ecfg else dcfg["eval"]["n_rollout"]
    data, man = load_dataset(Path("data_cache") / dcfg["system"])
    views = eval_views(data, man)
    device = torch.device(args.device)
    phi_mid = mode_shapes(system.beta, torch.tensor([0.5], dtype=torch.float64), 0)[:n, 0]   # (n,)

    def w_mid(traj):   # (T,B,2n) -> (T,B) midpoint deflection/h (master modes)
        return (traj[..., :n].to(torch.float64) @ phi_mid) / SQRT12

    # batch: first n_roll test trajectories of each tier
    tiers, refs, mus, slc = [], [], [], {}
    off = 0
    for tier, view in views.items():
        tr = view["traj"][:, :n_roll]
        n_mu, k, T1, D = tr.shape
        refs.append(tr.reshape(n_mu * k, T1, D)); mus.append(view["mu_grid"].repeat_interleave(k, 0))
        slc[tier] = (off, off + n_mu * k); off += n_mu * k; tiers.append(tier)
    ref = torch.cat(refs).to(torch.float64); mu_all = torch.cat(mus).to(torch.float64)
    Ts = n_steps // se + 1
    ref_t = ref.permute(1, 0, 2)[:Ts]                                   # (Ts,B,2n)
    w_true = w_mid(ref_t)                                              # (Ts,B)
    w_dyn = w_true - w_true.mean(0, keepdim=True)
    amp_dyn = w_dyn.abs().max(0).values                                # (B,) dynamic peak
    cyc = torch.tensor([cycles_of(w_true[:, b]) for b in range(w_true.shape[1])])
    period_len = t_end / cyc.clamp_min(0.5)                            # mean cycle length per trajectory

    mcfg = ecfg["models"]; models = args.models or mcfg["list"]
    rows = []
    hdr = f"{'seed':4s} {'model':11s} {'tier':7s} {'rms/h median[IQR]':>26s} {'max/h med':>10s} {'rel_amp md':>10s} {'VPT(cyc) median':>14s} {'cycles median[min,max]':>26s}"
    print(f"t_end={t_end} (data time span), dt={dt}, {Ts} samples; midpoint deflection in units of h; float64\n" + hdr)
    if args.sindy:
        from scripts.sindy_model import SindyPotentialModel
        jobs = [(0, Path(pth).stem.replace("coef_", "sindy_"), SindyPotentialModel(pth, n)) for pth in args.sindy]
    else:
        jobs = []
        for seed in args.seeds:
            for name in models:
                ck = Path("results") / args.exp / str(seed) / f"{name}.pt"
                if not ck.exists():
                    print(f"skip {ck}"); continue
                model, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"])
                model.load_state_dict(torch.load(ck, map_location="cpu"))
                jobs.append((seed, name, model))
    for seed, name, model in jobs:
        if True:
            model = model.double().to(device).eval()
            with torch.no_grad():
                pred = rollout_sampled_graphed(model_field(model, mu_all.to(device)), ref[:, 0].to(device), dt, n_steps, se)
            w_hat = w_mid(pred.cpu())                                  # (Ts,B)
            err = (w_hat - w_true)                                     # (Ts,B)
            err = torch.nan_to_num(err, nan=1e9, posinf=1e9, neginf=-1e9)
            rms_h = err.pow(2).mean(0).sqrt(); max_h = err.abs().max(0).values
            rel = rms_h / w_dyn.pow(2).mean(0).sqrt()
            over = err.abs() > 0.1 * amp_dyn                            # (Ts,B)
            t_idx = torch.where(over.any(0), over.double().argmax(0), torch.full((err.shape[1],), Ts - 1))
            vpt_cycles = (t_idx.double() * (t_end / (Ts - 1))) / period_len
            for tier in tiers:
                a, b = slc[tier]
                q_rms = q25_50_75(rms_h[a:b]); q_max = q25_50_75(max_h[a:b]); q_rel = q25_50_75(rel[a:b])
                q_t10 = q25_50_75(vpt_cycles[a:b]); c = cyc[a:b]
                print(f"{seed:<4d} {name:11s} {tier:7s} {q_rms[1]:9.3g} [{q_rms[0]:.2g},{q_rms[2]:.2g}] {q_max[1]:10.3g} {q_rel[1]:10.3g} "
                      f"{q_t10[1]:8.1f} [{q_t10[0]:.1f},{q_t10[2]:.1f}] {float(c.median()):8.1f} [{float(c.min()):.0f},{float(c.max()):.0f}]", flush=True)
                for k_, (lo, md, hi) in {"rms_h": q_rms, "max_h": q_max, "rel_amp": q_rel, "vpt_cycles": q_t10}.items():
                    rows.append({"seed": seed, "model": name, "tier": tier, "metric": k_, "q25": lo, "median": md, "q75": hi,
                                 "cycles_median": float(c.median()), "cycles_min": float(c.min()), "cycles_max": float(c.max()),
                                 "t_end": t_end})
    tag = "sindy" if args.sindy else f"s{'-'.join(map(str, args.seeds))}"
    out = Path("results") / args.exp / f"midpoint_error_{tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"written {out}")


if __name__ == "__main__":
    main()
