#!/usr/bin/env python3
"""E6 (FPU chain, grid design) post-hoc evaluation: fixed-time trajectory L2, energy drift and VPT on stored checkpoints (no retraining).

The evaluation sets (interp LHS24, three shells of 8 points each, evaluation ICs) are generated from fixed seeds independent of the
training seed and are rebuilt exactly (reusing PhysFPU / gen_tier / fixed_eval_sets from scripts/run_fpu.py).
Long-horizon ground truth is integrated here: same ICs, same RK4 (dt=0.01, float64) up to t_long, same storage interval as the data (0.4).
Per-trajectory results are written to npz, so table values at any time can be read from them; the csv holds the default times.

Metrics (per trajectory, then aggregated over trajectories per tier):
  rollout_l2@t     = ‖x̂(t) − x(t)‖₂ (mean over trajectories)           t in {t_short=20 (end of the data window), 3 linear periods ≈ 18.85}
  energy_drift@t   = |H(x̂(t)) − H(x̂(0))| / |H(x̂(0))| (true H, mean over trajectories)  t = t_long
  vpt_cycles       = first time ‖x̂ − x‖₂ exceeds 0.1 × RMS_t‖x(t)‖₂ (of that trajectory) / 2π (linear period), median over trajectories (q25/q75 also stored)
Usage: python3 scripts/eval_fpu_grid.py --ks 2 3 4 --seeds 0 1 2 3 4 [--t-long 125.66]
Output: results/e6_fpu/long/metrics_long_grid_k<k>_s<seed>.csv + traj_grid_k<k>_s<seed>_<model>.npz
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fpu as rfs  # noqa: E402
from src.data.integrators import rollout_sampled, rollout_sampled_graphed  # noqa: E402
from src.data.mu_design import fixed_eval_sets, full_grid  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.synthetic.planted import PlantedFamily  # noqa: E402

OUT = rfs.OUT / "long"
MODELS = ["pshnn", "pshnn_concat", "pshnn_mlp", "sep_hnn", "symoden_mu", "ahnn", "mlp_ode"]   # the seven models of the unified protocol
PERIOD = 2 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--t-long", type=float, default=20 * PERIOD, help="long horizon (default 20 linear periods, about 125.66)")
    ap.add_argument("--n-chain", type=int, default=8)
    ap.add_argument("--sample-every", type=int, default=40)
    ap.add_argument("--models", nargs="*", default=None, help="evaluate only these models and merge into the existing metrics_long_grid_k<k>_s<seed>.csv")
    args = ap.parse_args()
    models = args.models or MODELS
    rfs.N_CHAIN, rfs.SAMPLE_EVERY = args.n_chain, args.sample_every
    n = args.n_chain
    for _m in ("pshnn", "pshnn_mlp", "pshnn_silu"):
        rfs.MODELS_CFG[_m]["R"] = 3; rfs.MODELS_CFG[_m]["bias_head"] = True   # matches config_grid_k*.json of the unified protocol (R=3; the M representation is already in MODELS_CFG)
    fam = PlantedFamily.load(Path("data_cache") / f"fpu_family_n{n}.npz"); sys_ = rfs.PhysFPU(fam)
    mb = fam.phi.box; box_mu = [(mb[i, 0].item(), mb[i, 1].item()) for i in range(rfs.M_MU)]
    fe = fixed_eval_sets(box_mu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt, se = rfs.DT, args.sample_every
    n_long = int(round(args.t_long / dt)) // se * se
    t_axis = torch.arange(n_long // se + 1, dtype=torch.float64) * dt * se
    i_short = int(round(rfs.T_SPAN / (dt * se)))                       # t = 20
    i_3per = int(round(3 * PERIOD / (dt * se)))                        # 3 linear periods, about 18.8
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={device} t_long={t_axis[-1]:.2f} ({t_axis[-1]/PERIOD:.1f} periods), {len(t_axis)} stored points", flush=True)

    for k in args.ks:
        eval_mu = {"train": full_grid(k, box_mu), "interp": fe["interp"], "ex1": fe["ex1"], "ex2": fe["ex2"], "ex3": fe["ex3"]}
        packs = {tier: rfs.gen_tier(sys_, ml, rfs.TRAJ_EVAL, 7777)[1] for tier, ml in eval_mu.items()}
        tiers = list(packs); sizes = [packs[t][0].shape[0] for t in tiers]
        y0 = torch.cat([packs[t][0] for t in tiers]); mu = torch.cat([packs[t][1] for t in tiers])
        def f_true(y):                                                                      # ground truth integrated on CPU in float64
            qd, pd = sys_.vector_field(y[:, :n], y[:, n:], mu)
            return torch.cat([qd, pd], -1)
        with torch.no_grad():
            truth = rollout_sampled(f_true, y0, dt, n_long, se)                             # (T, B, 2n) float64
        Hx = lambda tr: sys_.hamiltonian(tr.reshape(-1, 2 * n)[:, :n], tr.reshape(-1, 2 * n)[:, n:], mu.repeat(tr.shape[0], 1)).reshape(tr.shape[0], -1)
        H_true = Hx(truth); dr_true = ((H_true - H_true[0]).abs() / H_true[0].abs()).max().item()
        amp = truth.pow(2).sum(-1).mean(0).sqrt()                                           # (B,) phase-space RMS amplitude
        print(f"[k={k}] {y0.shape[0]} evaluation trajectories, ground-truth long-horizon drift {dr_true:.1e}", flush=True)
        mu32 = mu.to(torch.float32).to(device)
        for seed in args.seeds:
            rows = []
            fl_old = OUT / f"metrics_long_grid_k{k}_s{seed}.csv"
            if args.models and fl_old.exists():                                       # incremental: keep rows of other models
                with open(fl_old, newline="") as fh:
                    rows = [r for r in csv.DictReader(fh) if r["model"] not in models]
            for name in models:
                ck = rfs.OUT / f"ckpt_grid_k{k}_s{seed}_{name}.pt"   # per-seed checkpoint (run_fpu naming)
                if not ck.exists():
                    print(f"  skip {ck}"); continue
                model, _ = build_model(name, n, rfs.M_MU, rfs.MODELS_CFG, rfs.MODELS_CFG["target_params"])
                model.load_state_dict(torch.load(ck, map_location="cpu")); model.to(device).eval()
                with torch.no_grad():
                    pred = rollout_sampled_graphed(lambda y: model.time_derivative(y, mu32, create_graph=False),
                                                   y0.to(torch.float32).to(device), dt, n_long, se).cpu().double()
                pred = torch.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6)
                l2 = (pred - truth).pow(2).sum(-1).sqrt()                                   # (T, B)
                Hp = Hx(pred); drift = (Hp - Hp[0]).abs() / Hp[0].abs().clamp_min(1e-12)   # (T, B)
                over = l2 > 0.1 * amp
                t_idx = torch.where(over.any(0), over.double().argmax(0), torch.full((l2.shape[1],), l2.shape[0] - 1))
                vpt = t_axis[t_idx] / PERIOD
                np.savez(OUT / f"traj_grid_k{k}_s{seed}_{name}.npz", t=t_axis.numpy(), l2=l2.numpy(), drift=drift.numpy(),
                         vpt=vpt.numpy(), tiers=np.array(tiers), sizes=np.array(sizes))
                off = 0
                for tier, sz in zip(tiers, sizes):
                    sl = slice(off, off + sz); off += sz
                    vals = {"rollout_l2_t20": l2[i_short, sl].mean().item(), "rollout_l2_3per": l2[i_3per, sl].mean().item(),
                            "rollout_l2_long": l2[-1, sl].mean().item(),
                            "energy_drift_final": drift[-1, sl].mean().item(), "energy_drift_max": drift[:, sl].max(0).values.mean().item(),
                            "vpt_cycles": vpt[sl].median().item(), "vpt_q25": vpt[sl].quantile(0.25).item(), "vpt_q75": vpt[sl].quantile(0.75).item()}
                    rows += [{"seed": seed, "model": name, "tier": tier, "metric": m_, "value": v} for m_, v in vals.items()]
                print(f"  k={k} seed={seed} {name:13s} " + " ".join(
                    f"{r['tier']}:L2={r['value']:.3g}" for r in rows if r["model"] == name and r["metric"] == "rollout_l2_t20") +
                    "  VPT(interp/ex3)=" + "/".join(f"{r['value']:.1f}" for r in rows if r["model"] == name and r["metric"] == "vpt_cycles" and r["tier"] in ("interp", "ex3")), flush=True)
            with open(OUT / f"metrics_long_grid_k{k}_s{seed}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["seed", "model", "tier", "metric", "value"]); w.writeheader(); w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
