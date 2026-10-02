#!/usr/bin/env python3
"""Supplementary evaluation for the beam experiment: vector-field error per extrapolation range and engineering quantities.

For each results/<exp>/<seed>/{model}.pt:
  1) vector-field MSE split by extrapolation range ex1/ex2/ex3 (manifest["extrap_shells"]);
  2) static equilibrium: root of ṗ=0 at p=0 under the learned vector field (Newton with a central-difference Jacobian),
     reported as the relative L2 error of the master-mode coordinates against the true static equilibrium;
  3) backbone curve ω₁(A;μ): A/h in AMPS; truth and model use the same initial condition (first-mode turning point of the
     true system, at rest), the same RK4 and the same estimator (src/analysis/backbone.py); relative error reported.
Evaluation parameter points = interp (24) + extrap (24, labelled by extrapolation range).

Usage: python3 scripts/beam_eval_shells.py --exp beam_main [--seeds 0 1 2 3 4] [--models ...]
Output: results/<exp>/eval_beam.csv (long table: seed, model, tier, shell, metric, value)
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.backbone import estimate_omega1, rest_start_ic  # noqa: E402
from src.data.dataset import eval_views, load_dataset  # noqa: E402
from src.data.integrators import rollout_sampled, rollout_sampled_graphed, system_field  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402
from src.train.trainer import evaluate_loss  # noqa: E402

AMPS = [1.0, 1.5, 2.0, 2.5, 3.0]
BB_DT, BB_T = 1e-3, 6.0


def model_field(model, mu):
    def f(y):
        return model.time_derivative(y, mu, create_graph=False)
    return f


def model_static_equilibrium(model, q0, mu, iters=25):
    """Newton iteration for ṗ(q, p=0; μ) = 0, started from the true static equilibrium (float64)."""
    n = q0.shape[1]
    q = q0.clone()
    for _ in range(iters):
        def F(qq):
            x = torch.cat([qq, torch.zeros_like(qq)], -1)
            return model.time_derivative(x, mu, create_graph=False)[:, n:]
        r = F(q)
        if float(r.abs().max()) < 1e-10:
            break
        # Jacobian by central differences: HNN-type models run time_derivative under enable_grad+detach,
        # so the output carries no graph and autograd.functional.jacobian silently returns 0 (static-equilibrium error would read as exactly 0)
        h = 1e-6 * (1.0 + q.abs())
        cols = []
        for j in range(n):
            e = torch.zeros_like(q); e[:, j] = h[:, j]
            cols.append((F(q + e) - F(q - e)) / (2 * h[:, j:j + 1]))
        J = torch.stack(cols, dim=-1)                      # (B, n, n)
        # the learned field may have a singular Jacobian at some evaluation points (model error): pseudo-inverse least-squares Newton step,
        # with the step norm capped at 0.5 x the initial norm to prevent divergence; the static-equilibrium error is the reported metric
        step = (torch.linalg.pinv(J) @ r.unsqueeze(-1)).squeeze(-1)
        cap = 0.5 * q0.norm(dim=-1, keepdim=True)
        step = step * torch.clamp(cap / (step.norm(dim=-1, keepdim=True) + 1e-30), max=1.0)
        q = q - step
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--config", default=None, help="default configs/<exp>.yaml; a results directory config_snapshot.yaml may be given instead")
    ap.add_argument("--sindy", nargs="*", default=[], help="sparse-regression coefficient files (coef_*.pt), evaluated for static equilibrium and backbone frequency with the same definitions; if given, only these are evaluated")
    args = ap.parse_args()
    root = Path("results") / args.exp
    cfg_path = Path(args.config) if args.config else Path("configs") / f"{args.exp}.yaml"
    ecfg = yaml.safe_load(cfg_path.read_text()); ecfg = ecfg["experiment"] if isinstance(ecfg.get("experiment"), dict) else ecfg
    dcfg = yaml.safe_load(Path(ecfg["data_config"]).read_text())
    system = get_system(dcfg["system"])
    data, manifest = load_dataset(Path("data_cache") / dcfg["system"])
    views = eval_views(data, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = ecfg["models"]
    models = args.models or mcfg["list"]

    # ---- split by extrapolation range (extrap view is flattened in μ-row order: (n_mu, n_traj, T)) ----
    shells = manifest.get("extrap_shells")
    shell_views = {}
    if shells:
        ex = views["extrap"]
        n_mu = ex["mu_grid"].shape[0]
        per_mu = ex["x"].shape[0] // n_mu
        lab = torch.tensor(shells).repeat_interleave(per_mu)
        for k in sorted(set(shells)):
            m = lab == k
            shell_views[f"ex{k}"] = {"x": ex["x"][m], "mu": ex["mu"][m], "dx": ex["dx"][m]}

    # ---- evaluation parameter points and ground truth for the engineering quantities ----
    mu_eval = torch.cat([views["interp"]["mu_grid"], views["extrap"]["mu_grid"]]).to(torch.float64)
    tier_of = ["interp"] * views["interp"]["mu_grid"].shape[0] + \
              [f"ex{k}" for k in shells] if shells else ["extrap"] * views["extrap"]["mu_grid"].shape[0]
    a_s_true = system.static_equilibrium(mu_eval)
    amps = torch.tensor(AMPS, dtype=torch.float64)
    mu_bb = mu_eval.repeat_interleave(len(AMPS), 0)
    amp_bb = amps.repeat(mu_eval.shape[0])
    y0 = rest_start_ic(system, mu_bb, amp_bb)
    n_steps = int(round(BB_T / BB_DT))
    with torch.no_grad():
        y_true = rollout_sampled(system_field(system, mu_bb), y0, BB_DT, n_steps, 1)
    om_true = estimate_omega1(y_true[:, :, 0], BB_DT)

    rows = []
    if args.sindy:   # sparse regression: no seed, recorded as seed 0; model name = coefficient file name
        from scripts.sindy_model import SindyPotentialModel
        jobs = [(0, Path(pth).stem.replace("coef_", "sindy_"), SindyPotentialModel(pth, system.dim)) for pth in args.sindy]
    else:
        jobs = []
        for seed in args.seeds:
            for name in models:
                ck = root / str(seed) / f"{name}.pt"
                if not ck.exists():
                    print(f"skip {ck} (missing)"); continue
                model, _ = build_model(name, system.dim, system.mu_dim, mcfg, mcfg["target_params"])
                model.load_state_dict(torch.load(ck, map_location="cpu")); jobs.append((seed, name, model))
    for seed, name, model in jobs:
        if True:
            model = model.double().to(device).eval()
            # 1) vector-field MSE per extrapolation range
            for k, v in shell_views.items():
                vv = {kk: t.double() for kk, t in v.items()}
                rows.append({"seed": seed, "model": name, "tier": "extrap", "shell": k,
                             "metric": "vf_mse", "value": evaluate_loss(model, vv, device)})
            # 2) static equilibrium
            q_hat = model_static_equilibrium(model, a_s_true.to(device), mu_eval.to(device)).cpu()
            rel = ((q_hat - a_s_true).norm(dim=-1) / a_s_true.norm(dim=-1))
            for i, t in enumerate(tier_of):
                rows.append({"seed": seed, "model": name, "tier": "extrap" if t.startswith("ex") else t,
                             "shell": t if t.startswith("ex") else "", "metric": "static_eq_rel_l2",
                             "value": float(rel[i])})
            # 3) backbone curve
            with torch.no_grad():
                y_hat = rollout_sampled_graphed(model_field(model, mu_bb.to(device)), y0.to(device),
                                        BB_DT, n_steps, 1).cpu()
            om_hat = estimate_omega1(y_hat[:, :, 0], BB_DT)
            err = ((om_hat - om_true).abs() / om_true.abs()).reshape(mu_eval.shape[0], len(AMPS))
            for i, t in enumerate(tier_of):
                for j, A in enumerate(AMPS):
                    rows.append({"seed": seed, "model": name,
                                 "tier": "extrap" if t.startswith("ex") else t,
                                 "shell": t if t.startswith("ex") else "",
                                 "metric": f"omega1_rel_err_A{A:g}", "value": float(err[i, j])})
            print(f"[{args.exp} seed={seed}] {name:13s} static-eq rel err median={rel.median():.2e}  "
                  f"backbone ω₁ rel err median={err.median():.2e}", flush=True)
    out = root / ("eval_beam_sindy.csv" if args.sindy else "eval_beam.csv" if not args.models else "eval_beam_" + "-".join(args.models) + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "model", "tier", "shell", "metric", "value"])
        w.writeheader(); w.writerows(rows)
    print(f"written {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
