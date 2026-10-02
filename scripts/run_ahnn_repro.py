#!/usr/bin/env python3
"""AHNN reproduction runner: parameter-matched training + AHNN metrics (⟨ΔV⟩/Taylor) + full artifacts.

Reuses the existing building blocks (load_dataset / build_model / train_with_lr_sweep / evaluate_model)
and the potential metrics in src.analysis.potential; no early stopping, full convergence budget, 5 seeds.

Output layout: results/ahnn_repro/<name>/<seed>/
    metrics.csv             vf_mse (three tiers) + ⟨ΔV⟩ region summaries
    <model>.pt              best-val checkpoint
    history_<model>.npz     full training curve per lr (step,train,val) + chosen_lr
    potential_<model>.npz   ⟨ΔV⟩ dense-sweep curve/grid + Taylor coefficients (Hamiltonian models)
    curves_<model>.npz      rollout/energy-drift evaluation curves
    config_snapshot.yaml

Usage:
    python3 scripts/run_ahnn_repro.py --config configs/ahnn_hh1p.yaml
    # smoke test: --seeds 0 --models pshnn ahnn --max-steps 2000
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.potential import deltaV_curve, taylor_coeffs, monomial_exponents  # noqa: E402
from src.data.dataset import eval_views, load_dataset, training_views  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.train.trainer import train_with_lr_sweep, evaluate_loss  # noqa: E402
from src.systems import get_system  # noqa: E402

RESULTS_ROOT = Path("results/ahnn_repro")
NO_HAMILTONIAN = {"mlp_ode"}  # regresses the vector field directly, no scalar H => skip potential metrics


def potential_metrics(name, model, system, dcfg, device):
    """⟨ΔV⟩ dense sweep + Taylor coefficients (Hamiltonian models only). Returns (npz_dict, csv_rows)."""
    if name in NO_HAMILTONIAN:
        return None, []
    ham = lambda q, p, mu: model.hamiltonian(q, p, mu)  # noqa: E731
    dv = dcfg["deltaV"]
    g = torch.linspace(dv["lam_min"], dv["lam_max"], dv["n_lam"],
                       device=device, dtype=torch.float32)
    if system.mu_dim == 1:                                   # one-parameter curve
        lam = g.unsqueeze(1)
        curve = deltaV_curve(ham, system, lam).cpu().numpy()
        coeffs, exps = taylor_coeffs(ham, lam)
        npz = {"lam": g.cpu().numpy(), "deltaV": curve,
               "taylor": coeffs.cpu().numpy(),
               "taylor_exps": np.array(monomial_exponents())}
        gl = g.cpu().numpy()
        inb = (gl >= 0.2) & (gl <= 0.8)                      # training λ range
        rows = [{"tier": "deltaV", "metric": "train_region",
                 "value": float(curve[inb].mean())},
                {"tier": "deltaV", "metric": "extrap",
                 "value": float(curve[~inb].mean())}]
    else:                                                   # two-parameter grid
        mu2 = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        n = dv["n_lam"]
        grid = deltaV_curve(ham, system, mu2, grid_n=97).reshape(n, n).cpu().numpy()
        npz = {"lam": g.cpu().numpy(), "deltaV_grid": grid}
        gl = g.cpu().numpy()
        c = (gl >= 0.2) & (gl <= 0.8)
        # with the integration domain restricted to the bound well the relative ⟨ΔV⟩ is well defined everywhere; central region / full plane use the same protocol
        rows = [{"tier": "deltaV", "metric": "central_region",
                 "value": float(grid[np.ix_(c, c)].mean())},
                {"tier": "deltaV", "metric": "full_plane",
                 "value": float(grid.mean())}]
    return npz, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    system = get_system(cfg["system"])
    data, manifest = load_dataset(Path("data_cache") / cfg["system"])
    tviews = training_views(data, manifest)
    eviews = eval_views(data, manifest)

    tcfg = dict(cfg["train"])
    if args.max_steps:
        tcfg["max_steps"] = args.max_steps
    seeds = args.seeds if args.seeds is not None else tcfg["seeds"]
    mcfg = cfg["models"]
    run_models = args.models if args.models else mcfg["list"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if tcfg.get("device", "auto") == "auto" else torch.device(tcfg["device"])
    name = cfg["name"]
    print(f"[ahnn_repro/{name}] device={device} seeds={seeds} models={run_models} "
          f"train_N={tviews['train']['x'].shape[0]} steps={tcfg['max_steps']} "
          f"early_stop={tcfg.get('early_stop', True)}", flush=True)

    for seed in seeds:
        out = RESULTS_ROOT / name / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        rows, infos = [], {}
        old = out / "metrics.csv"
        if old.exists():
            with open(old, newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r["model"] not in run_models]
        for mname in run_models:
            t0 = time.time()

            def build_fn(_n=mname):
                m, _ = build_model(_n, system.dim, system.mu_dim, mcfg,
                                   mcfg["target_params"])
                return m

            _, info = build_model(mname, system.dim, system.mu_dim, mcfg,
                                  mcfg["target_params"])
            model, tinfo = train_with_lr_sweep(
                build_fn, tviews["train"], tviews["val"], tcfg, seed, device)
            model.eval()
            # light evaluation: vf_mse (derivative MSE) on three tiers + AHNN potential metrics; the rollout
            # evaluation is expensive (long autograd-through-H) and not part of the AHNN protocol; orbit
            # figures are generated separately in make_ahnn_figs.py at specified λ.
            metrics = {(tier, "vf_mse"): evaluate_loss(model, view, device)
                       for tier, view in eviews.items()}
            pnpz, prows = potential_metrics(mname, model, system, cfg, device)

            torch.save(model.state_dict(), out / f"{mname}.pt")
            np.savez(out / f"history_{mname}.npz",
                     chosen_lr=tinfo["chosen_lr"],
                     **{f"lr_{lr}": np.array(h, dtype=np.float64)
                        for lr, h in tinfo["histories"].items()})
            if pnpz is not None:
                np.savez(out / f"potential_{mname}.npz", **pnpz)
            for (tier, metric), v in metrics.items():
                rows.append({"model": mname, "tier": tier, "metric": metric,
                             "value": v})
            for r in prows:
                rows.append({"model": mname, **r})
            infos[mname] = {**info, **{k: tinfo[k] for k in
                                       ("chosen_lr", "best_val", "n_evals")}}
            dvs = {r["metric"]: r["value"] for r in prows}
            dv_str = f"ΔV[{list(dvs)[0]}]={list(dvs.values())[0]:.3e} " if dvs else ""
            print(f"[{name} s={seed}] {mname:12s} w={info['width']:4d} "
                  f"p={info['n_params']} lr={tinfo['chosen_lr']:.0e} "
                  f"val={tinfo['best_val']:.2e} "
                  f"vf_ex={metrics.get(('extrap', 'vf_mse'), float('nan')):.2e} "
                  f"{dv_str}({time.time() - t0:.0f}s)", flush=True)

        with open(out / "metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "tier", "metric", "value"])
            w.writeheader()
            w.writerows(rows)
        snap_f = out / "config_snapshot.yaml"                      # incremental mode: keep run_info of models not rerun
        if snap_f.exists():
            try: infos = {**(yaml.safe_load(snap_f.read_text()).get("run_info") or {}), **infos}
            except Exception: pass
        (out / "config_snapshot.yaml").write_text(
            yaml.dump({"config": cfg, "run_info": infos, "device": str(device),
                       "max_steps": tcfg["max_steps"]},
                      allow_unicode=True, sort_keys=False))
    print(f"[ahnn_repro/{name}] done, {len(seeds)} seeds", flush=True)


if __name__ == "__main__":
    main()
