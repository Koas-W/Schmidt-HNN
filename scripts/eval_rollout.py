#!/usr/bin/env python3
"""Compute trajectory L2 and energy drift on stored checkpoints (no retraining); same protocol as evaluate_model in E1/E2.

Purpose: the E3/E4 (AHNN reproduction) runs only did a light evaluation (vf_mse + potential metrics); this adds rollout_l2_* and energy_drift_*.
Usage: python3 scripts/eval_rollout.py --config configs/ahnn_hh1p.yaml --results-dir results/ahnn_repro/hh1p --seeds 0 1 2 3 4
Output: <results-dir>/<seed>/metrics_rollout.csv (model,tier,metric,value) + curves_<model>.npz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.evaluate import evaluate_model  # noqa: E402
from src.data.dataset import eval_views, load_dataset  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--t-long", type=float, default=None, help="override the energy-drift horizon (e.g. 125.6 = 20 periods of 2*pi)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dcfg = yaml.safe_load(Path(cfg["data_config"]).read_text()) if "data_config" in cfg else cfg
    system = get_system(dcfg["system"])
    data, manifest = load_dataset(Path("data_cache") / dcfg["system"])
    views = eval_views(data, manifest)
    mcfg = cfg["models"]; models = args.models or mcfg["list"]
    eval_cfg = dict(cfg.get("eval") or dcfg["eval"])
    if args.t_long is not None:
        eval_cfg["rollout_t_long"] = args.t_long
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.results_dir}] device={device} t_short={eval_cfg['rollout_t_short']} t_long={eval_cfg['rollout_t_long']} "
          f"n_rollout={eval_cfg['n_rollout']} tiers={list(views)}", flush=True)
    for seed in args.seeds:
        out = Path(args.results_dir) / str(seed); rows = []
        if args.models and (out / "metrics_rollout.csv").exists():      # incremental: keep rows of other models
            with open(out / "metrics_rollout.csv", newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r["model"] not in models]
        for name in models:
            ck = out / f"{name}.pt"
            if not ck.exists():
                print(f"  skip {ck}"); continue
            model, _ = build_model(name, system.dim, system.mu_dim, mcfg, mcfg["target_params"])
            model.load_state_dict(torch.load(ck, map_location="cpu")); model.to(device).eval()
            metrics, curves = evaluate_model(model, system, views, dcfg["data"], eval_cfg, device)
            keep = {k: v for k, v in metrics.items() if k[1] != "vf_mse"}          # vf_mse already in metrics.csv
            rows += [{"model": name, "tier": t, "metric": m, "value": v} for (t, m), v in keep.items()]
            np.savez(out / f"curves_{name}.npz", **{k: v.cpu().numpy() for k, v in curves.items()})
            print(f"  seed {seed} {name:13s} " + " ".join(f"{t}/{m}={v:.3g}" for (t, m), v in keep.items() if m.endswith("final")), flush=True)
        with open(out / "metrics_rollout.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "tier", "metric", "value"]); w.writeheader(); w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main()
