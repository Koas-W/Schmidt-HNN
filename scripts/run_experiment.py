#!/usr/bin/env python3
"""Run one experiment: models x seeds, train + evaluate + write results/<exp>/<seed>/.

Artifacts: {model}.pt, curves_{model}.npz, history_{model}.npz ((step,train,val) curve per lr +
chosen_lr), metrics.csv, config_snapshot.yaml (with chosen_lr/best_val/stop_step).

Usage:
    python3 scripts/run_experiment.py --config configs/e2_double_pendulum.yaml
    # smoke test: --seeds 0 --max-steps 300
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

from src.analysis.evaluate import evaluate_model  # noqa: E402
from src.data.dataset import eval_views, load_dataset, training_views  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.train.trainer import train_with_lr_sweep  # noqa: E402
from src.systems import get_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="run only these models and merge into the existing metrics.csv (incremental baselines)")
    args = ap.parse_args()

    ecfg = yaml.safe_load(Path(args.config).read_text())
    dcfg = yaml.safe_load(Path(ecfg["data_config"]).read_text())
    system = get_system(dcfg["system"])
    data, manifest = load_dataset(Path("data_cache") / dcfg["system"])
    tviews = training_views(data, manifest)
    eviews = eval_views(data, manifest)

    tcfg = dict(ecfg["train"])
    if args.max_steps:
        tcfg["max_steps"] = args.max_steps
    seeds = args.seeds if args.seeds is not None else tcfg["seeds"]
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu") if tcfg.get("device", "auto") == "auto" \
        else torch.device(tcfg["device"])
    exp = ecfg["experiment"]
    mcfg = ecfg["models"]
    run_models = args.models if args.models else mcfg["list"]
    print(f"[{exp}] device={device}, seeds={seeds}, models={run_models}, "
          f"train N={tviews['train']['x'].shape[0]}")

    for seed in seeds:
        out = Path("results") / exp / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        rows, infos = [], {}
        # incremental mode: keep rows of models not in this run from the existing metrics
        old_csv = out / "metrics.csv"
        if old_csv.exists():
            with open(old_csv, newline="") as fh:
                rows = [r for r in csv.DictReader(fh)
                        if r["model"] not in run_models]
        for name in run_models:
            t0 = time.time()

            def build_fn(_name=name):
                model, _ = build_model(_name, system.dim, system.mu_dim,
                                       mcfg, mcfg["target_params"])
                return model

            _, info = build_model(name, system.dim, system.mu_dim, mcfg,
                                  mcfg["target_params"])
            model, tinfo = train_with_lr_sweep(
                build_fn, tviews["train"], tviews["val"], tcfg, seed, device)
            metrics, curves = evaluate_model(model, system, eviews,
                                             dcfg["data"], dcfg["eval"],
                                             device)
            torch.save(model.state_dict(), out / f"{name}.pt")
            np.savez(out / f"curves_{name}.npz",
                     **{k: v.detach().numpy() for k, v in curves.items()})
            # full training curve per lr (step, train, val) + chosen lr + stop step
            hist_npz = {"chosen_lr": np.array(tinfo["chosen_lr"])}
            for lr_key, hist in tinfo["histories"].items():
                arr = np.asarray(hist, dtype=np.float64).reshape(-1, 3)
                hist_npz[f"lr_{lr_key}"] = arr          # columns: step, train_loss, val_loss
            np.savez(out / f"history_{name}.npz", **hist_npz)
            stop_step = int(tinfo["history"][-1][0]) if tinfo["history"] else 0
            for (tier, metric), v in metrics.items():
                rows.append({"model": name, "tier": tier,
                             "metric": metric, "value": v})
            infos[name] = {**info,
                           "chosen_lr": tinfo["chosen_lr"],
                           "best_val": tinfo["best_val"],
                           "n_evals": tinfo["n_evals"],
                           "stop_step": stop_step}   # curves are not stored in the yaml snapshot
            print(f"[{exp} seed={seed}] {name:8s} width={info['width']:4d} "
                  f"params={info['n_params']} lr={tinfo['chosen_lr']:.0e} "
                  f"val={tinfo['best_val']:.3e} stop={infos[name]['stop_step']} "
                  f"extrap_vf={metrics.get(('extrap', 'vf_mse'), float('nan')):.3e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        with open(out / "metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "tier", "metric",
                                               "value"])
            w.writeheader()
            w.writerows(rows)
        snap_f = out / "config_snapshot.yaml"                      # incremental mode: keep run_info of models not rerun
        if snap_f.exists():
            try: infos = {**(yaml.safe_load(snap_f.read_text()).get("run_info") or {}), **infos}
            except Exception: pass
        snapshot = {"experiment": ecfg, "data": dcfg, "run_info": infos,
                    "device": str(device)}
        (out / "config_snapshot.yaml").write_text(
            yaml.dump(snapshot, allow_unicode=True, sort_keys=False))
    print(f"[{exp}] done, {len(seeds)} seeds")


if __name__ == "__main__":
    main()
