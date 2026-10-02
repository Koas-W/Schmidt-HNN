#!/usr/bin/env python3
"""E3 Taylor-derived quantities per seed: from results/ahnn_repro/hh1p/<seed>/potential_<model>.npz compute
  phys_dev        = max over the 6 nonzero ground-truth terms of |α_pred − α_true(λ)|
  spur_with_const = max over the remaining 29 terms (constant term included) of |α_pred|
  spur_no_const   = max over the remaining 28 terms (constant term excluded) of |α_pred|
Each quantity is first reduced to its median over the 96 λ values (the seed's value) and written to <seed>/metrics_taylor.csv (tier = taylor).
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path("results/ahnn_repro/hh1p")
MODELS = ["pshnn", "pshnn_concat", "pshnn_mlp", "pshnn_silu", "sep_hnn", "ahnn", "symoden_mu"]


def main():
    for sd in sorted(ROOT.glob("[0-9]*")):
        rows = []
        for m in MODELS:
            fn = sd / f"potential_{m}.npz"
            if not fn.exists():
                continue
            d = np.load(fn); lam, tm, exps = d["lam"], d["taylor"], [tuple(int(x) for x in e) for e in d["taylor_exps"]]
            idx = {e: i for i, e in enumerate(exps)}
            true = {(2, 0, 0, 0): 0.5, (0, 2, 0, 0): 0.5, (0, 0, 2, 0): 0.5, (0, 0, 0, 2): 0.5}
            phys = np.stack([np.abs(tm[:, idx[e]] - v) for e, v in true.items()]
                            + [np.abs(tm[:, idx[(2, 1, 0, 0)]] - lam), np.abs(tm[:, idx[(0, 3, 0, 0)]] + lam / 3)], 1).max(1)
            phys_set = set(true) | {(2, 1, 0, 0), (0, 3, 0, 0)}
            spur_c = [i for e, i in idx.items() if e not in phys_set]
            spur_n = [i for e, i in idx.items() if e not in phys_set and e != (0, 0, 0, 0)]
            assert len(spur_c) == 29 and len(spur_n) == 28
            vals = {"phys_dev": np.median(phys), "spur_with_const": np.median(np.abs(tm[:, spur_c]).max(1)),
                    "spur_no_const": np.median(np.abs(tm[:, spur_n]).max(1))}
            rows += [{"model": m, "tier": "taylor", "metric": k, "value": float(v)} for k, v in vals.items()]
        with open(sd / "metrics_taylor.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "tier", "metric", "value"]); w.writeheader(); w.writerows(rows)
        print(sd, len(rows), "rows")


if __name__ == "__main__":
    main()
