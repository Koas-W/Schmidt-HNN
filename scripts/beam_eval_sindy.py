#!/usr/bin/env python3
"""Standard metrics for the sparse-regression potential (same definitions as the networks): vf_mse, rollout L2(t_short), energy drift (t_long).
Usage: python3 scripts/beam_eval_sindy.py results/beam_sindy/coef_potential_dq4_dmu3.pt [...]
Output: results/beam_sindy/metrics_sindy.csv
"""
import csv, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sindy_model import SindyPotentialModel  # noqa: E402
from src.analysis.evaluate import evaluate_model  # noqa: E402
from src.data.dataset import eval_views, load_dataset  # noqa: E402
from src.systems import get_system  # noqa: E402

torch.set_num_threads(12)
dcfg = yaml.safe_load(open("configs/clamped_beam.yaml"))
system = get_system(dcfg["system"])
data, man = load_dataset(Path("data_cache") / dcfg["system"]); views = eval_views(data, man)
rows = []
for pth in sys.argv[1:]:
    name = Path(pth).stem.replace("coef_", "sindy_")
    model = SindyPotentialModel(pth, system.dim)
    metrics, _ = evaluate_model(model, system, views, dcfg["data"], dcfg["eval"], torch.device("cpu"))
    for (tier, k), v in metrics.items():
        rows.append({"model": name, "tier": tier, "metric": k, "value": v})
    for tier in ("train", "interp", "extrap"):
        print(f"{name} {tier:7s} vf_mse={metrics[(tier,'vf_mse')]:.4g} L2(2.5)={metrics[(tier,'rollout_l2_final')]:.4g} "
              f"Edrift(5)={metrics[(tier,'energy_drift_final')]:.3g}", flush=True)
out = Path("results/beam_sindy/metrics_sindy.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model", "tier", "metric", "value"]); w.writeheader(); w.writerows(rows)
print(f"written {out}")
