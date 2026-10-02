#!/usr/bin/env python3
"""Generate the ground-truth dataset for a config and write it to data_cache/<system>/.

Usage: python3 scripts/generate_data.py --config configs/double_pendulum.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import TIERS, generate_dataset, load_dataset  # noqa: E402
from src.data.integrators import energy_drift  # noqa: E402
from src.systems import get_system  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path("data_cache") / config["system"]
    generate_dataset(config, out_dir)

    # acceptance check: ground-truth energy drift on the stored trajectories must be <= 1e-8
    data, manifest = load_dataset(out_dir)
    system = get_system(config["system"])
    print(f"[{config['system']}] dataset written to {out_dir}/")
    worst = 0.0
    for tier in TIERS:
        if f"mu_{tier}" not in data:
            continue
        traj, mu = data[f"traj_{tier}"], data[f"mu_{tier}"]
        n_mu, n_traj, T, D = traj.shape
        flat = traj.reshape(n_mu * n_traj, T, D).permute(1, 0, 2)
        # mu must enter H in float64 (avoids float32 mixed-precision artifacts, see mu_split.py)
        mu_rep = mu.to(torch.float64).repeat_interleave(n_traj, dim=0)
        drift = energy_drift(system, flat, mu_rep).max().item()
        worst = max(worst, drift)
        print(f"  {tier:7s} mu_points={n_mu:3d} traj/mu={n_traj:3d} "
              f"points/traj={T} max_energy_drift={drift:.2e}")
    if config["data"].get("noise_std", 0.0) == 0.0:
        assert worst <= 1e-8, f"energy drift {worst:.2e} exceeds the acceptance threshold 1e-8"
        print("  acceptance passed (drift <= 1e-8)")
    else:
        print("  (noise_std>0: train/val roles are noisy, drift check is informational only)")


if __name__ == "__main__":
    main()
