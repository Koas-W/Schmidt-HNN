#!/usr/bin/env python3
"""Long-horizon supplementary evaluation (uses stored checkpoints, no retraining).

Literature basis:
- HNN: per-step coordinate MSE of the rollout + total energy curve; the short-horizon version exists already.
  Here the pointwise L2 is extended to 50 characteristic periods, which is only meaningful for regular systems
  (pendulum); for chaotic systems (double pendulum) the pointwise error saturates exponentially and the
  literature (AHNN) evaluates dynamical invariants instead; long-horizon physical consistency is covered by energy drift.
- AHNN Eq. 4: relative potential/Hamiltonian error ⟨ΔH⟩ = avg|Ĥc−Hc|/avg|Hc|,
  constant alignment: H is identifiable only up to a constant, and under the separable structure the constant
  may vary with μ => center the test states per (tier, μ point) group before comparing.
- VPT (valid prediction time): first time the L2 exceeds a threshold θ; θ = 0.1 × RMS phase-space
  amplitude of the ground-truth trajectories (same θ for all models within an experiment).

Output: results/<exp>/<seed>/metrics_long.csv + long_curves_<model>.npz.
Usage: python3 scripts/eval_long.py --config configs/e1_pendulum.yaml \
        --seeds 0 1 2 [--device cpu]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Resource constraint (measured): batch-8 model rollouts are bound by per-step dispatch;
# CPU with 2 threads (3.9 ms/step) is 3x faster than CUDA (12.8 ms/step) and does not contend with the main experiments.
torch.set_num_threads(2)
os.nice(10)

from src.analysis.evaluate import model_field  # noqa: E402
from src.data.dataset import eval_views, load_dataset  # noqa: E402
from src.data.integrators import rollout_sampled, system_field  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402


def load_checkpoint(name, system, mcfg, ckpt_path, device):
    model, _ = build_model(name, system.dim, system.mu_dim, mcfg,
                           mcfg["target_params"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                     weights_only=True))
    return model.to(device).eval()


def h_regression_error(model, system, views) -> dict:
    """AHNN Eq. 4 style: relative Ĥ error after constant alignment per (tier, μ point)."""
    out = {}
    for tier, view in views.items():
        traj, mu_grid = view["traj"], view["mu_grid"]   # (n_mu,k,T,2n)
        n_mu, k, T, D = traj.shape
        n = system.dim
        errs, refs = [], []
        for i in range(n_mu):
            x = traj[i].reshape(k * T, D).to(torch.float32)
            mu_rep = mu_grid[i].expand(k * T, -1).to(torch.float32)
            with torch.no_grad():
                H_hat = model.hamiltonian(x[:, :n], x[:, n:], mu_rep).double()
            H_true = system.hamiltonian(x.double()[:, :n], x.double()[:, n:],
                                        mu_grid[i].expand(k * T, -1))
            dH = (H_hat - H_hat.mean()) - (H_true - H_true.mean())
            errs.append(dH.abs().mean())
            refs.append((H_true - H_true.mean()).abs().mean())
        out[tier] = (torch.stack(errs).mean()
                     / torch.stack(refs).mean()).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: small-batch rollout is faster on CPU and does not occupy the GPU")
    ap.add_argument("--vpt-frac", type=float, default=0.1)
    ap.add_argument("--models", nargs="*", default=None, help="evaluate only these models and merge into the existing metrics_long.csv")
    args = ap.parse_args()

    ecfg = yaml.safe_load(Path(args.config).read_text())
    dcfg = yaml.safe_load(Path(ecfg["data_config"]).read_text())
    system = get_system(dcfg["system"])
    data, manifest = load_dataset(Path("data_cache") / dcfg["system"])
    views = eval_views(data, manifest)
    device = torch.device(args.device)
    d, ev = dcfg["data"], dcfg["eval"]
    dt, se = d["dt"], d["sample_every"]
    exp = ecfg["experiment"]
    chaotic = system.name == "double_pendulum"  # see the module docstring

    # ---- ground-truth long-horizon reference (regular systems only; first n_rollout test ICs per tier) ----
    truth_long, mu_all, slices = None, None, {}
    if not chaotic:
        refs, mus, off = [], [], 0
        for tier, view in views.items():
            traj = view["traj"][:, :ev["n_rollout"]]
            n_mu, k = traj.shape[:2]
            refs.append(traj[:, :, 0].reshape(n_mu * k, -1))
            mus.append(view["mu_grid"].repeat_interleave(k, 0))
            slices[tier] = (off, off + n_mu * k)
            off += n_mu * k
        y0 = torch.cat(refs)
        mu_all = torch.cat(mus)
        n_long = int(round(ev["rollout_t_long"] / dt))
        with torch.no_grad():
            truth_long = rollout_sampled(system_field(system, mu_all),
                                         y0, dt, n_long, se)  # (Tl,B,2n) f64
        # VPT threshold: ground-truth phase-space RMS amplitude x vpt_frac (same threshold within an experiment)
        theta = args.vpt_frac * truth_long.pow(2).sum(-1).mean().sqrt().item()
        print(f"[{exp}] ground-truth long reference t={ev['rollout_t_long']}, "
              f"VPT threshold θ={theta:.4f}", flush=True)

    for seed in args.seeds:
        out_dir = Path("results") / exp / str(seed)
        run_models = args.models or ecfg["models"]["list"]; rows = []
        if args.models and (out_dir / "metrics_long.csv").exists():   # incremental: keep rows of other models
            with open(out_dir / "metrics_long.csv", newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if r["model"] not in run_models]
        for name in run_models:
            ckpt = out_dir / f"{name}.pt"
            if not ckpt.exists():
                print(f"  skip {name} (no checkpoint)")
                continue
            model = load_checkpoint(name, system, ecfg["models"], ckpt, device)
            curves = {}
            # H regression error (only HNN-type models have Ĥ)
            if hasattr(model, "hamiltonian"):
                for tier, v in h_regression_error(model, system,
                                                  views).items():
                    rows.append({"model": name, "tier": tier,
                                 "metric": "h_rel_err", "value": v})
            # long-horizon pointwise L2 + VPT (regular systems only)
            if truth_long is not None:
                f = model_field(model, mu_all.to(torch.float32).to(device))
                with torch.no_grad():
                    pred = rollout_sampled(
                        f, truth_long[0].to(torch.float32).to(device),
                        dt, int(round(ev["rollout_t_long"] / dt)), se)
                l2 = ((pred.cpu().double() - truth_long) ** 2).sum(-1).sqrt()
                for tier, (a, b) in slices.items():
                    curve = l2[:, a:b].mean(1)
                    curves[f"{tier}/rollout_l2_long"] = curve
                    crossed = (l2[:, a:b] > theta)
                    # VPT per trajectory (full horizon if never crossed), then the mean
                    t_axis = torch.arange(l2.shape[0], dtype=torch.float64) \
                        * dt * se
                    vpt = torch.where(crossed.any(0),
                                      t_axis[crossed.double().argmax(0)],
                                      t_axis[-1])
                    rows.append({"model": name, "tier": tier,
                                 "metric": "vpt", "value": vpt.mean().item()})
                    rows.append({"model": name, "tier": tier,
                                 "metric": "rollout_l2_long_final",
                                 "value": curve[-1].item()})
            if curves:
                np.savez(out_dir / f"long_curves_{name}.npz",
                         **{k: v.numpy() for k, v in curves.items()})
            print(f"[{exp} seed={seed}] {name}: " + ", ".join(
                f"{r['metric']}[{r['tier']}]={r['value']:.3e}"
                for r in rows if r["model"] == name), flush=True)
        if rows:
            with open(out_dir / "metrics_long.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["model", "tier", "metric",
                                                   "value"])
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
