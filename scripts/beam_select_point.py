#!/usr/bin/env python3
"""Load-case point selection: for the 24 interp μ points, compute the free-vibration midpoint-deflection error of Schmidt-HNN (given seed) and of
sparse regression (same definitions: median rms_h and median VPT over 4 test trajectories), rank by the sparse-regression/Schmidt-HNN rms_h ratio and list candidate (μ₁, μ₂) loading lines.
Output: results/beam_main/load_cases/point_selection_s<seed>.csv
"""
import argparse, csv, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import eval_views, load_dataset
from src.data.integrators import rollout_sampled_graphed
from src.models.factory import build_model
from src.systems import get_system
from src.systems.clamped_beam import SQRT12, mode_shapes
from scripts.sindy_model import SindyPotentialModel
from scripts.beam_midpoint_error import cycles_of

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--sindy", default="results/beam_sindy/coef_potential_dq4_dmu3.pt")
ap.add_argument("--tier", default="interp", choices=["interp", "extrap"]); ap.add_argument("--shell", type=int, default=None, help="for extrap, restrict to this extrapolation range (1/2/3)")
args = ap.parse_args()
torch.set_default_dtype(torch.float64); dev = torch.device("cuda")
ecfg = yaml.safe_load(open("configs/beam_main.yaml")); dcfg = yaml.safe_load(open(ecfg["data_config"]))
system = get_system("clamped_beam"); n = system.dim
dt, se, t_end = dcfg["data"]["dt"], dcfg["data"]["sample_every"], dcfg["data"]["t_span"]; n_steps = int(round(t_end / dt))
data, man = load_dataset(Path("data_cache/clamped_beam")); v = eval_views(data, man)[args.tier]
shells = man.get("extrap_shells") if args.tier == "extrap" else None
tr = v["traj"][:, :4]; n_mu, k, T1, D = tr.shape
ref = tr.reshape(n_mu * k, T1, D).double(); mu_all = v["mu_grid"].double().repeat_interleave(k, 0)
phi = mode_shapes(system.beta, torch.tensor([0.5]), 0)[:n, 0]
w = lambda x: (x[..., :n].double() @ phi) / SQRT12
w_true = w(ref.permute(1, 0, 2)); w_dyn = w_true - w_true.mean(0, keepdim=True); amp = w_dyn.abs().max(0).values
cyc = torch.tensor([cycles_of(w_true[:, b]) for b in range(w_true.shape[1])]); per = t_end / cyc
mu_dev = mu_all.to(dev); y0_dev = ref[:, 0].to(dev).contiguous()

def metrics(model):
    model = model.double().to(dev).eval()
    with torch.no_grad():
        pred = rollout_sampled_graphed(lambda y: model.time_derivative(y, mu_dev, create_graph=False), y0_dev, dt, n_steps, se).cpu()
    err = w(pred) - w_true; rms = err.pow(2).mean(0).sqrt(); over = err.abs() > 0.1 * amp
    t_idx = torch.where(over.any(0), over.double().argmax(0), torch.full((err.shape[1],), err.shape[0] - 1))
    vpt = t_idx.double() * (t_end / (err.shape[0] - 1)) / per
    return rms.reshape(n_mu, k).median(1).values, vpt.reshape(n_mu, k).median(1).values

mcfg = ecfg["models"]
m, _ = build_model("pshnn", n, system.mu_dim, mcfg, mcfg["target_params"]); m.load_state_dict(torch.load(f"results/beam_main/{args.seed}/pshnn.pt", map_location="cpu"))
rp, vp = metrics(m); rs, vs = metrics(SindyPotentialModel(args.sindy, n))
mus = v["mu_grid"]; rows = []
for i in range(n_mu):
    if shells is not None and args.shell is not None and shells[i] != args.shell:
        continue
    rows.append({"idx": i, "shell": (shells[i] if shells else 0), "mu1": float(mus[i, 0]), "mu2": float(mus[i, 1]), "mu3": float(mus[i, 2]), "pshnn_rms_h": float(rp[i]), "pshnn_vpt": float(vp[i]),
                 "sindy_rms_h": float(rs[i]), "sindy_vpt": float(vs[i]), "ratio_sindy_over_pshnn": float(rs[i] / rp[i])})
rows.sort(key=lambda r: -r["ratio_sindy_over_pshnn"])
out = Path("results/beam_main/load_cases") / (f"point_selection_s{args.seed}.csv" if args.tier == "interp" else f"point_selection_{args.tier}{args.shell or ''}_s{args.seed}.csv")
with open(out, "w", newline="") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(rows[0])); wcsv.writeheader(); wcsv.writerows(rows)
print(f"{'rank':>4} {'idx':>3} {'mu1':>7} {'mu2':>7} {'mu3':>6} | {'pshnn rms':>9} {'VPT':>5} | {'sindy rms':>9} {'VPT':>5} | ratio")
for r_, r in enumerate(rows):
    print(f"{r_+1:4d} {r['idx']:3d} s{r['shell']} {r['mu1']:7.3f} {r['mu2']:7.3f} {r['mu3']:6.3f} | {r['pshnn_rms_h']:9.4f} {r['pshnn_vpt']:5.1f} | {r['sindy_rms_h']:9.4f} {r['sindy_vpt']:5.1f} | {r['ratio_sindy_over_pshnn']:5.1f}")
print("written", out)
