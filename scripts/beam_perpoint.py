"""Beam free vibration: midpoint-deflection error at every evaluation parameter point (full statistics, no cherry-picking).

For each of the 24 interp and 24 extrap parameter points (three extrapolation ranges), take its 4 test trajectories, integrate the learned vector field
from the true initial state to the end of the data time span (same definitions as beam_midpoint_error.py: same RK4, dt, storage interval), compute the
midpoint-deflection RMS/h and VPT per trajectory, median over the 4 trajectories: one number per (model, seed, parameter point). Models: pshnn, silu_hnn, ahnn, symoden_mu, mlp_ode x 5 seeds, plus the single sparse-regression (4,3) fit.
Output: results/beam_main/perpoint_midpoint.csv (long table: model, seed, tier, shell, idx, mu1, mu2, mu3, rms_h, vpt).
Usage: python3 scripts/beam_perpoint.py [--seeds 0 1 2 3 4] [--models ...]
"""
import argparse, csv, sys
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.sindy_model import SindyPotentialModel  # noqa: E402
from src.data.dataset import eval_views, load_dataset  # noqa: E402
from src.data.integrators import rollout_sampled_graphed  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402
from src.systems.clamped_beam import SQRT12, mode_shapes  # noqa: E402


def cycles_of(w):
    s = torch.sign(w - w.mean()); return float((s[1:] * s[:-1] < 0).sum()) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=["pshnn", "silu_hnn", "ahnn", "symoden_mu", "mlp_ode"])
    ap.add_argument("--sindy", default="results/beam_sindy/coef_potential_dq4_dmu3.pt")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecfg = yaml.safe_load(open(ROOT / "configs/beam_main.yaml")); dcfg = yaml.safe_load(open(ROOT / ecfg["data_config"])); mcfg = ecfg["models"]
    system = get_system("clamped_beam"); n = system.dim
    dt, se, t_end = dcfg["data"]["dt"], dcfg["data"]["sample_every"], dcfg["data"]["t_span"]; n_steps = int(round(t_end / dt))
    data, man = load_dataset(ROOT / "data_cache/clamped_beam"); views = eval_views(data, man)
    phi = mode_shapes(system.beta, torch.tensor([0.5]), 0)[:n, 0]
    w = lambda x: (x[..., :n].double() @ phi) / SQRT12
    tiers = {}
    for tier in ["interp", "extrap"]:
        v = views[tier]; tr = v["traj"][:, :4]; n_mu, k, T1, D = tr.shape
        ref = tr.reshape(n_mu * k, T1, D).double(); mu_all = v["mu_grid"].double().repeat_interleave(k, 0)
        w_true = w(ref.permute(1, 0, 2)); w_dyn = w_true - w_true.mean(0, keepdim=True); amp = w_dyn.abs().max(0).values
        cyc = torch.tensor([cycles_of(w_true[:, b]) for b in range(w_true.shape[1])]); per = t_end / cyc
        shells = man.get("extrap_shells") if tier == "extrap" else [0] * n_mu
        tiers[tier] = dict(n_mu=n_mu, k=k, mus=v["mu_grid"], shells=shells, w_true=w_true, amp=amp, per=per,
                           mu_dev=mu_all.to(dev), y0_dev=ref[:, 0].to(dev).contiguous())

    def metrics(model, T):
        model = model.double().to(dev).eval()
        with torch.no_grad():
            pred = rollout_sampled_graphed(lambda y: model.time_derivative(y, T["mu_dev"], create_graph=False), T["y0_dev"], dt, n_steps, se).cpu()
        err = w(pred) - T["w_true"]; rms = err.pow(2).mean(0).sqrt(); over = err.abs() > 0.1 * T["amp"]
        t_idx = torch.where(over.any(0), over.double().argmax(0), torch.full((err.shape[1],), err.shape[0] - 1))
        vpt = t_idx.double() * (t_end / (err.shape[0] - 1)) / T["per"]
        return rms.reshape(T["n_mu"], T["k"]).median(1).values, vpt.reshape(T["n_mu"], T["k"]).median(1).values

    jobs = [(s, m) for s in a.seeds for m in a.models] + [(0, "sindy")]
    rows = []
    for seed, name in jobs:
        if name == "sindy": model = SindyPotentialModel(str(ROOT / a.sindy), n)
        else:
            model, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"])
            model.load_state_dict(torch.load(ROOT / f"results/beam_main/{seed}/{name}.pt", map_location="cpu"))
        for tier, T in tiers.items():
            rms, vpt = metrics(model, T)
            for i in range(T["n_mu"]):
                rows.append(dict(model=name, seed=seed, tier=tier, shell=int(T["shells"][i]), idx=i, mu1=float(T["mus"][i, 0]), mu2=float(T["mus"][i, 1]), mu3=float(T["mus"][i, 2]),
                                 rms_h=float(rms[i]), vpt=float(vpt[i])))
        print(f"seed {seed} {name:11s} done", flush=True)
    out = ROOT / "results/beam_main/perpoint_midpoint.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("written", out, len(rows), "rows")


if __name__ == "__main__":
    main()
