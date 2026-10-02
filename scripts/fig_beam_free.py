"""Midpoint deflection time histories of the free vibration of the clamped beam (beam_free.pdf).

Rows = parameter values (default: typical interpolated point, point #7 of the 0-10% extrapolation range); columns = full first 20 periods, zoom on the last 3 periods.
Curves: reference solution, Schmidt-HNN, SiLU potential network, AHNN, sparse regression (d_q=4, d_mu=3); seed 2 (same as the load cases).
Trajectories cached in results/beam_main/figs/beam_free_cache.npz; --replot skips recomputation.
Style follows scripts/fig_verif_curves.py (SciencePlots + newtx, 5.4 in text width).
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from fig_verif_curves import COLOR, LEG  # noqa: E402

POINTS = {"typical": [-0.761, -0.025, 0.276], "extrap": [0.357, 0.253, 0.576]}   # both points are the median point of the sparse-regression / Schmidt-HNN error ratio within their tier (median over five seeds, scripts/fig_beam_perpoint.py)
MODELS = ["pshnn", "silu_hnn", "ahnn", "symoden_mu", "mlp_ode"]   # same as the verification time-history figures: all three baselines drawn
LAB = {"truth": "reference", "pshnn": "Schmidt-HNN (present)", "silu_hnn": "AHNN-SiLU", "ahnn": "AHNN (Han et al. 2021)",
       "symoden_mu": "SymODEN (Zhong et al. 2020)", "mlp_ode": "MLP (Greydanus et al. 2019)", "sindy": "SINDy (4, 3)"}
STY = {"pshnn": dict(color=COLOR["pshnn"], linestyle=(0, (5.5, 2.8)), linewidth=0.8, zorder=10),
       "silu_hnn": dict(color="#E69F00", linestyle=(0, (4, 1.5)), linewidth=0.7, alpha=0.92, zorder=8),
       "ahnn": dict(color=COLOR["ahnn"], linestyle=(0, (2.5, 1.2)), linewidth=0.7, alpha=0.92, zorder=7),
       "symoden_mu": dict(color=COLOR["symoden_mu"], linestyle=(0, (2.5, 1.2)), linewidth=0.7, alpha=0.92, zorder=6),
       "mlp_ode": dict(color=COLOR["mlp_ode"], linestyle=(0, (1, 1.2)), linewidth=0.7, alpha=0.92, zorder=5),
       "sindy": dict(color="#8E5BA6", linestyle=(0, (3, 1, 1, 1)), linewidth=0.7, alpha=0.92, zorder=9)}


def compute(exp, seed, sindy, n_cyc, ic_seed, t_max):
    from scripts.sindy_model import SindyPotentialModel
    from src.data.integrators import rollout_sampled, rollout_sampled_graphed, system_field
    from src.models.factory import build_model
    from src.systems import get_system
    from src.systems.clamped_beam import SQRT12, mode_shapes
    torch.set_default_dtype(torch.float64)
    ecfg = yaml.safe_load(open(ROOT / f"configs/{exp}.yaml")); dcfg = yaml.safe_load(open(ROOT / ecfg["data_config"]))
    system = get_system(dcfg["system"]); n = system.dim; dt = dcfg["data"]["dt"]; se = 100
    phi_mid = mode_shapes(system.beta, torch.tensor([0.5]), 0)[:n, 0]
    w = lambda tr: (tr[:, 0, :n] @ phi_mid) / SQRT12
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); mcfg = ecfg["models"]
    out = {}
    for tag, mu in POINTS.items():
        mu_c = torch.tensor([mu], dtype=torch.float64); g = torch.Generator().manual_seed(ic_seed)
        y0 = system.sample_ic(mu_c[0], 4, g)[:1]
        with torch.no_grad(): tr_true = rollout_sampled(system_field(system, mu_c), y0, dt, int(round(t_max / dt)), se)
        wt = w(tr_true); t = torch.arange(wt.shape[0]) * dt * se
        s = torch.sign(wt - wt.mean()); zc = torch.nonzero(s[1:] * s[:-1] < 0).squeeze(-1); k = int(2 * n_cyc)
        t_end = float(t[zc[k - 1]]); n_steps = int(round(t_end / dt)) // se * se; T = n_steps // se + 1
        out[f"{tag}/t"] = (t[:T] / (t_end / n_cyc)).numpy(); out[f"{tag}/truth"] = wt[:T].numpy(); out[f"{tag}/peak"] = float(system.peak_deflection(y0[:, :n], mu_c))
        out[f"{tag}/truth_x"] = tr_true[:T, 0].numpy()   # full state (T, 2n), for the configuration plane
        mu_d, y0_d = mu_c.to(dev), y0.to(dev).contiguous()
        for name in MODELS:
            m, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"])
            m.load_state_dict(torch.load(ROOT / "results" / exp / str(seed) / f"{name}.pt", map_location="cpu")); m = m.double().to(dev).eval()
            with torch.no_grad(): tr = rollout_sampled_graphed(lambda y: m.time_derivative(y, mu_d, create_graph=False), y0_d, dt, n_steps, se).cpu()
            out[f"{tag}/{name}"] = w(tr).numpy(); out[f"{tag}/{name}_x"] = tr[:, 0].numpy(); print(f"  {tag} {name} done", flush=True)
        m = SindyPotentialModel(str(sindy), n).to(dev)
        with torch.no_grad(): tr = rollout_sampled_graphed(lambda y: m.time_derivative(y, mu_d, create_graph=False), y0_d, dt, n_steps, se).cpu()
        out[f"{tag}/sindy"] = w(tr).numpy(); out[f"{tag}/sindy_x"] = tr[:, 0].numpy(); print(f"  {tag} sindy done (peak {out[f'{tag}/peak']:.2f}h, {n_cyc:g} periods = t {t_end:.3f})", flush=True)
    return out


def plot(Z, zoom_cyc, n_cyc, plane=(0, 1), plane_cyc=1.0):
    """One parameter value per row: left = midpoint deflection over the last zoom_cyc periods, right = configuration plane (q_a, q_b) over the same window, square."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, scienceplots  # noqa: F401
    os.environ["PATH"] = str(ROOT / ".texlive/bin/x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(ROOT / "scripts/figstyle/science.mplstyle")])
    HLEG = dict(LEG, fontsize=LEG["fontsize"] * 0.66, handlelength=1.1, borderpad=0.3, labelspacing=0.2, handletextpad=0.4)
    tags = list(POINTS); nr = len(tags); a, b = plane
    RH = 1.9   # row height (in), consistent with the 2x2 verif/beam_loads figures (3.5-3.8 in)
    fig, axes = plt.subplots(nr, 2, figsize=(5.4, RH * nr), gridspec_kw={"width_ratios": [3, 2]}, squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=1 - 0.18 / (RH * nr), bottom=0.36 / (RH * nr), wspace=0.32, hspace=0.36)
    order = ["mlp_ode", "symoden_mu", "ahnn", "silu_hnn", "sindy", "pshnn"]
    for r, tag in enumerate(tags):
        t = Z[f"{tag}/t"]; tr = Z[f"{tag}/truth"]; lo, hi = n_cyc - zoom_cyc, n_cyc; sl = (t >= lo) & (t <= hi)
        sp = (t >= n_cyc - plane_cyc) & (t <= n_cyc)   # right panel shows only the last plane_cyc periods
        axL, axR = axes[r]
        axL.plot(t[sl], tr[sl], color="k", linewidth=0.8, label=LAB["truth"], zorder=5)
        X = Z[f"{tag}/truth_x"]; axR.plot(X[sp, a], X[sp, b], color="k", linewidth=0.8, zorder=5)
        for key in order:
            if f"{tag}/{key}" not in Z: continue
            axL.plot(t[sl], Z[f"{tag}/{key}"][sl], label=LAB[key], **STY[key])
            Xk = Z[f"{tag}/{key}_x"]; axR.plot(Xk[sp, a], Xk[sp, b], **STY[key])
        axL.set_xlim(lo, hi); axL.set_ylim(tr.min() - 0.3, tr.max() + 0.3); axL.set_ylabel(r"$w(L/2)/h$")
        n = X.shape[1] // 2; lab = lambda i: rf"$q_{{{i+1}}}$" if i < n else rf"$p_{{{i-n+1}}}$"
        axR.set_xlabel(lab(a)); axR.set_ylabel(lab(b)); axR.set_box_aspect(1)
        ra, rb = np.abs(X[sp, a]).max(), np.abs(X[sp, b]).max(); axR.set_xlim(-1.3 * ra, 1.3 * ra); axR.set_ylim(-1.3 * rb, 1.3 * rb)   # axis limits set by the reference solution
        if r == nr - 1: axL.set_xlabel(r"$t/T_{1}$")
        axL.set_title(f"({chr(97 + 2 * r)})", loc="left", pad=3); axR.set_title(f"({chr(98 + 2 * r)})", loc="left", pad=3)
    axes[0][0].legend(loc="lower right", ncol=1, **HLEG).set_zorder(20)
    fig.savefig(ROOT / "paper/figures/beam_free.pdf"); fig.savefig(ROOT / "paper/figures/beam_free.png"); plt.close(fig)
    print("written paper/figures/beam_free.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="beam_main"); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--sindy", default="results/beam_sindy/coef_potential_dq4_dmu3.pt")
    ap.add_argument("--n-cyc", type=float, default=20.0); ap.add_argument("--zoom-cyc", type=float, default=3.0)
    ap.add_argument("--ic-seed", type=int, default=2026); ap.add_argument("--t-max", type=float, default=8.0)
    ap.add_argument("--replot", action="store_true"); ap.add_argument("--plane", type=int, nargs=2, default=[0, 1], help="two coordinates of the right-panel configuration plane (0-based)"); ap.add_argument("--plane-cyc", type=float, default=1.0, help="number of trailing periods drawn in the right panel"); a = ap.parse_args()
    cache = ROOT / "results" / a.exp / "figs" / "beam_free_cache.npz"
    if a.replot and cache.exists(): Z = dict(np.load(cache))
    else:
        Z = compute(a.exp, a.seed, a.sindy, a.n_cyc, a.ic_seed, a.t_max); cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, **Z)
    plot(Z, a.zoom_cyc, a.n_cyc, tuple(a.plane), a.plane_cyc)
