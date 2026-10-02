"""Beam backbone curves omega_1(A)/omega_1,lin: reference and models, at the two parameter values used by beam_free.

(a) interpolated point, (b) extrapolated point: x-axis total peak deflection A/h (0.5 to 3, 11 amplitudes), y-axis first frequency divided by the linear frequency at that parameter value;
(c,d) relative error |omega_1_hat - omega_1|/omega_1 versus amplitude at the same two points, log scale. Frequency estimate as in scripts/beam_eval_shells.py
(static equilibrium displaced along the first mode to A*h, released from rest; RK4 dt=1e-3, t in [0,6]; Hann-windowed FFT + log-parabolic interpolation).
Cache: results/beam_main/figs/beam_backbone_cache.npz; --replot only redraws.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from fig_verif_curves import COLOR, LEG  # noqa: E402
from fig_beam_free import POINTS, LAB, STY  # noqa: E402

AMPS = np.linspace(0.5, 3.0, 11); BB_DT, BB_T = 1e-3, 6.0
MODELS = ["pshnn", "silu_hnn", "ahnn", "symoden_mu"]


def compute(seed, sindy):
    from scripts.sindy_model import SindyPotentialModel
    from src.analysis.backbone import estimate_omega1, rest_start_ic
    from src.data.integrators import rollout_sampled, rollout_sampled_graphed, system_field
    from src.models.factory import build_model
    from src.systems import get_system
    torch.set_default_dtype(torch.float64)
    ecfg = yaml.safe_load(open(ROOT / "configs/beam_main.yaml")); mcfg = ecfg["models"]
    system = get_system("clamped_beam"); n = system.dim; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = int(round(BB_T / BB_DT)); out = {"amps": AMPS}
    for tag, mu in POINTS.items():
        mu_c = torch.tensor([mu], dtype=torch.float64); mu_bb = mu_c.repeat(len(AMPS), 1); amp = torch.tensor(AMPS, dtype=torch.float64)
        y0 = rest_start_ic(system, mu_bb, amp)
        with torch.no_grad(): y_true = rollout_sampled(system_field(system, mu_bb), y0, BB_DT, n_steps, 1)
        om_true = estimate_omega1(y_true[:, :, 0], BB_DT).numpy(); out[f"{tag}/truth"] = om_true
        # linear frequency: first frequency of the linearisation at the static equilibrium, approximated by the same estimator at a tiny amplitude (A = static deflection + 0.02h)
        a_s = system.static_equilibrium(mu_c); peak_s = float(system.peak_deflection(a_s, mu_c))
        y0l = rest_start_ic(system, mu_c, torch.tensor([peak_s + 0.02], dtype=torch.float64))
        with torch.no_grad(): yl = rollout_sampled(system_field(system, mu_c), y0l, BB_DT, n_steps, 1)
        out[f"{tag}/omega_lin"] = float(estimate_omega1(yl[:, :, 0], BB_DT)[0])
        mu_d, y0_d = mu_bb.to(dev), y0.to(dev).contiguous()
        for name in MODELS + ["sindy"]:
            if name == "sindy": m = SindyPotentialModel(str(ROOT / sindy), n)
            else:
                m, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"]); m.load_state_dict(torch.load(ROOT / f"results/beam_main/{seed}/{name}.pt", map_location="cpu"))
            m = m.double().to(dev).eval()
            with torch.no_grad(): y = rollout_sampled_graphed(lambda yy: m.time_derivative(yy, mu_d, create_graph=False), y0_d, BB_DT, n_steps, 1).cpu()
            out[f"{tag}/{name}"] = estimate_omega1(y[:, :, 0], BB_DT).numpy(); print(f"  {tag} {name} done", flush=True)
    return out


def plot(Z):
    """1x2: interpolated point only. (a) backbone curve omega_1(A)/omega_1,lin, legend lower right; (b) relative error versus amplitude, log scale."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, scienceplots  # noqa: F401
    os.environ["PATH"] = str(ROOT / ".texlive/bin/x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(ROOT / "scripts/figstyle/science.mplstyle")])
    HLEG = dict(LEG, fontsize=LEG["fontsize"] * 0.66, handlelength=1.1, borderpad=0.3, labelspacing=0.2, handletextpad=0.4)
    A = Z["amps"]; tag = "typical"; wl = Z[f"{tag}/omega_lin"]
    MK = {"pshnn": "o", "silu_hnn": "v", "ahnn": "^", "symoden_mu": "D", "sindy": "P"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.1)); fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.2, wspace=0.32)
    ax.plot(A, Z[f"{tag}/truth"] / wl, color="k", linewidth=0.8, label=LAB["truth"], zorder=5)
    for key in ["symoden_mu", "sindy", "ahnn", "silu_hnn", "pshnn"]:
        ax.plot(A, Z[f"{tag}/{key}"] / wl, linestyle="none", marker=MK[key], markersize=3, markerfacecolor="white", markeredgewidth=0.6, label=LAB[key], color=STY[key]["color"], zorder=STY[key]["zorder"])
        err = np.abs(Z[f"{tag}/{key}"] - Z[f"{tag}/truth"]) / Z[f"{tag}/truth"]
        ax2.plot(A, err, marker=MK[key], markersize=3, markerfacecolor="white", markeredgewidth=0.6, linewidth=0.7, color=STY[key]["color"], zorder=STY[key]["zorder"])
    ax.set_xlabel(r"$A/h$"); ax.set_ylabel(r"$\omega_{1}(A)/\omega_{1,\mathrm{lin}}$"); ax.set_title("(a)", loc="left", pad=3)
    ax.legend(loc="lower right", ncol=1, **HLEG).set_zorder(20)
    ax2.set_yscale("log"); ax2.set_xlabel(r"$A/h$"); ax2.set_ylabel(r"$|\hat{\omega}_{1}-\omega_{1}|/\omega_{1}$"); ax2.set_title("(b)", loc="left", pad=3)
    fig.savefig(ROOT / "paper/figures/beam_backbone.pdf"); fig.savefig(ROOT / "paper/figures/beam_backbone.png"); plt.close(fig)
    print(f"{tag}: omega_lin {wl:.3f}; errors at A/h=3: " + ", ".join(f"{k} {abs(Z[f'{tag}/{k}'][-1]-Z[f'{tag}/truth'][-1])/Z[f'{tag}/truth'][-1]:.1e}" for k in ["pshnn", "sindy", "silu_hnn", "ahnn"]))
    print("written paper/figures/beam_backbone.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=2); ap.add_argument("--sindy", default="results/beam_sindy/coef_potential_dq4_dmu3.pt"); ap.add_argument("--replot", action="store_true")
    a = ap.parse_args(); cache = ROOT / "results/beam_main/figs/beam_backbone_cache.npz"
    if a.replot and cache.exists(): Z = dict(np.load(cache))
    else: Z = compute(a.seed, a.sindy); np.savez(cache, **Z)
    plot(Z)
