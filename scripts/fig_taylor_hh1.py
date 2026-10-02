"""One-parameter Henon-Heiles: Taylor-coefficient check (following Han et al. 2021 Fig. 3b).

Top row: coefficients of a least-squares fit of the identified Hamiltonian near the origin to the 35 monomials of total degree <= 3 in (q1, q2, p1, p2), versus lambda,
      (a) Schmidt-HNN, (b) AHNN; four quadratic terms (true value 1/2), two cubic terms (true values lambda and -lambda/3), 28 spurious terms (true value 0, constant term excluded).
Bottom row: (c) maximum coefficient deviation over the six physical terms, (d) maximum magnitude over the 28 spurious terms, versus lambda, both models in one panel.
Data: results/ahnn_repro/hh1p/<seed>/potential_<model>.npz (same source as taylor_per_seed.py), median over five seeds.
Output: paper/figures/taylor_E3.pdf/png
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fig_verif_curves import COLOR, LABEL, LEG  # noqa: E402  colours, names and legend conventions shared with the verification figures

SEEDS = [0, 1, 2, 3, 4]
BOX = (0.2, 0.8)
C_QUAD, C_CUB, C_SPUR = "#0072B2", "#D55E00", "#9A9A9A"


def load(model):
    z = [np.load(ROOT / "results/ahnn_repro/hh1p" / str(s) / f"potential_{model}.npz") for s in SEEDS]
    lam, exps = z[0]["lam"], [tuple(int(x) for x in e) for e in z[0]["taylor_exps"]]
    T = np.median(np.stack([d["taylor"] for d in z]), 0)          # (96, 35) median over five seeds
    return lam, exps, T


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    os.environ["PATH"] = str(ROOT / ".texlive" / "bin" / "x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(ROOT / "scripts" / "figstyle" / "science.mplstyle")])

    models = [("pshnn", "(a)"), ("ahnn", "(b)")]
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.11, wspace=0.3, hspace=0.42)
    dev = {}
    for k, (m, lab) in enumerate(models):
        lam, exps, T = load(m)
        quad = [exps.index(e) for e in [(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)]]
        cub = [exps.index((2, 1, 0, 0)), exps.index((0, 3, 0, 0))]
        const = exps.index((0, 0, 0, 0))
        spur = [i for i in range(len(exps)) if i not in quad + cub + [const]]
        ax = axes[0, k]
        ax.axvspan(*BOX, color="0.92", zorder=0.55, lw=0)
        ax.plot(lam, 0.5 * np.ones_like(lam), "k--", lw=0.5, zorder=1); ax.plot(lam, lam, "k--", lw=0.5, zorder=1); ax.plot(lam, -lam / 3, "k--", lw=0.5, zorder=1)
        for i in spur: ax.plot(lam, T[:, i], color=C_SPUR, lw=0.4, zorder=2)
        for j, i in enumerate(quad): ax.plot(lam, T[:, i], color=C_QUAD, lw=0.7, zorder=3, label="quadratic terms" if j == 0 else None)
        for j, i in enumerate(cub): ax.plot(lam, T[:, i], color=C_CUB, lw=0.7, zorder=3, label="cubic terms" if j == 0 else None)
        ax.plot([], [], color=C_SPUR, lw=0.4, label="other terms"); ax.plot([], [], "k--", lw=0.5, label="exact")
        ax.set_xlim(lam[0], lam[-1]); ax.set_ylim(-0.45, 1.1); ax.set_ylabel("Coefficient")
        ax.set_title(f"{lab} {LABEL[m].split(' (')[0]}", loc="left", pad=3)
        if k == 0: ax.legend(loc="upper left", ncol=1, **LEG)
        true = np.zeros_like(T); true[:, quad] = 0.5; true[:, cub[0]] = lam; true[:, cub[1]] = -lam / 3
        dev[m] = (np.abs(T[:, quad + cub] - true[:, quad + cub]).max(1), np.abs(T[:, spur]).max(1))
    for k, (ttl, lab) in enumerate((("Physical terms, max. deviation", "(c)"), ("Other terms, max. magnitude", "(d)"))):
        ax = axes[1, k]; ax.axvspan(*BOX, color="0.92", zorder=0.55, lw=0)
        for m, ls in (("pshnn", "-"), ("ahnn", "--")):
            ax.plot(lam, dev[m][k], ls, color=COLOR[m], lw=0.8, label=LABEL[m], zorder=3)
        ax.set_yscale("log"); ax.set_xlim(lam[0], lam[-1]); ax.set_ylim(1e-5, 1e-1)
        ax.set_xlabel(r"$\lambda$"); ax.set_title(f"{lab} {ttl}", loc="left", pad=3)
        if k == 0: ax.set_ylabel("Coefficient error"); ax.legend(loc="upper left", **LEG)
    for ax in axes[0]: ax.set_xlabel(r"$\lambda$")
    out = ROOT / "paper" / "figures"
    fig.savefig(out / "taylor_E3.pdf"); fig.savefig(out / "taylor_E3.png"); plt.close(fig)
    print("written", out / "taylor_E3.pdf")


if __name__ == "__main__":
    main()
