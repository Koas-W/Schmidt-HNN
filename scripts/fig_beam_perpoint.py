"""Figure + statistics: midpoint deflection error of the beam free vibration at every evaluation point (guards against cherry-picking).

Input: results/beam_main/perpoint_midpoint.csv (scripts/beam_perpoint.py).
Figure beam_perpoint.pdf: (a) 24 interpolated points, (b) 24 extrapolated points (shade or marker by extrapolation range); x-axis = points sorted by increasing Schmidt-HNN error,
y-axis = RMS/h of each model (median over five seeds), log scale; the points used in the time-history and load-case figures are marked with large hollow circles.
Statistics (stdout + paper/tables/tab_beam_ratio.tex): per-point error ratio of each model to Schmidt-HNN, median [min, max] over each tier,
and the rank of the two used points within their tier by that ratio.
"""
import csv, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from fig_verif_curves import COLOR, LEG  # noqa: E402

MODELS = ["pshnn", "silu_hnn", "ahnn", "symoden_mu", "mlp_ode", "sindy"]
LAB = {"pshnn": "Schmidt-HNN (present)", "silu_hnn": "AHNN-SiLU", "ahnn": "AHNN (Han et al. 2021)", "symoden_mu": "SymODEN (Zhong et al. 2020)",
       "mlp_ode": "MLP (Greydanus et al. 2019)", "sindy": "SINDy (4, 3)"}
TLAB = {"pshnn": "Schmidt-HNN", "silu_hnn": "AHNN-SiLU", "ahnn": "AHNN", "symoden_mu": "SymODEN", "mlp_ode": "MLP", "sindy": "SINDy (4, 3)"}
STY = {"pshnn": dict(color=COLOR["pshnn"], marker="o", zorder=10), "silu_hnn": dict(color="#E69F00", marker="v", zorder=8),
       "ahnn": dict(color=COLOR["ahnn"], marker="^", zorder=7), "symoden_mu": dict(color=COLOR["symoden_mu"], marker="D", zorder=6),
       "mlp_ode": dict(color=COLOR["mlp_ode"], marker="s", zorder=5), "sindy": dict(color="#8E5BA6", marker="P", zorder=9)}
USED = {"interp": [-0.761, -0.025, 0.276], "extrap": [0.357, 0.253, 0.576]}   # points used in the time-history and load-case figures


def load():
    v = defaultdict(list); meta = {}
    for r in csv.DictReader(open(ROOT / "results/beam_main/perpoint_midpoint.csv")):
        key = (r["model"], r["tier"], int(r["idx"])); v[key].append(float(r["rms_h"]))
        meta[(r["tier"], int(r["idx"]))] = (int(r["shell"]), float(r["mu1"]), float(r["mu2"]), float(r["mu3"]))
    med = {k: float(np.median(x)) for k, x in v.items()}
    return med, meta


def main():
    med, meta = load()
    tiers = {"interp": sorted(i for (t, i) in meta if t == "interp"), "extrap": sorted(i for (t, i) in meta if t == "extrap")}
    # ---- statistics: per-point ratios ----
    # saturation fraction: results/beam_main/saturation.csv (scripts/beam_saturation.py), seed median of the last hidden layer
    sat = {}
    for r in csv.DictReader(open(ROOT / "results/beam_main/saturation.csv")):
        if r["kind"] == "saturation": sat[r["name"]] = float(r["frac_median"])   # overwritten layer by layer, the last layer remains
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{Ratio of the midpoint error of each model to that of the",
         r"    proposed architecture, median [min, max] over the interpolated and the extrapolated parameter values, and",
         r"    fraction of the units of the last hidden layer with $|z|>2$ on the test states.}",
         r"  \label{tab:beam-ratio}", r"  \footnotesize\setlength{\tabcolsep}{5pt}",
         r"  \begin{tabular}{lccc}", r"    \toprule", r"    Model & interpolated & extrapolated & $|z|>2$ (\%) \\", r"    \midrule"]
    L.append(f"    {TLAB.get('pshnn', 'Schmidt-HNN')} & 1 & 1 & -- \\\\")   # proposed model is the unit of the ratio
    used_rank = {}
    for m in MODELS[1:]:
        cells = []
        for t, idxs in tiers.items():
            ratio = np.array([med[(m, t, i)] / med[("pshnn", t, i)] for i in idxs])
            cells.append(f"{np.median(ratio):.1f} [{ratio.min():.1f}, {ratio.max():.1f}]")
            if m == "sindy":
                mu_used = USED[t]; iu = min(idxs, key=lambda i: sum((a - b) ** 2 for a, b in zip(meta[(t, i)][1:], mu_used)))
                order = sorted(idxs, key=lambda i: -med[("sindy", t, i)] / med[("pshnn", t, i)])
                used_rank[t] = (order.index(iu) + 1, len(idxs), float(med[("sindy", t, iu)] / med[("pshnn", t, iu)]))
            print(f"{m:11s} {t:7s} ratio median {np.median(ratio):7.1f}  min {ratio.min():7.1f}  max {ratio.max():7.1f}")
        cells.append(f"{100 * sat[m]:.0f}" if m in sat else "--")
        L.append(f"    {TLAB[m]} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (ROOT / "paper/tables/tab_beam_ratio.tex").write_text("\n".join(L) + "\n")
    for t, (rk, n, r) in used_rank.items(): print(f"used {t} point: sparse/Schmidt ratio {r:.1f}, rank {rk} of {n} by that ratio")
    # ---- figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, scienceplots  # noqa: F401
    os.environ["PATH"] = str(ROOT / ".texlive/bin/x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(ROOT / "scripts/figstyle/science.mplstyle")])
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.2, wspace=0.08)
    for ax, (t, idxs) in zip(axes, tiers.items()):
        order = sorted(idxs, key=lambda i: (meta[(t, i)][0], med[("pshnn", t, i)])); x = np.arange(1, len(order) + 1)   # extrapolated tier sorted by extrapolation range first, then by error
        for m in MODELS:
            y = [med[(m, t, i)] for i in order]
            ax.plot(x, y, linestyle="none", markersize=3.2, markeredgewidth=0.5, markerfacecolor="white" if m != "pshnn" else STY[m]["color"], label=LAB[m], **STY[m])
        if t == "extrap":   # separators between extrapolation ranges, annotated with the out-of-range depth
            sh = [meta[(t, i)][0] for i in order]
            for k, lab in ((1, r"0--10\%"), (2, r"10--20\%"), (3, r"20--30\%")):
                xs = [x[j] for j in range(len(order)) if sh[j] == k]
                if not xs: continue
                if k > 1: ax.axvline(min(xs) - 0.5, color="0.6", lw=0.5, zorder=0.6)
                ax.text(0.5 * (min(xs) + max(xs)), 2.6, lab, ha="center", va="bottom", fontsize=LEG["fontsize"])
        mu_used = USED[t]; iu = min(idxs, key=lambda i: sum((a - b) ** 2 for a, b in zip(meta[(t, i)][1:], mu_used)))
        xu = order.index(iu) + 1; ax.plot([xu], [med[("pshnn", t, iu)]], marker="o", markersize=8, markerfacecolor="none", markeredgecolor="k", markeredgewidth=0.7, linestyle="none", zorder=12)
        ax.set_yscale("log"); ax.set_xlim(0.5, len(order) + 0.5); ax.set_ylim(2e-3, 5); ax.set_xlabel("parameter value (sorted)")
        ax.set_title("(a) interpolated" if t == "interp" else "(b) extrapolated", loc="left", pad=3)
    axes[0].set_ylabel(r"Midpoint error RMS$/h$")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=LEG["fontsize"] * 0.75, handletextpad=0.3, columnspacing=1.0)
    fig.savefig(ROOT / "paper/figures/beam_perpoint.pdf"); fig.savefig(ROOT / "paper/figures/beam_perpoint.png"); plt.close(fig)
    print("written paper/figures/beam_perpoint.pdf and tables/tab_beam_ratio.tex")


if __name__ == "__main__":
    main()
