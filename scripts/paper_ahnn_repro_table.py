"""Short main-text table: AHNN reproduction level on Henon-Heiles (values reported by Han et al. 2021 / this reproduction at a fixed budget / this reproduction trained to convergence).

Quantity: relative potential error <Delta V> (Han et al. 2021 Eq. 4), maximum over the region and over five seeds, percent, single number.
Fixed budget = runs before the unified protocol (10^4 steps, no early stopping, results/ahnn_repro_fixed_budget);
converged = unified protocol (results/ahnn_repro). Reported values taken from the original text (one parameter: Fig. 3a caption and Sec. III.B; two parameters: Fig. 8 caption),
two-parameter full plane uses the upper limit of the colour bar of Fig. 8.
Output: paper/tables/tab_ahnn_repro.tex
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXED, CONV = ROOT / "results/ahnn_repro_fixed_budget", ROOT / "results/ahnn_repro"
ROWS = [("hh1p", "train_region", r"One parameter, $\lambda\in[0.2,0.8]$", r"$<2$"),
        ("hh1p", "extrap", r"One parameter, $\lambda\in[0.05,1]$ outside", r"$<8$"),
        ("hh2p", "central_region", r"Two parameters, $[0.2,0.8]^{2}$", r"$<5$"),
        ("hh2p", "full_plane", r"Two parameters, $[0.05,1]^{2}$", r"$\le 20$")]


def cell(root, sysid, model, tier):
    """The reported value is an upper bound over the region, so take the maximum <Delta V> over the region, then the maximum over five seeds (single number)."""
    v = []
    for s in range(5):
        z = np.load(root / sysid / str(s) / f"potential_{model}.npz"); lam = z["lam"]; inb = (lam >= 0.2) & (lam <= 0.8)
        if sysid == "hh1p": d = z["deltaV"]; v.append(d[inb].max() if tier == "train_region" else d[~inb].max())
        else: g = z["deltaV_grid"]; v.append(g[np.ix_(inb, inb)].max() if tier == "central_region" else g.max())
    return f"{100 * max(v):.1f}"


def main():
    lines = [r"\begin{table}[htbp]", r"  \centering",
             r"  \caption{Relative potential error $\langle\Delta V\rangle$ (per cent) of the AHNN, maximum over each",
             r"    parameter region and over five seeds, as reported in~\cite{han2021ahnn} and as reproduced at a fixed",
             r"    budget of $10^{4}$ steps and after training to convergence.}",
             r"  \label{tab:ahnn-repro}",
             r"  \footnotesize\setlength{\tabcolsep}{5pt}",
             r"  \begin{tabular}{lccc}", r"    \toprule",
             r"    Region & Reported & Fixed budget & Converged \\", r"    \midrule"]
    for sysid, tier, name, rep in ROWS:
        lines.append(f"    {name} & {rep} & {cell(FIXED, sysid, 'ahnn', tier)} & {cell(CONV, sysid, 'ahnn', tier)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = ROOT / "paper" / "tables" / "tab_ahnn_repro.tex"; out.write_text("\n".join(lines) + "\n"); print("written", out)


if __name__ == "__main__":
    main()
