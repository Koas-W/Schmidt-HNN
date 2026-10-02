#!/usr/bin/env python3
"""Verification tables (LaTeX): rows = blocks of seven models x three quantities, columns = parameter tiers; each cell = median over five seeds [min, max], common exponent.

Three quantities: vector-field residual (test stored states, same definition as the loss), trajectory error after 3 linear periods (t=20),
relative energy drift after 20 linear periods (t=125.6) (true Hamiltonian). E1 has only the train tier; its drift is read from the long-horizon curve at t=125.6;
the three extrapolation ranges of E6 are merged into one tier (equal-weight average over the 8 points of each range).
Usage: python3 scripts/paper_latex_tables.py [E1 E2 E2A E3 E4 E6]   -> paper/tables/tab_verif_<id>.tex (full appendix table: landscape layout, three quantities x {in range, extrap.} = six columns, two rows per model: median / [min, max])\n      and tab_verif_<id>_main.tex (compact main-text table: train and interp merged vs extrap, medians only, best per column in bold)
"""
import csv, glob, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
MODELS = ["pshnn", "pshnn_concat", "pshnn_mlp", "pshnn_silu", "sep_hnn", "ahnn", "symoden_mu", "mlp_ode"]
LABEL = {"pshnn": r"Schmidt-HNN", "pshnn_concat": r"\;\;w/o separation", "pshnn_mlp": r"\;\;w/o gating (tanh)", "pshnn_silu": r"\;\;w/o gating (SiLU)", "pshnn_concat_minv": r"\;\;w/o separation, $\Mm^{-1}$",
         "sep_hnn": r"\;\;w/o Cholesky", "ahnn": "AHNN", "symoden_mu": "SymODEN", "mlp_ode": "MLP"}
BLOCKS = [("vf", r"Vector-field residual"), ("l2", r"Trajectory error, 3 periods"), ("drift", r"Energy drift, 20 periods")]
SHORT = {"vf": r"Residual", "l2": r"Traj.\ error (3\,$T$)", "drift": r"Energy drift (20\,$T$)"}
TIER_LABEL = {"train": "trained", "interp": "interpolated", "extrap": "extrapolated"}
SPECS = {
    "E1": dict(root="results/e1_pendulum", tiers=["train"], caption="pendulum", label="tab:verif-pendulum"),
    "E2": dict(root="results/e2_double_pendulum", tiers=["train", "interp", "extrap"], caption="double pendulum", label="tab:verif-dp"),
    "E2A": dict(root="results/e2_double_pendulum_minv", tiers=["train", "interp", "extrap"], caption=r"double pendulum with $\Mm^{-1}$ represented", label="tab:verif-dp-minv", models=["pshnn", "pshnn_concat", "pshnn_mlp"]),
    "E3": dict(root="results/ahnn_repro/hh1p", tiers=["train", "interp", "extrap"], caption="one-parameter H\\'enon--Heiles system", label="tab:verif-hh1"),
    "E4": dict(root="results/ahnn_repro/hh2p", tiers=["train", "interp", "extrap"], caption="two-parameter H\\'enon--Heiles system", label="tab:verif-hh2"),
    "E6": dict(root="results/e6_fpu", tiers=["train", "interp", "extrap"], caption="Fermi--Pasta--Ulam chain", label="tab:verif-fpu", k=4,
               models=MODELS[:2] + ["pshnn_concat_minv"] + MODELS[2:]),   # appendix table adds a comparison row (residual only)
}


def load_std(root, tiers):
    """<root>/<seed>/metrics*.csv (+ curves for E1 drift) → {(model,tier,block): [values]}"""
    v = defaultdict(list)
    for s in SEEDS:
        d = Path(root) / str(s); rows = defaultdict(dict)
        for fn in sorted(d.glob("metrics*.csv")):
            with open(fn) as fh:
                for r in csv.DictReader(fh):
                    rows[(r["model"], r["tier"])][(fn.name, r["metric"])] = float(r["value"])
        for m in MODELS:
            for t in tiers:
                rr = rows.get((m, t), {})
                vf = [x for (f, k), x in rr.items() if k == "vf_mse"]
                l2 = [x for (f, k), x in rr.items() if k == "rollout_l2_final" and f == ("metrics_rollout.csv" if (d / "metrics_rollout.csv").exists() else "metrics.csv")]
                if (d / "metrics_rollout.csv").exists():
                    dr = [x for (f, k), x in rr.items() if k == "energy_drift_final" and f == "metrics_rollout.csv"]
                else:   # E1: drift curve stored up to t=314 (one point every 0.1), value taken at t=125.6
                    c = np.load(d / f"curves_{m}.npz")[f"{t}/energy_drift"]; dr = [float(c[1256])]
                if vf: v[(m, t, "vf")].append(vf[0])
                if l2: v[(m, t, "l2")].append(l2[0])
                if dr: v[(m, t, "drift")].append(dr[0])
    return v


def load_e6(root, k):
    v = defaultdict(list)
    with open(Path(root) / f"metrics_grid_k{k}.csv") as fh:
        wide = {(r["model"], int(r["seed"])): r for r in csv.DictReader(fh)}
    fc = Path(root) / f"metrics_grid_k{k}_concat_minv.csv"   # comparison: w/o separation with M^{-1} represented (vector-field residual only)
    if fc.exists():
        with open(fc) as fh:
            for r in csv.DictReader(fh): wide[("pshnn_concat_minv", int(r["seed"]))] = r
    for s in SEEDS:
        fl = Path(root) / "long" / f"metrics_long_grid_k{k}_s{s}.csv"
        long = defaultdict(dict)
        if fl.exists():
            with open(fl) as fh:
                for r in csv.DictReader(fh):
                    long[(r["model"], r["tier"])][r["metric"]] = float(r["value"])
        for m in MODELS + ["pshnn_concat_minv"]:
            w = wide.get((m, s))
            if w is None: continue
            for t in ["train", "interp"]:
                v[(m, t, "vf")].append(float(w[f"vf_{t}"]))
                if (m, t) in long:
                    v[(m, t, "l2")].append(long[(m, t)]["rollout_l2_t20"]); v[(m, t, "drift")].append(long[(m, t)]["energy_drift_final"])
            v[(m, "extrap", "vf")].append(np.mean([float(w[f"vf_ex{i}"]) for i in (1, 2, 3)]))
            if all((m, f"ex{i}") in long for i in (1, 2, 3)):
                v[(m, "extrap", "l2")].append(np.mean([long[(m, f"ex{i}")]["rollout_l2_t20"] for i in (1, 2, 3)]))
                v[(m, "extrap", "drift")].append(np.mean([long[(m, f"ex{i}")]["energy_drift_final"] for i in (1, 2, 3)]))
    return v


def cell(vals):
    if not vals: return "--"
    a = np.array(vals, dtype=float); med, lo, hi = np.median(a), a.min(), a.max()
    e = int(math.floor(math.log10(med))) if med > 0 else 0
    f = lambda x: f"{x / 10 ** e:.1f}"
    return rf"${f(med)}\,[{f(lo)},{f(hi)}]\cdot10^{{{e}}}$"


def cell2(vals):
    """Two-line cell: first line the median (\\num), second line [min, max] in units of the median's power of ten, scriptsize."""
    if not vals: return "--", ""
    a = np.array(vals, dtype=float); med, lo, hi = np.median(a), a.min(), a.max()
    e = int(math.floor(math.log10(med))) if med > 0 else 0
    return rf"\num{{{med:.1e}}}", rf"{{\scriptsize$[{lo / 10 ** e:.1f},\,{hi / 10 ** e:.1f}]$}}"


def build(eid):
    """Full appendix table (portrait page, landscape layout): rows = seven models (two rows per model: median / [min, max]),
    columns = three quantities x {in range (train and interp merged), extrapolated}; best median per column in bold."""
    sp = SPECS[eid]; tiers = sp["tiers"]; models = sp.get("models", MODELS)
    v = load_e6(sp["root"], sp["k"]) if eid == "E6" else load_std(sp["root"], tiers)
    groups = ["in", "ex"] if "extrap" in tiers else ["in"]
    GL = {"in": "in range", "ex": "extrap."}
    def pooled(m, key, g):
        ts = ["train", "interp"] if g == "in" else ["extrap"]
        return [x for t in ts if t in tiers for x in v.get((m, t, key), [])]
    best = {(key, g): min((np.median(pooled(m, key, g)) for m in models if pooled(m, key, g)), default=None) for key, _ in BLOCKS for g in groups}
    ncol = len(BLOCKS) * len(groups)
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{Verification on the " + sp["caption"] + r": vector-field residual of Section~\ref{sec:verification:setup} on the stored test states, "
         r"trajectory error $\|\hat{\x}-\x\|_{2}$ after three linear periods and relative energy drift of the reference Hamiltonian "
         r"after twenty linear periods"
         + (r", at the trained and interpolated parameter values together (in range) and at the extrapolated ones. "
            r"Each entry is the median over the five seeds and, in range, the two sets of parameter values" if "extrap" in tiers
            else r". Each entry is the median over the five seeds")
         + r"; the second line gives the range $[\min,\max]$ "
         r"in the units of the power of ten of the median. Best value per column in bold.}",
         r"  \label{" + sp["label"] + "}", r"  \footnotesize\setlength{\tabcolsep}{2.5pt}\sisetup{tight-spacing=true,exponent-product=\cdot}",
         r"  \begin{tabular}{l" + "c" * ncol + "}", r"    \toprule"]
    L.append("    " + " & ".join([r"\multirow{2}{*}{Model}"] + [r"\multicolumn{" + str(len(groups)) + "}{c}{" + SHORT[k] + "}" for k, _ in BLOCKS]) + r" \\")
    L.append("    " + " ".join(rf"\cmidrule(lr){{{2 + i * len(groups)}-{1 + (i + 1) * len(groups)}}}" for i in range(len(BLOCKS))))
    L.append("    " + " & ".join([""] + [GL[g] for _ in BLOCKS for g in groups]) + r" \\")
    L.append(r"    \midrule")
    for m in models:
        top, bot = [], []
        for key, _ in BLOCKS:
            for g in groups:
                vals = pooled(m, key, g); a, b = cell2(vals)
                if vals and np.median(vals) == best[(key, g)]: a = r"\bfseries " + a
                top.append(a); bot.append(b)
        L.append("    " + LABEL[m] + " & " + " & ".join(top) + r" \\")
        L.append("    " + " & " + " & ".join(bot) + r" \\[2pt]")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = Path("paper/tables") / f"tab_verif_{eid}.tex"; out.write_text("\n".join(L) + "\n", encoding="utf-8")
    n = sum(1 for m in models for t in tiers for key, _ in BLOCKS if len(v.get((m, t, key), [])) == 5)
    print(f"{eid}: {out}  ({n}/{len(models) * len(tiers) * 3} tier-cells with 5 seeds)")


def build_compact(eid):
    """Main-text table: columns = three quantities x {in-range (train and interp merged), extrapolated}, cell = median (\\num), best per column in bold; ranges are in the full appendix table."""
    sp = SPECS[eid]; tiers = sp["tiers"]; models = [m for m in sp.get("models", MODELS) if m != "pshnn_concat_minv"]   # compact main-text table has no comparison row
    v = load_e6(sp["root"], sp["k"]) if eid == "E6" else load_std(sp["root"], tiers)
    groups = ["in", "ex"] if "extrap" in tiers else ["in"]
    def pooled(m, key, g):
        ts = ["train", "interp"] if g == "in" else ["extrap"]
        return [x for t in ts if t in tiers for x in v.get((m, t, key), [])]
    med = {(m, key, g): (np.median(pooled(m, key, g)) if pooled(m, key, g) else None) for m in models for key, _ in BLOCKS for g in groups}
    best = {(key, g): min(x for (m, k2, g2), x in med.items() if k2 == key and g2 == g and x is not None) for key, _ in BLOCKS for g in groups}
    GL = {"in": "in range", "ex": "extrap."}
    ncol = len(BLOCKS) * len(groups)
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + sp["caption"][0].upper() + sp["caption"][1:] + r": median over seeds of the vector-field residual, of the trajectory error after three linear periods "
         r"and of the energy drift after twenty linear periods"
         + (r", at the trained and interpolated parameter values together (in range) and at the extrapolated ones. " if "extrap" in tiers else r". ")
         + r"Best value per column in bold; seed ranges per set of parameter values in Table~\ref{" + sp["label"] + r"}.}",
         r"  \label{" + sp["label"] + r"-main}", r"  \footnotesize\setlength{\tabcolsep}{3pt}\sisetup{tight-spacing=true,exponent-product=\cdot}", r"  \begin{tabular}{l" + "c" * ncol + "}", r"    \toprule"]
    L.append("    " + " & ".join([r"\multirow{2}{*}{Model}"] + [r"\multicolumn{" + str(len(groups)) + "}{c}{" + SHORT[k] + "}" for k, _ in BLOCKS]) + r" \\")
    L.append("    " + " ".join(rf"\cmidrule(lr){{{2 + i * len(groups)}-{1 + (i + 1) * len(groups)}}}" for i in range(len(BLOCKS))))
    L.append("    " + " & ".join([""] + [GL[g] for _ in BLOCKS for g in groups]) + r" \\")
    L.append(r"    \midrule")
    for m in models:
        cells = []
        for key, _ in BLOCKS:
            for g in groups:
                x = med[(m, key, g)]
                if x is None: cells.append("--"); continue
                c = rf"\num{{{x:.1e}}}"
                cells.append((r"\bfseries " if x == best[(key, g)] else "") + c)
        L.append("    " + LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = Path("paper/tables") / f"tab_verif_{eid}_main.tex"; out.write_text("\n".join(L) + "\n", encoding="utf-8"); print(f"{eid}: {out}")


def build_rank_sweep():
    """Main-text table (landscape): rank sweep of Schmidt-HNN on FPU k=4, R_V in {1,2,3,4,6,8} (bias head on, R_L=1), vector-field residual,
    columns = R_V, rows = {in range (train and interp merged), extrapolated (three extrapolation ranges merged)}, two lines per row = median / [min, max]."""
    import pandas as pd
    root = Path("results/e6_fpu")
    files = [(1, "metrics_grid_k4_R1.csv"), (2, "metrics_grid_k4_R2.csv"), (3, "metrics_grid_k4.csv"), (4, "metrics_grid_k4_R4.csv"), (6, "metrics_grid_k4_R6.csv"), (8, "metrics_grid_k4_R8.csv")]
    rows = {"in range": [], "extrapolated": []}
    for R, fn in files:
        d = pd.read_csv(root / fn); d = d[d.model == "pshnn"]
        rows["in range"].append(cell2(list(d.vf_train) + list(d.vf_interp))); rows["extrapolated"].append(cell2(list((d.vf_ex1 + d.vf_ex2 + d.vf_ex3) / 3)))
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{Fermi--Pasta--Ulam chain: vector-field residual of the proposed architecture against the rank $R_{V}$, "
         r"with the bias head and $R_{L}=1$, in range (trained and interpolated parameter values) and extrapolated. Median over five seeds; "
         r"the second line gives the range $[\min,\max]$ in the units of the power of ten of the median. $R_{V}=3$ is the value read from the spectrum.}",
         r"  \label{tab:rank-sweep}", r"  \footnotesize\setlength{\tabcolsep}{5pt}\sisetup{tight-spacing=true,exponent-product=\cdot}",
         r"  \begin{tabular}{l" + "c" * len(files) + "}", r"    \toprule",
         "    $R_{V}$ & " + " & ".join(str(R) for R, _ in files) + r" \\", r"    \midrule"]
    for name, cells in rows.items():
        L.append(f"    {name} & " + " & ".join(a for a, _ in cells) + r" \\")
        L.append("     & " + " & ".join(b for _, b in cells) + r" \\[2pt]")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = Path("paper/tables") / "tab_rank_sweep.tex"; out.write_text("\n".join(L) + "\n", encoding="utf-8"); print(f"rank sweep: {out}")


if __name__ == "__main__":
    ids = sys.argv[1:] or ["E1", "E2", "E2A", "E3", "E4"]
    for eid in ids:
        if eid == "E6R": build_rank_sweep(); continue
        build(eid); build_compact(eid)
