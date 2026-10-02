"""Tables for the clamped-beam free-vibration identification, engineering quantities.

tab_beam_free.tex: rows = Schmidt-HNN, SiLU potential network, AHNN, SymODEN, MLP, sparse regression (d_q=4, d_mu=3);
columns = VPT (periods) and midpoint deflection error RMS/h, each split into in range (trained and interpolated) and extrapolated,
plus relative static-equilibrium error and relative backbone-frequency error (A/h=3), each split into interpolated and extrapolated (three extrapolation ranges merged).
Each cell = median over five seeds (sparse regression is deterministic, single value); best per column in bold.
Data: results/beam_main/midpoint_error_s*.csv, midpoint_error_sindy.csv, eval_beam*.csv, eval_beam_sindy.csv.
tab_beam_sindy.tex: sparse-regression dictionary sweep (results/beam_sindy/sindy_beam_*.csv), vector-field MSE over the three tiers.
"""
import csv, glob, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BK = ROOT / "results/beam_main"; BS = ROOT / "results/beam_sindy"
MODELS = ["pshnn", "silu_hnn", "ahnn", "symoden_mu", "mlp_ode", "sindy_potential_dq4_dmu3"]
LABEL = {"pshnn": "Schmidt-HNN", "silu_hnn": "AHNN-SiLU", "ahnn": "AHNN", "symoden_mu": "SymODEN", "mlp_ode": "MLP",
         "sindy_potential_dq4_dmu3": "SINDy (4, 3)"}


def load_midpoint():
    """{(model, tier, metric): [median per seed]}; tier in train/interp/extrap."""
    v = defaultdict(list)
    for f in sorted(glob.glob(str(BK / "midpoint_error_s*.csv"))) + [str(BK / "midpoint_error_sindy.csv")]:
        for r in csv.DictReader(open(f)):
            v[(r["model"], r["tier"], r["metric"])].append(float(r["median"]))
    return v


def load_shells():
    """{(model, metric, group): [values]}; group in interp/extrap (three extrapolation ranges merged); static equilibrium and backbone frequency per evaluation point."""
    v = defaultdict(list)
    for f in [BK / "eval_beam.csv", BK / "eval_beam_silu_hnn.csv", BK / "eval_beam_sindy.csv"]:
        if not f.exists(): continue
        for r in csv.DictReader(open(f)):
            if r["metric"] in ("static_eq_rel_l2", "omega1_rel_err_A3"):
                v[(r["model"], r["metric"], r["tier"])].append(float(r["value"]))
    return v


def fmt(x, kind):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "--"
    if kind == "vpt": return f"{x:.1f}"
    return rf"\num{{{x:.1e}}}"


def build_free():
    mp, sh = load_midpoint(), load_shells()
    def pooled_mp(m, metric, g):
        ts = ["train", "interp"] if g == "in" else ["extrap"]
        vals = [x for t in ts for x in mp.get((m, t, metric), [])]
        return float(np.median(vals)) if vals else None
    def pooled_sh(m, metric, g):
        vals = sh.get((m, metric, "interp" if g == "in" else "extrap"), [])
        return float(np.median(vals)) if vals else None
    cols = [("vpt_cycles", "in", "vpt", "mp"), ("vpt_cycles", "ex", "vpt", "mp"), ("rms_h", "in", "sci", "mp"), ("rms_h", "ex", "sci", "mp"),
            ("static_eq_rel_l2", "in", "sci", "sh"), ("static_eq_rel_l2", "ex", "sci", "sh")]   # backbone frequency is covered by the beam_backbone figure instead
    table = {m: [(pooled_mp if src == "mp" else pooled_sh)(m, k, g) for k, g, _, src in cols] for m in MODELS}
    best = []
    for j, (k, g, kind, _) in enumerate(cols):
        vals = [table[m][j] for m in MODELS if table[m][j] is not None]
        best.append((max(vals) if kind == "vpt" else min(vals)) if vals else None)
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{VPT in periods, RMS midpoint error in units of $h$ and relative error of the static deflection,",
         r"    median over the parameter values and the seeds; the static deflection is evaluated at the interpolated",
         r"    and the extrapolated values, not at the trained ones. Best value per column in bold.}",
         r"  \label{tab:beam-free}",
         r"  \footnotesize\setlength{\tabcolsep}{3pt}\sisetup{tight-spacing=true,exponent-product=\cdot}",
         r"  \begin{tabular}{lcccccc}", r"    \toprule",
         r"    \multirow{2}{*}{Model} & \multicolumn{2}{c}{VPT} & \multicolumn{2}{c}{RMS error} & \multicolumn{2}{c}{Static deflection} \\",
         r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         r"     & in range & extrap. & in range & extrap. & in range & extrap. \\", r"    \midrule"]
    for m in MODELS:
        cells = []
        for j, (k, g, kind, _) in enumerate(cols):
            x = table[m][j]; c = fmt(x, kind)
            if x is not None and best[j] is not None and x == best[j]: c = r"\bfseries " + c
            cells.append(c)
        L.append("    " + LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = ROOT / "paper/tables/tab_beam_free.tex"; out.write_text("\n".join(L) + "\n"); print("written", out)


def build_sindy():
    rows = {}
    for f in sorted(glob.glob(str(BS / "sindy_beam_dq*_dmu*.csv"))):   # the oracle with exact parameter features is kept on disk only, not in the paper
        for r in csv.DictReader(open(f)):
            key = (int(r["dq"]), int(r["dmu"]))
            rows.setdefault(key, {})[r["tier"]] = float(r["vf_mse"]); rows[key]["n_cols"] = int(r["n_cols"]); rows[key]["lam"] = float(r["lambda"]); rows[key]["nnz"] = int(r["n_nonzero"])
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{SINDy on dictionaries of degree $d_{q}$ in $\q$ and $d_{\mu}$ in $\mub$: number of",
         r"    terms, terms retained after thresholding, and MSE of the restoring force on the test states at the trained,",
         r"    interpolated and extrapolated values. The dictionary with the smallest extrapolated MSE is used.}",
         r"  \label{tab:beam-sindy}",
         r"  \footnotesize\setlength{\tabcolsep}{4pt}\sisetup{tight-spacing=true,exponent-product=\cdot}",
         r"  \begin{tabular}{ccccccc}", r"    \toprule",
         r"    $d_{q}$ & $d_{\mu}$ & terms & retained & trained & interpolated & extrapolated \\", r"    \midrule"]
    for (dq, dm), r in sorted(rows.items()):
        L.append(f"    {dq} & {dm} & {r['n_cols']} & {r['nnz']} & " + " & ".join(rf"\num{{{np.format_float_positional(r[t], precision=4, unique=False, fractional=False, trim='-')}}}" for t in ("train", "interp", "extrap")) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = ROOT / "paper/tables/tab_beam_sindy.tex"; out.write_text("\n".join(L) + "\n"); print("written", out)


if __name__ == "__main__":
    build_free(); build_sindy()
