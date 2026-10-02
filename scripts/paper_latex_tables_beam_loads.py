"""Table tab_beam_loads.tex: beam dynamic load cases, two loading lines x two cases (near-resonant harmonic pressure, uniform pressure impulse).

Columns: reference peak w/h; for three models (Schmidt-HNN, SiLU potential network, sparse regression (4,3)) the relative peak error (relative to the dynamic peak, percent)
and the twenty-period midpoint deflection RMS error (h). Data: results/beam_main/dyn_cases/dyn_cases_<tag>_<cases>_20T1_s2.csv (scripts/beam_dyn_cases.py).
"""
import csv, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "results/beam_main/dyn_cases"
LINES = [(("typical", "typical_blast60"), "interpolated line"), (("extrap_r1",), "extrapolated line")]   # (set of line tags in the csv, row label)
CASES = [("res", "near-resonant pressure"), ("blast", "pressure impulse")]
MODELS = [("pshnn", "Schmidt-HNN"), ("silu_hnn", "AHNN-SiLU"), ("sindy", "SINDy (4, 3)")]   # two header rows, controls the table width


def load():
    rows = {}
    for f in glob.glob(str(D / "dyn_cases_*_s2.csv")):
        for r in csv.DictReader(open(f)):
            rows[(r["line"], r["case"], r["model"])] = r
    return rows


def main():
    R = load()
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{Reference peak deflection, relative error of the dynamic peak measured from the static deflection,",
         r"    and RMS error of the midpoint deflection over twenty periods. Best value per row in bold.}",
         r"  \label{tab:beam-loads}",
         r"  \footnotesize\setlength{\tabcolsep}{3pt}\sisetup{tight-spacing=true,exponent-product=\cdot}",
         r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc" + "cc" * len(MODELS) + "@{}}", r"    \toprule",
         r"     & Peak & " + " & ".join(r"\multicolumn{2}{c}{\begin{tabular}[b]{@{}c@{}}" + lab + r"\end{tabular}}" for _, lab in MODELS) + r" \\",
         "    " + " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(MODELS))),
         r"     & $w/h$ & " + " & ".join([r"peak (\%) & RMS ($h$)"] * len(MODELS)) + r" \\", r"    \midrule"]
    ncol = 2 + 2 * len(MODELS)
    for ci, (c, cname) in enumerate(CASES):   # case as group header row, lines as rows
        if ci: L.append(r"    \addlinespace[2pt]")
        L.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l}}{{\itshape {cname[0].upper() + cname[1:]}}} \\")
        for tags, lname in LINES:
            rs = [next((R[(t, c, m)] for t in tags if (t, c, m) in R), None) for m, _ in MODELS]
            if rs[0] is None: continue
            bp = min(float(r["peak_rel_err_dyn"]) for r in rs if r); br = min(float(r["rms_err_h"]) for r in rs if r)
            cells = []
            for r in rs:
                if r is None: cells += ["--", "--"]; continue
                pk, rm = float(r["peak_rel_err_dyn"]), float(r["rms_err_h"])
                cells += [(r"\bfseries " if pk == bp else "") + f"{100 * pk:.2f}", (r"\bfseries " if rm == br else "") + rf"\num{{{rm:.1e}}}"]
            L.append(f"    {lname} & {float(rs[0]['peak_ref_h']):.2f} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular*}", r"\end{table}"]
    out = ROOT / "paper/tables/tab_beam_loads.tex"; out.write_text("\n".join(L) + "\n"); print("written", out)


if __name__ == "__main__":
    main()
