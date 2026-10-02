"""Figure beam_loads.pdf: midpoint deflection time histories of the beam under the dynamic load cases.

2x2: rows = load case (near-resonant harmonic pressure, uniform pressure impulse), columns = loading line (interpolated, extrapolated).
The near-resonant row shows all twenty periods (beat envelope), the impulse row only the last three periods (where the phase drift has accumulated); shared legend on top.
Curves: reference solution, Schmidt-HNN, SiLU potential network, sparse regression (4,3); style follows fig_beam_free.py.
Data: results/beam_main/dyn_cases/traj_*.npz (scripts/beam_dyn_cases.py).
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from fig_verif_curves import LEG  # noqa: E402
from fig_beam_free import LAB, STY  # noqa: E402

D = ROOT / "results/beam_main/dyn_cases"
# (row label, case name) -> npz file of each line and the column index of that case in the file
SRC = {"res": {"typical": ("traj_typical_res-param-impact_20T1_s2.npz", 0), "extrap": ("traj_extrap_r1_res-blast_20T1_s2.npz", 0)},
       "blast": {"typical": ("traj_typical_blast60_blast_20T1_s2.npz", 0), "extrap": ("traj_extrap_r1_res-blast_20T1_s2.npz", 1)}}
ROWS = [("res", "near-resonant pressure", (0, 20)), ("blast", "pressure impulse", (17, 20))]   # (case, title, t/T_1 range)
COLS = [("typical", r"$(\mu_1,\mu_2)=(-0.761,-0.025)$"), ("extrap", r"$(-1.153,-0.259)$")]


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, scienceplots  # noqa: F401
    os.environ["PATH"] = str(ROOT / ".texlive/bin/x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(ROOT / "scripts/figstyle/science.mplstyle")])
    HLEG = dict(LEG, fontsize=LEG["fontsize"] * 0.8, handlelength=1.4, borderpad=0.3, columnspacing=1.2, handletextpad=0.5)
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.8)); fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.11, wspace=0.25, hspace=0.45)
    for r, (c, cname, xl) in enumerate(ROWS):
        for k, (tag, lname) in enumerate(COLS):
            fn, j = SRC[c][tag]; z = np.load(D / fn); t = z["t"] / float(z["T1"]); ax = axes[r][k]
            ax.plot(t, z["w_ref"][:, j], color="k", linewidth=0.8, label=LAB["truth"], zorder=5)
            for key in ["sindy", "silu_hnn", "pshnn"]:
                ax.plot(t, z[f"w_{key}"][:, j], label=LAB[key], **STY[key])
            ax.set_xlim(*xl); ax.set_title(f"({chr(97 + 2 * r + k)}) {cname}, {lname}", loc="left", pad=3)
            if k == 0: ax.set_ylabel(r"$w(L/2)/h$")
            if r == 1: ax.set_xlabel(r"$t/T_{1}$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=4, **HLEG)
    fig.savefig(ROOT / "paper/figures/beam_loads.pdf"); fig.savefig(ROOT / "paper/figures/beam_loads.png"); plt.close(fig)
    print("written paper/figures/beam_loads.pdf")


if __name__ == "__main__":
    main()
