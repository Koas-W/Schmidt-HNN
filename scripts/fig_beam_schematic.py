"""Schematic of the clamped-clamped beam (beam_schematic.pdf).

Beam clamped at both ends: left end fixed, right end supported through an axial spring k and loaded by the axial preload N0; uniform pressure p; midpoint deflection w(L/2).
Pure vector graphics drawn with matplotlib, style consistent with the other figures (newtx font), width 3.6 in.
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, scienceplots  # noqa: F401
    from matplotlib.patches import FancyArrowPatch, Polygon
    os.environ["PATH"] = str(ROOT / ".texlive/bin/x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(ROOT / "scripts/figstyle/science.mplstyle")])
    fig, ax = plt.subplots(figsize=(3.6, 1.45)); fig.subplots_adjust(0, 0, 1, 1); ax.set_axis_off()
    ax.set_xlim(-0.55, 11.5); ax.set_ylim(-1.6, 1.75); ax.set_aspect("equal")
    L, h = 10.0, 0.28; x0 = 0.0
    # undeformed beam (thin line) and deformed beam (solid)
    x = np.linspace(x0, x0 + L, 200); w = 0.8 * np.sin(np.pi * (x - x0) / L) ** 2
    ax.plot([x0, x0 + L], [0, 0], color="0.55", lw=0.6, ls=(0, (3, 2)))
    ax.fill_between(x, -w - h / 2, -w + h / 2, color="0.75", ec="k", lw=0.6, zorder=3)   # deformed beam drawn as a band with thickness
    # left clamp: wall with hatching
    ax.plot([x0, x0], [-0.9, 0.9], color="k", lw=1.0)
    for y in np.linspace(-0.8, 0.8, 7): ax.plot([x0 - 0.28, x0], [y - 0.2, y], color="k", lw=0.5)
    # right end: vertically clamped (rotation restrained) but axially movable support, drawn as a slider connected to the wall through the axial spring
    xr = x0 + L
    ax.plot([xr, xr], [-0.6, 0.6], color="k", lw=1.0)
    ax.add_patch(Polygon([[xr, -0.6], [xr, 0.6], [xr + 0.18, 0.6], [xr + 0.18, -0.6]], closed=True, fc="0.85", ec="k", lw=0.5))
    # spring k
    xs = np.linspace(xr + 0.18, xr + 0.95, 9); ys = 0.16 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
    ax.plot(xs, ys, color="k", lw=0.8)
    ax.plot([xr + 0.95, xr + 0.95], [-0.9, 0.9], color="k", lw=1.0)
    for y in np.linspace(-0.8, 0.8, 7): ax.plot([xr + 0.95, xr + 1.23], [y, y + 0.2], color="k", lw=0.5)
    ax.text(xr + 0.56, 0.42, r"$k$", ha="center", va="bottom")
    # axial preload N0 (acting on the right slider, pointing left = compression)
    ax.add_patch(FancyArrowPatch((xr + 1.15, -0.9), (xr + 0.2, -0.9), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="k"))
    ax.text(xr + 0.68, -0.98, r"$N_{0}$", ha="center", va="top")
    # uniform pressure p
    for xi in np.linspace(x0 + 0.5, x0 + L - 0.5, 12):
        ax.add_patch(FancyArrowPatch((xi, 1.35), (xi, 0.42), arrowstyle="-|>", mutation_scale=5, lw=0.6, color="k"))
    ax.plot([x0 + 0.5, x0 + L - 0.5], [1.35, 1.35], color="k", lw=0.6)
    ax.text(x0 + L / 2, 1.42, r"$p$", ha="center", va="bottom")
    # midpoint deflection w(L/2), thickness h and length L
    xm = x0 + L / 2
    ax.add_patch(FancyArrowPatch((xm, 0), (xm, -0.8), arrowstyle="<|-|>", mutation_scale=5, lw=0.6, color="k", zorder=4))
    ax.text(xm + 0.15, -0.36, r"$w(L/2)$", ha="left", va="center")
    ax.add_patch(FancyArrowPatch((x0, -1.4), (x0 + L, -1.4), arrowstyle="<|-|>", mutation_scale=5, lw=0.6, color="k"))
    ax.text(x0 + L / 2, -1.47, r"$L$", ha="center", va="top")   # label at midspan
    xh = x0 + 0.9; wh = 0.8 * np.sin(np.pi * (xh - x0) / L) ** 2
    ax.add_patch(FancyArrowPatch((xh, -wh - h / 2 - 0.35), (xh, -wh - h / 2), arrowstyle="-|>", mutation_scale=4, lw=0.5, color="k"))
    ax.add_patch(FancyArrowPatch((xh, -wh + h / 2 + 0.35), (xh, -wh + h / 2), arrowstyle="-|>", mutation_scale=4, lw=0.5, color="k"))
    ax.text(xh + 0.15, -wh + h / 2 + 0.2, r"$h$", ha="left", va="center")
    out = ROOT / "paper/figures"; fig.savefig(out / "beam_schematic.pdf"); fig.savefig(out / "beam_schematic.png"); plt.close(fig)
    print("written", out / "beam_schematic.pdf")


if __name__ == "__main__":
    main()
