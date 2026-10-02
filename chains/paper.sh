#!/usr/bin/env bash
# LaTeX tables and figures of the paper, written to paper/tables and paper/figures.
# The tables and the beam figures use the metrics, caches and checkpoints in results/.
# The verification figures replay the checkpoints on the trajectory data, which must exist in
# data_cache/ (chains/verification.sh generates them). Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p paper/tables paper/figures

# Tables
python3 scripts/paper_latex_tables.py E1 E2 E2A E3 E4 E6 E6R
python3 scripts/paper_latex_tables_beam.py
python3 scripts/paper_latex_tables_beam_loads.py
python3 scripts/paper_ahnn_repro_table.py

# Figures that need no trajectory data
python3 scripts/fig_taylor_hh1.py
python3 scripts/fig_beam_schematic.py
python3 scripts/fig_beam_free.py --replot
python3 scripts/fig_beam_backbone.py --replot
python3 scripts/fig_beam_loads.py
python3 scripts/fig_beam_perpoint.py

# Verification figures: replay the checkpoints (cached under results/fig_verif), then plot
if [ -d data_cache ]; then
  python3 scripts/fig_verif_curves.py replay E1 E2 E3 E4 E6   # error curves of every test trajectory
  python3 scripts/fig_verif_curves.py view   E1 E2 E3 E4 E6   # densely sampled representative trajectory
  python3 scripts/fig_verif_curves.py sweep  E2 E3 E4 E6      # residual against the parameter distance
  python3 scripts/fig_verif_curves.py plot   E1 E2 E3 E4 E6
else
  echo "data_cache/ not found: run chains/verification.sh first to draw the verification figures"
fi
