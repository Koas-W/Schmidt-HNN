#!/usr/bin/env bash
# Clamped-clamped von Karman beam: data checks, data, training, SINDy regressions, evaluation
# and the load cases of the paper. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
SINDY=results/beam_sindy/coef_potential_dq4_dmu3.pt

# Data checks: number of retained modes, integration step, drift of the stored trajectories
python3 scripts/beam_convergence.py --n-list 5 6
python3 scripts/beam_dt_sweep.py --dts 5e-4 2.5e-4 1e-4 5e-5 --t-span 5
python3 scripts/generate_data.py --config configs/clamped_beam.yaml
python3 scripts/beam_data_qa.py --data data_cache/clamped_beam

# Training: five models, five seeds
python3 scripts/run_experiment.py --config configs/beam_main.yaml

# SINDy on polynomial dictionaries (degree in q, degree in mu); the (4, 3) dictionary is used in the paper
python3 scripts/beam_sindy.py --variants potential --dq 4 --dmu 3 && cp results/beam_sindy/sindy_beam.csv results/beam_sindy/sindy_beam_dq4_dmu3.csv
python3 scripts/beam_sindy.py --variants potential --dq 4 --dmu 4 && cp results/beam_sindy/sindy_beam.csv results/beam_sindy/sindy_beam_dq4_dmu4.csv
python3 scripts/beam_sindy.py --variants potential --dq 6 --dmu 3 && cp results/beam_sindy/sindy_beam.csv results/beam_sindy/sindy_beam_dq6_dmu3.csv
python3 scripts/beam_sindy.py --variants potential --dq 6 --dmu 4 && cp results/beam_sindy/sindy_beam.csv results/beam_sindy/sindy_beam_dq6_dmu4.csv
python3 scripts/beam_sindy.py --variants potential --dq 4 --dmu 3 --mu-basis exact && cp results/beam_sindy/sindy_beam.csv results/beam_sindy/sindy_beam_dq4_exact.csv
python3 scripts/beam_eval_sindy.py $SINDY

# Free vibration: midpoint error, static deflection and backbone curve, per-point errors, saturation
python3 scripts/beam_midpoint_error.py --exp beam_main --seeds 0 1 2 3 4 --device cuda
python3 scripts/beam_midpoint_error.py --exp beam_main --sindy $SINDY
python3 scripts/beam_eval_shells.py --exp beam_main
python3 scripts/beam_eval_shells.py --exp beam_main --models silu_hnn
python3 scripts/beam_eval_shells.py --exp beam_main --sindy $SINDY
python3 scripts/beam_perpoint.py --seeds 0 1 2 3 4
python3 scripts/beam_saturation.py

# Load cases: parameter lines selected by the median error ratio, near-resonant pressure and pressure impulse
python3 scripts/beam_select_point.py --seed 2
python3 scripts/beam_select_point.py --seed 2 --tier extrap --shell 1
python3 scripts/beam_dyn_cases.py --mu12 -0.761 -0.025 --cases res param impact --tag typical
python3 scripts/beam_dyn_cases.py --mu12 -0.761 -0.025 --cases blast --impulse 60 --tag typical_blast60
python3 scripts/beam_dyn_cases.py --mu12 -1.153 -0.259 --cases res blast --impulse 60 --tag extrap_r1
