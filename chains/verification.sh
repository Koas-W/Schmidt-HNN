#!/usr/bin/env bash
# Verification examples of the paper: pendulum, double pendulum, Henon-Heiles (one and two
# parameters) and the Fermi-Pasta-Ulam chain. Data, training of all models with five seeds,
# long rollouts and Taylor coefficients. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pendulum
python3 scripts/generate_data.py  --config configs/pendulum.yaml
python3 scripts/run_experiment.py --config configs/e1_pendulum.yaml
python3 scripts/eval_long.py      --config configs/e1_pendulum.yaml --seeds 0 1 2 3 4

# Double pendulum (main run and the run with the inverse-mass representation)
python3 scripts/generate_data.py  --config configs/double_pendulum.yaml
python3 scripts/run_experiment.py --config configs/e2_double_pendulum.yaml
python3 scripts/run_experiment.py --config configs/e2_double_pendulum_minv.yaml
python3 scripts/eval_rollout.py   --config configs/e2_double_pendulum.yaml      --results-dir results/e2_double_pendulum      --seeds 0 1 2 3 4 --t-long 125.6
python3 scripts/eval_rollout.py   --config configs/e2_double_pendulum_minv.yaml --results-dir results/e2_double_pendulum_minv --seeds 0 1 2 3 4 --t-long 125.6

# Henon-Heiles, one and two parameters (data are generated inside run_ahnn_repro.py)
python3 scripts/run_ahnn_repro.py --config configs/ahnn_hh1p.yaml
python3 scripts/run_ahnn_repro.py --config configs/ahnn_hh2p.yaml
python3 scripts/eval_rollout.py   --config configs/ahnn_hh1p.yaml --results-dir results/ahnn_repro/hh1p --seeds 0 1 2 3 4 --t-long 125.6
python3 scripts/eval_rollout.py   --config configs/ahnn_hh2p.yaml --results-dir results/ahnn_repro/hh2p --seeds 0 1 2 3 4 --t-long 125.6
python3 scripts/taylor_per_seed.py

# Fermi-Pasta-Ulam chain: 4 levels per parameter axis, rank 3 with bias head, then the rank sweep
python3 scripts/run_fpu.py --models mlp_ode ahnn symoden_mu pshnn_concat sep_hnn pshnn_mlp pshnn_silu pshnn \
    --n-chain 8 --sample-every 40 --pshnn-R 3 --bias-head --mu-design grid --grid-k 4 --fixed-eval --cuda-graph \
    --max-steps 20000 --patience 30 --seeds 0 1 2 3 4 --tag grid_k4
for R in 1 2 4 6 8; do
  python3 scripts/run_fpu.py --models pshnn \
      --n-chain 8 --sample-every 40 --pshnn-R $R --bias-head --mu-design grid --grid-k 4 --fixed-eval --cuda-graph \
      --max-steps 20000 --patience 30 --seeds 0 1 2 3 4 --tag grid_k4_R$R
done
python3 scripts/run_fpu_concat_minv.py --models pshnn_concat \
    --n-chain 8 --sample-every 40 --pshnn-R 3 --bias-head --mu-design grid --grid-k 4 --fixed-eval --cuda-graph \
    --max-steps 20000 --patience 30 --seeds 0 1 2 3 4 --tag grid_k4_concat_minv
python3 scripts/eval_fpu_grid.py --ks 4 --seeds 0 1 2 3 4

# Singular value spectra and ranks of the restoring forces
python3 scripts/theory_checks.py eta
python3 scripts/rank_from_data.py
