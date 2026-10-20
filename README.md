# Schmidt Hamiltonian neural networks

Code, configurations and result tables for the paper
*Schmidt Hamiltonian neural networks: identification of parametric mechanical
energies from recorded motions*.

The network identifies the Hamiltonian H(q, p; mu) of a parametric natural
mechanical system from recorded motions. A parameter network and a coordinate
network are coupled by an inner product (the Schmidt expansion of the energy),
the kinetic energy is represented by a Cholesky factor of the mass matrix, and
gated residual blocks represent the polynomial part of the energy exactly. The
loss is the residual of Hamilton's equations on the recorded states.

## Layout

```
src/        library: systems, data generation, models, training, evaluation
  systems/    pendulum, double pendulum, Henon-Heiles, von Karman beam (q, p state order)
  data/       fixed-step RK4 in float64, trajectory sampling, parameter designs
  models/     Schmidt-HNN, its ablation variants, baselines, parameter-count matching
  train/      loss on Hamilton's equations, AdamW trainer with early stopping
  analysis/   rollout errors, energy drift, backbone curves, Taylor coefficients
  synthetic/  Fermi-Pasta-Ulam chain with parametric coupling
configs/    one YAML per data set and per experiment (no hyperparameter is hard-coded)
scripts/    command-line entry points, grouped by prefix
  generate_, run_, eval_   data generation, training, evaluation
  beam_                    beam-specific evaluation (SINDy regression, static deflection,
                           midpoint error, saturation, load cases)
  fig_, paper_             figures and LaTeX tables of the paper
tests/      unit tests (python3 -m unittest discover -s tests)
chains/     shell scripts that run the pipelines in order
results/    metrics (csv), configuration snapshots and checkpoints of the paper runs
paper/      output directory of the fig_ and paper_ scripts
```

## Names used in the code

| Code name       | Name in the paper                          |
|-----------------|--------------------------------------------|
| `pshnn`         | Schmidt-HNN (proposed architecture)        |
| `pshnn_concat`  | w/o separation (parameters concatenated)   |
| `pshnn_mlp`     | w/o gating, tanh blocks                    |
| `pshnn_silu`    | w/o gating, SiLU blocks                    |
| `sep_hnn`       | w/o Cholesky (no kinetic-energy structure) |
| `ahnn`          | AHNN (Han et al. 2021)                     |
| `silu_hnn`      | AHNN-SiLU                                  |
| `symoden_mu`    | SymODEN (Zhong et al. 2020)                |
| `mlp_ode`       | MLP (direct vector-field regression)       |
| `sindy`         | SINDy on polynomial dictionaries           |

Parameter tiers: `train` (trained values), `interp` (interpolated values inside
the trained box), `extrap` (extrapolated values outside the box; the beam uses
three extrapolation ranges `ex1`, `ex2`, `ex3`).

Experiments and their result directories:

| Directory                          | System                                   |
|------------------------------------|------------------------------------------|
| `results/e1_pendulum`              | pendulum                                 |
| `results/e2_double_pendulum`       | double pendulum, two parameters          |
| `results/e2_double_pendulum_minv`  | same, Schmidt-HNN family with M^-1 = L L^T |
| `results/ahnn_repro/hh1p`, `hh2p`  | Henon-Heiles, one and two parameters     |
| `results/e6_fpu`                   | Fermi-Pasta-Ulam chain, four parameters  |
| `results/beam_main`                | clamped-clamped von Karman beam          |
| `results/beam_sindy`               | SINDy regressions on the beam data       |
| `results/beam_checks`              | beam data checks (modal convergence, step size, drift) |
| `results/theory`                   | singular value spectra and ranks         |

## Requirements

Python 3.10 with `torch`, `numpy`, `matplotlib`, `pyyaml`, `pandas` and
`scienceplots` (figures only). The paper runs used torch 2.11 with CUDA;
training falls back to the CPU but is slow there. The integrators are
implemented in `src/data/integrators.py`; no ODE library is needed.

The figure scripts typeset their text with LaTeX (the newtx fonts, matching
the Elsevier class). They look for a TeX Live installation at `.texlive/` in
the repository root and otherwise use the `latex` found on the PATH, which
must provide the `newtx` package.

## Reproducing the paper

Every step reads its settings from `configs/` and writes under `results/`.
The scripts in `chains/` run the steps in order:

```
bash chains/verification.sh      # pendulum, double pendulum, Henon-Heiles, FPU chain
bash chains/beam.sh              # von Karman beam: data, training, SINDy, evaluation, load cases
bash chains/paper.sh             # all LaTeX tables and figures into paper/
```

`chains/paper.sh` runs on the checkpoints and metrics in `results/`,
so the tables and figures of the paper can be regenerated without training.
The only exception is the verification figures, whose per-trajectory curve
cache (`results/fig_verif/`, about 500 MB) is not kept in the repository; `chains/paper.sh`
rebuilds it from the checkpoints before plotting.

Trajectory data (`data_cache/`) are not kept in the repository. `generate_data.py` recreates
them deterministically from the data configurations; the step size of every
system is frozen in its configuration after a convergence study, with a
relative energy drift of the reference solution below 1e-8 over the recorded
span.

## Tests

```
python3 -m unittest discover -s tests
```

The tests cover the sign conventions of Hamilton's equations, the integrators
and their frozen step sizes, the analytical gradients of every model, the
parameter-count matching, the data pipeline and the beam model.
