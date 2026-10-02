"""Model evaluation, three metrics:
1. vector-field test MSE (per tier: train/interp/extrap);
2. rollout pointwise L2 error curve (fixed horizon t_short, compared with the stored ground-truth trajectory at every time);
3. energy drift |H_true(t)-H_true(0)|/|H_true(0)| (horizon t_long, ground-truth Hamiltonian
   evaluated along the predicted trajectory, a measure of physical consistency).

The rollout uses the same integrator as the ground truth (same RK4, same step);
all tiers are concatenated into one batch and integrated once (faster).
"""
import torch

from ..data.integrators import rollout_sampled, rollout_sampled_graphed  # noqa: F401
from ..train.trainer import evaluate_loss


def model_field(model, mu: torch.Tensor):
    def f(y: torch.Tensor) -> torch.Tensor:
        return model.time_derivative(y, mu, create_graph=False)
    return f


@torch.no_grad()
def _true_H(system, states: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """states (T,B,2n) float64 -> H (T,B)."""
    n = system.dim
    T, B, D = states.shape
    flat = states.reshape(T * B, D)
    return system.hamiltonian(flat[:, :n], flat[:, n:],
                              mu.repeat(T, 1)).reshape(T, B)


def evaluate_model(model, system, views: dict, data_cfg: dict,
                   eval_cfg: dict, device) -> tuple[dict, dict]:
    """Return (metrics, curves).

    metrics rows: {(tier, metric): value}; curves: {f"{tier}/{name}": tensor}.
    """
    model.to(device).eval()
    dt = data_cfg["dt"]
    sample_every = data_cfg["sample_every"]
    n_rollout = eval_cfg["n_rollout"]
    metrics: dict[tuple[str, str], float] = {}
    curves: dict[str, torch.Tensor] = {}

    # ---- 1. vector-field MSE (per tier) ------------------------------------
    for tier, view in views.items():
        metrics[(tier, "vf_mse")] = evaluate_loss(model, view, device)

    # ---- batch: the first n_rollout test trajectories of every tier ---------
    tiers, refs, mus, slices = [], [], [], {}
    offset = 0
    for tier, view in views.items():
        traj = view["traj"][:, :n_rollout]            # (n_mu, k, T+1, 2n)
        n_mu, k, T1, D = traj.shape
        refs.append(traj.reshape(n_mu * k, T1, D))
        mus.append(view["mu_grid"].repeat_interleave(k, 0))
        slices[tier] = (offset, offset + n_mu * k)
        offset += n_mu * k
        tiers.append(tier)
    ref = torch.cat(refs)                              # (B, T+1, 2n) float64
    mu_all = torch.cat(mus)
    y0 = ref[:, 0].to(torch.float32).to(device)
    mu_dev = mu_all.to(torch.float32).to(device)
    f = model_field(model, mu_dev)

    # ---- 2. short-horizon pointwise L2 --------------------------------------
    # no_grad: the MLP-ODE output stays in the graph (HNN-style models enable_grad+detach internally);
    # otherwise a 10k-step rollout would drag along the full computation graph
    n_short = int(round(eval_cfg["rollout_t_short"] / dt))
    with torch.no_grad():
        pred = rollout_sampled_graphed(f, y0, dt, n_short, sample_every)  # (Ts,B,D)
    pred64 = pred.cpu().to(torch.float64)
    Ts = pred64.shape[0]
    ref_t = ref.permute(1, 0, 2)[:Ts]                  # (Ts, B, D)
    l2 = ((pred64 - ref_t) ** 2).sum(-1).sqrt()        # (Ts, B)
    for tier in tiers:
        a, b = slices[tier]
        curve = l2[:, a:b].mean(1)
        curves[f"{tier}/rollout_l2"] = curve
        metrics[(tier, "rollout_l2_final")] = curve[-1].item()
        metrics[(tier, "rollout_l2_mean")] = curve.mean().item()

    # ---- 3. long-horizon energy drift ---------------------------------------
    n_long = int(round(eval_cfg["rollout_t_long"] / dt))
    with torch.no_grad():
        pred_long = rollout_sampled_graphed(f, y0, dt, n_long, sample_every)
    pl64 = pred_long.cpu().to(torch.float64)
    H = _true_H(system, pl64, mu_all)
    drift = (H - H[0]).abs() / H[0].abs().clamp_min(1e-12)  # (Tl, B)
    for tier in tiers:
        a, b = slices[tier]
        curve = drift[:, a:b].mean(1)
        curves[f"{tier}/energy_drift"] = curve
        metrics[(tier, "energy_drift_final")] = curve[-1].item()
        metrics[(tier, "energy_drift_max")] = curve.max().item()
    return metrics, curves
