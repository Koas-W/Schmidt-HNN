"""Ground-truth trajectory generation.

- Initial conditions: uniform proposals inside phase_space_box; if the system
  defines energy_window(mu), rejection sampling on the energy window (same as
  AHNN's "energy below the escape threshold"); for the double pendulum this
  keeps trajectories in the oscillatory/compact region and out of the rotating
  chaotic region;
- Integration: fixed-step RK4, float64, step size from config (convergence verified);
- Storage: one state every sample_every steps (matches the 0.1 sampling interval of AHNN);
- Training target: analytic vector field on the stored (clean) states;
- Noise: optional, added only to the states of the train/val roles; test-role
  trajectories stay clean as the rollout reference (default noise_std=0, as in AHNN).

Role split (fairness: fixed at generation time, shared by all models/seeds):
the trajectories of each mu are ordered [train | val | test], boundaries in the manifest.
"""
import torch

from ..systems.base import HamiltonianSystem
from .integrators import rollout_sampled, system_field


def sample_initial_conditions(system: HamiltonianSystem,
                              mu_row: torch.Tensor, n_traj: int,
                              lo: torch.Tensor, hi: torch.Tensor,
                              g: torch.Generator) -> torch.Tensor:
    """Uniform proposals inside the box + optional energy-window rejection sampling; returns (n_traj, 2n) float64."""
    n = system.dim
    custom = system.sample_ic(mu_row.to(torch.float64), n_traj, g)
    if custom is not None:                   # system-specific sampler (see base.sample_ic)
        return custom
    window = system.energy_window(mu_row[None].to(torch.float64))
    if window is None:
        return lo + (hi - lo) * torch.rand(n_traj, 2 * n, generator=g,
                                           dtype=torch.float64)
    e_min = window[0].item()
    e_max = window[1].item()
    accepted, need = [], n_traj
    for _ in range(200):  # safety cap; a too-low acceptance rate is a configuration error
        m = max(4096, 4 * need)
        prop = lo + (hi - lo) * torch.rand(m, 2 * n, generator=g,
                                           dtype=torch.float64)
        mu_rep = mu_row.expand(m, -1).to(torch.float64)
        H = system.hamiltonian(prop[:, :n], prop[:, n:], mu_rep)
        mask = ((H >= e_min) & (H <= e_max)
                & system.ic_valid(prop[:, :n], prop[:, n:], mu_rep))
        ok = prop[mask]
        accepted.append(ok[:need])
        need -= min(need, ok.shape[0])
        if need == 0:
            return torch.cat(accepted)
    raise RuntimeError(
        f"energy-window rejection sampling did not fill up after 200 rounds: E in [{e_min:.3g},{e_max:.3g}], "
        f"mu={mu_row.tolist()} (acceptance rate too low; check the sampling box / energy window)")


def generate_tier(system: HamiltonianSystem,
                  mu_values: torch.Tensor,
                  n_traj: int,
                  dt: float,
                  t_span: float,
                  sample_every: int,
                  seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate trajectories for one tier of mu values.

    Returns (traj, deriv):
      traj  (n_mu, n_traj, T+1, 2n) subsampled states (float64)
      deriv same shape, analytic (qdot, pdot) on the stored states
    mu_values has shape (n_mu, m); n_mu=1 when m=0.
    """
    g = torch.Generator().manual_seed(seed)
    n = system.dim
    n_mu = max(mu_values.shape[0], 1)
    n_steps = int(round(t_span / dt))
    assert n_steps % sample_every == 0, "t_span/dt must be divisible by sample_every"

    box = torch.tensor(system.phase_space_box(), dtype=torch.float64)
    lo, hi = box[:, 0], box[:, 1]

    # Initial conditions are sampled per mu (keeps RNG order and determinism); the whole
    # tier is integrated as one batch: RK4 and the vector field are elementwise, so batching
    # is bitwise equivalent to a per-mu loop but needs n_steps instead of n_mu*n_steps Python iterations.
    y0 = torch.cat([
        sample_initial_conditions(system, mu_values[i], n_traj, lo, hi, g)
        for i in range(n_mu)])                        # (n_mu·n_traj, 2n)
    mu_rep = mu_values.to(torch.float64).repeat_interleave(n_traj, dim=0)

    f = system_field(system, mu_rep)
    with torch.no_grad():
        sub = rollout_sampled(f, y0, dt, n_steps, sample_every)
    T1 = sub.shape[0]                                 # (T+1, n_mu·n_traj, 2n)
    flat = sub.reshape(T1 * n_mu * n_traj, 2 * n)
    mu_flat = mu_rep.repeat(T1, 1)
    with torch.no_grad():
        qdot, pdot = system.vector_field(flat[:, :n], flat[:, n:], mu_flat)
    deriv = torch.cat([qdot, pdot], dim=-1).reshape(T1, n_mu, n_traj, 2 * n)
    traj = sub.reshape(T1, n_mu, n_traj, 2 * n)
    return traj.permute(1, 2, 0, 3), deriv.permute(1, 2, 0, 3)


def add_state_noise(traj: torch.Tensor, noise_std: float,
                    seed: int) -> torch.Tensor:
    """train/val roles only: Gaussian noise on the states, derivative targets stay exact (same order as HNN)."""
    if noise_std <= 0:
        return traj
    g = torch.Generator().manual_seed(seed)
    return traj + noise_std * torch.randn(traj.shape, generator=g,
                                          dtype=traj.dtype)
