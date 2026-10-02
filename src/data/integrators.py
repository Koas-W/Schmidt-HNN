"""Fixed-step classical RK4 integrator, the only integration scheme in the project.

Ground-truth trajectory generation and the rollout of every evaluated model use
the same fixed-step RK4. Ground truth is generated in float64; the step size of
each system is fixed in its config after a convergence check, with acceptance
criterion ground-truth energy drift <= 1e-8 over the data time span (stricter
than the 1e-6 data-quality criterion of AHNN).

State packing: y = concat(q, p), shape (B, 2n), following the (q, p) state order.
"""
from collections.abc import Callable

import torch

Field = Callable[[torch.Tensor], torch.Tensor]  # y (B,D) -> ẏ (B,D)


def rk4_step(f: Field, y: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(f: Field, y0: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
    """Integrate n_steps from y0; return (n_steps+1, B, D) including the initial state."""
    ys = [y0]
    y = y0
    for _ in range(n_steps):
        y = rk4_step(f, y, dt)
        ys.append(y)
    return torch.stack(ys, dim=0)


def rollout_sampled(f: Field, y0: torch.Tensor, dt: float, n_steps: int,
                    sample_every: int) -> torch.Tensor:
    """Same as rollout but keeps only every sample_every-th state (saves memory in long-horizon evaluation).

    Returns (n_steps//sample_every + 1, B, D), including the initial state.
    """
    assert n_steps % sample_every == 0
    ys = [y0]
    y = y0
    for i in range(1, n_steps + 1):
        y = rk4_step(f, y, dt)
        if i % sample_every == 0:
            ys.append(y)
    return torch.stack(ys, dim=0)


def system_field(system, mu: torch.Tensor) -> Field:
    """Wrap the ground-truth vector field of a HamiltonianSystem as f(y). mu has shape (B, mu_dim)."""
    n = system.dim

    def f(y: torch.Tensor) -> torch.Tensor:
        q, p = y[:, :n], y[:, n:]
        qdot, pdot = system.vector_field(q, p, mu)
        return torch.cat([qdot, pdot], dim=-1)

    return f


def energy_drift(system, traj: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """|H(t)-H(0)| / max(|H(0)|, eps); traj (T,B,2n) -> (T,B)."""
    n = system.dim
    T, B, _ = traj.shape
    flat = traj.reshape(T * B, 2 * n)
    H = system.hamiltonian(flat[:, :n], flat[:, n:],
                           mu.repeat(T, 1)).reshape(T, B)
    H0 = H[0]
    return (H - H0).abs() / H0.abs().clamp_min(1e-12)


def rollout_sampled_graphed(f: Field, y0: torch.Tensor, dt: float, n_steps: int,
                            sample_every: int, warmup: int = 3) -> torch.Tensor:
    """CUDA Graph version of rollout_sampled: captures one full RK4 step (4 field evaluations + update)
    as a graph and replays it step by step, removing the kernel-launch overhead of small-batch rollouts.

    Bitwise equivalent to rollout_sampled (same expressions, same order); falls back to eager when y0 is not on CUDA.
    f must keep shapes/devices fixed during capture and avoid CPU syncs (autograd.grad inside an HNN-style time_derivative is capturable).
    """
    if y0.device.type != "cuda":
        return rollout_sampled(f, y0, dt, n_steps, sample_every)
    assert n_steps % sample_every == 0
    y_s = y0.detach().clone()                       # static state buffer (the graph advances it in place)
    y_in = y_s.clone()                              # warm-up copy, keeps y_s unpolluted

    def body():
        y_new = rk4_step(f, y_s, dt)
        y_s.copy_(y_new)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    y_s.copy_(y_in)                                 # back to the initial state
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    y_s.copy_(y_in)                                 # capture itself advanced one step; back to the initial state again
    ys = [y_in.clone()]
    for i in range(1, n_steps + 1):
        graph.replay()
        if i % sample_every == 0:
            ys.append(y_s.clone())
    torch.cuda.synchronize()
    return torch.stack(ys, dim=0)
