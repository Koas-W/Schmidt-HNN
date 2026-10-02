"""Base class and registry for the test systems.

Conventions (must not be violated):
- state order (q, p); q̇ = ∂H/∂p, ṗ = −∂H/∂q;
- only natural mechanical systems H = ½ pᵀ M⁻¹(q;μ) p + V(q;μ), M symmetric positive definite;
- all systems are nondimensionalized; the parameter family μ is a vector (may be empty);
- potential zero at the equilibrium, V(q_eq;μ) = 0.

All interfaces are batched torch implementations: q, p of shape (B, n), mu of shape (B, mu_dim);
dtype/device follow the input.
"""
from abc import ABC, abstractmethod

import torch


class HamiltonianSystem(ABC):
    """Single test system: analytic H, analytic gradients, equilibrium, default μ ranges, sampling domain."""

    name: str = ""
    dim: int = 0      # degrees of freedom n
    mu_dim: int = 0   # number of dimensionless parameters (0 = parameter-free system)

    @abstractmethod
    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor,
                    mu: torch.Tensor) -> torch.Tensor:
        """H(q,p;μ), returns (B,)."""

    @abstractmethod
    def grad_q(self, q: torch.Tensor, p: torch.Tensor,
               mu: torch.Tensor) -> torch.Tensor:
        """Analytic ∂H/∂q, returns (B, n)."""

    @abstractmethod
    def grad_p(self, q: torch.Tensor, p: torch.Tensor,
               mu: torch.Tensor) -> torch.Tensor:
        """Analytic ∂H/∂p, returns (B, n)."""

    def vector_field(self, q: torch.Tensor, p: torch.Tensor,
                     mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ground-truth Hamiltonian vector field (q̇, ṗ) = (∂H/∂p, −∂H/∂q)."""
        return self.grad_p(q, p, mu), -self.grad_q(q, p, mu)

    @abstractmethod
    def equilibrium(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Equilibrium (q_eq, p_eq), each of shape (n,)."""

    @abstractmethod
    def default_mu_ranges(self) -> list[tuple[float, float]]:
        """Train-tier μ grid ranges [(lo, hi)] × mu_dim (the extrap tier is extended outward by the data pipeline)."""

    @abstractmethod
    def phase_space_box(self) -> list[tuple[float, float]]:
        """Initial-condition sampling box [(lo, hi)] × 2n, ordered (q_1..q_n, p_1..p_n)."""

    def energy_window(self, mu: torch.Tensor) -> tuple[torch.Tensor,
                                                       torch.Tensor] | None:
        """Initial-condition energy window (E_min, E_max), each (B,); None means use the sampling box only.

        Purpose: bounded systems give a μ-dependent energy cap for rejection sampling
        (same practice as AHNN's "energy below the escape threshold"), so that trajectories
        stay forever in the oscillation region / compact phase-space domain. The lower
        bound excludes near-equilibrium zero-signal trajectories.
        """
        return None

    def ic_valid(self, q: torch.Tensor, p: torch.Tensor,
                 mu: torch.Tensor) -> torch.Tensor:
        """Additional initial-condition validity mask (B,) bool; all valid by default.

        Purpose: geometric constraints beyond the energy window. E.g. the Hénon-Heiles
        escape channels open at the saddle points of the potential; energy below the
        saddle height guarantees permanent confinement only if the start point lies inside
        the central well (energy conservation + leaving the well requires crossing a saddle).
        """
        return torch.ones(q.shape[0], dtype=torch.bool, device=q.device)

    def sample_ic(self, mu_row: torch.Tensor, n_traj: int,
                  g: torch.Generator) -> torch.Tensor | None:
        """System-specific initial-condition sampler, returns (n_traj, 2n) float64; None means
        use the generic path (uniform proposal in phase_space_box + energy-window rejection).

        Purpose: in high-dimensional phase space, "uniform proposal then energy rejection" pushes
        samples toward the top of the energy window (energy-shell volume ∝ E^{n-1}), so the
        amplitude distribution loses coverage of the lower half of the window; systems whose
        equilibrium moves with μ also need sampling centered at q_s(μ).
        """
        return None

_REGISTRY: dict[str, type[HamiltonianSystem]] = {}


def register(cls: type[HamiltonianSystem]) -> type[HamiltonianSystem]:
    _REGISTRY[cls.name] = cls
    return cls


def get_system(name: str) -> HamiltonianSystem:
    if name not in _REGISTRY:
        raise KeyError(f"unknown system '{name}', available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_systems() -> list[str]:
    return sorted(_REGISTRY)
