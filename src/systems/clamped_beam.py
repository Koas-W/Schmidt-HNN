"""Modal Hamiltonian system of a clamped-clamped square beam with von Karman geometric nonlinearity.

Fundamental difference from the other systems in this project: the ground-truth potential is
not a finite-degree polynomial in q. The full model keeps N = n_master + n_slave modes; the
learning problem only exposes the first n_master (master modes), the rest are eliminated by
static condensation:

    V_eff(a; μ) = min_b V_full(a, b; μ)

b*(a;μ) is defined implicitly by a cubic system of equations, so V_eff is an algebraic function
of (a,μ) whose (μ,a) rank is not finite; it only has a decaying spectrum. The condensed
potential therefore has no closed form and is not exactly representable by the network.

Parameters (three, all on the stiffness/load side, so the mass matrix is the identity for all μ):
    μ₁ = log10(k̃)         axial support stiffness; enters the membrane coefficient g = k̃/(1+k̃) (rational function)
    μ₂ = Ñ₀/(4π²)         axial preload / buckling load of the clamped column
    μ₃ = w_s^lin(0.5)/h   midpoint value of the linear static deflection (proportional to the dimensionless pressure p̃)

Dimensionless potential:
    V_full(q;μ) = ½Σβᵢ⁴qᵢ² + (Ñ₀/2)qᵀGq + (g/8)(qᵀGq)² − p̃ fᵀq
"""
import math

import torch

from .base import HamiltonianSystem, register

N_CR = 4.0 * math.pi ** 2      # buckling load of the clamped column (dimensionless), scale of μ₂
SQRT12 = math.sqrt(12.0)       # r = h/√12: converts modal amplitude to multiples of the thickness


# --------------------------------------------------------------------------
# Modal basis (numerically stable form)
# --------------------------------------------------------------------------
def beta_roots(n: int) -> torch.Tensor:
    """First n positive roots of cos β cosh β = 1 by Newton iteration (float64).

    Initial guess is the asymptotic value (2i+1)π/2; a log-form residual avoids cosh overflow:
        F(β) = log|cos β| + β − log 2 = 0  is equivalent to cos β cosh β = 1 (for large β)
    Small roots use a high-precision initial guess + Newton on the original equation.
    """
    betas = []
    for i in range(1, n + 1):
        b = (2 * i + 1) * math.pi / 2.0
        for _ in range(100):
            # f(β) = cos β · cosh β − 1, scaled with cosh β ≈ e^β/2 to avoid overflow:
            # define F = cos β − 2 e^{−β} / (1 + e^{−2β})  (= cos β − 1/cosh β)
            e = math.exp(-b)
            sech = 2.0 * e / (1.0 + e * e)
            f = math.cos(b) - sech
            df = -math.sin(b) + sech * math.tanh(b)
            step = f / df
            b -= step
            if abs(step) < 1e-15:
                break
        betas.append(b)
    return torch.tensor(betas, dtype=torch.float64)


def _sigma_and_one_minus(beta: torch.Tensor):
    """Returns (σ, 1−σ), both computed without cancellation."""
    sinh_b = torch.sinh(beta)
    den = sinh_b - torch.sin(beta)
    one_minus = (torch.cos(beta) - torch.sin(beta) - torch.exp(-beta)) / den
    return 1.0 - one_minus, one_minus


def mode_shapes(beta: torch.Tensor, x: torch.Tensor, order: int = 0):
    """Clamped-clamped mode shapes φ (order=0) / φ' (1) / φ'' (2), shape (n_modes, n_x).

    Numerically stable form: φ = ½(1−σ)e^{βX} + ½(1+σ)e^{−βX} − cos βX + σ sin βX,
    where the small quantity (1−σ) comes exactly from _sigma_and_one_minus, so ½(1−σ)e^{βX} stays bounded.
    """
    sigma, one_minus = _sigma_and_one_minus(beta)
    b = beta.unsqueeze(1)                      # (n,1)
    s = sigma.unsqueeze(1)
    om = one_minus.unsqueeze(1)
    bx = b * x.unsqueeze(0)                    # (n,n_x)
    ep, em = torch.exp(bx), torch.exp(-bx)
    c, sn = torch.cos(bx), torch.sin(bx)
    if order == 0:
        return 0.5 * om * ep + 0.5 * (1 + s) * em - c + s * sn
    if order == 1:
        return b * (0.5 * om * ep - 0.5 * (1 + s) * em + sn + s * c)
    if order == 2:
        return b ** 2 * (0.5 * om * ep + 0.5 * (1 + s) * em + c - s * sn)
    raise ValueError("order ∈ {0,1,2}")


def gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n-point Gauss-Legendre nodes and weights on [0,1] (float64)."""
    import numpy as np
    xs, ws = np.polynomial.legendre.leggauss(n)
    return (torch.tensor(0.5 * (xs + 1.0), dtype=torch.float64),
            torch.tensor(0.5 * ws, dtype=torch.float64))


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------
class ClampedBeamBase(HamiltonianSystem):
    """n_master master modes (visible to the network), n_slave slave modes (eliminated by static condensation)."""

    name = ""
    mu_dim = 3

    N_QUAD = 1024          # Gauss-Legendre node count (highest wavenumber of the integrand ~2β_N)
    N_XGRID = 401          # uniform X grid for the peak deflection
    NEWTON_RTOL = 1e-14    # relative residual criterion |r_i| / (β_i⁴(1+|q_i|)); the force scale ~β⁴ reaches 1e5
    NEWTON_MAX = 60
    AMP_LO, AMP_HI = 1.0, 3.0   # total peak deflection window |w|_peak / h (absolute, includes the static deflection)
    DYN_MIN = 0.5               # minimum dynamic margin (h): total peak target ≥ static peak + DYN_MIN
    # short-window integration for initial-condition scaling (sample_ic stage two): bisect the scale factor on the realized peak
    IC_SEARCH_DT, IC_SEARCH_T, IC_SEARCH_EVERY = 2e-3, 0.5, 5   # scaling only; ω₅·dt=0.6 lies inside the RK4 stability region
    IC_TARGET_MARGIN = 0.1      # target cap AMP_HI − margin: leaves room for later peak growth due to beating
    # modal energy spectrum of the initial direction: share w_i ∝ ω_i^{−α}·exp(σξ_i), ξ~N(0,1); random phases.
    # for α=2.5 the expected shares are mode 1≈92%, mode 2≈7%, mode 5≈1.4e-3 (self-consistent truncation: highest master mode ≲1e-3).
    MODAL_ALPHA, MODAL_JITTER = 2.5, 0.5

    def __init__(self, n_master: int = 4, n_slave: int | None = None):
        self.n_master = n_master
        self.n_slave = n_master if n_slave is None else n_slave
        self.n_total = self.n_master + self.n_slave
        self.dim = self.n_master
        beta = beta_roots(self.n_total)
        self.beta = beta
        self.k_diag = beta ** 4                                   # (N,)
        xq, wq = gauss_legendre(self.N_QUAD)
        phi = mode_shapes(beta, xq, 0)
        dphi = mode_shapes(beta, xq, 1)
        self.G = torch.einsum("ik,jk,k->ij", dphi, dphi, wq)      # (N,N)
        self.G = 0.5 * (self.G + self.G.T)
        self.f = torch.einsum("ik,k->i", phi, wq)                 # (N,)
        self.norm = torch.einsum("ik,jk,k->ij", phi, phi, wq)     # for orthogonality checks
        xg = torch.linspace(0.0, 1.0, self.N_XGRID, dtype=torch.float64)
        self.phi_grid = mode_shapes(beta, xg, 0)                  # (N, n_x)
        # midpoint value of the linear static deflection (per unit p̃): q_lin = p̃ f / β⁴
        mid = mode_shapes(beta, torch.tensor([0.5], dtype=torch.float64), 0)[:, 0]
        self._w_mid_per_p = float((self.f / self.k_diag * mid).sum() / SQRT12)

    # -------- parameter decoding ----------------------------------------
    def decode_mu(self, mu: torch.Tensor):
        """μ -> (g, Ñ₀, p̃), each (B,)."""
        mu = mu.to(torch.float64)
        k_tilde = torch.pow(10.0, mu[:, 0])
        g = k_tilde / (1.0 + k_tilde)
        n0 = mu[:, 1] * N_CR
        p = mu[:, 2] / self._w_mid_per_p
        return g, n0, p

    # -------- full-model potential, gradient, Hessian -------------------
    def _consts(self, q: torch.Tensor):
        dev, dt = q.device, q.dtype
        return (self.k_diag.to(dev, dt), self.G.to(dev, dt), self.f.to(dev, dt))

    def _rel_residual(self, r: torch.Tensor, q_part: torch.Tensor,
                      k_part: torch.Tensor) -> float:
        """Maximum residual normalized by β⁴(1+|q|) (absolute residuals are meaningless for high modes with β⁴~1e5)."""
        return float((r.detach().abs() / (k_part * (1.0 + q_part.detach().abs()))).max())

    def V_full(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        k, G, f = self._consts(q)
        g, n0, p = self.decode_mu(mu)
        g, n0, p = g.to(q.dtype), n0.to(q.dtype), p.to(q.dtype)
        s = torch.einsum("bi,ij,bj->b", q, G, q)
        return (0.5 * (k * q ** 2).sum(-1) + 0.5 * n0 * s
                + g / 8.0 * s ** 2 - p * (q @ f))

    def grad_V_full(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        k, G, f = self._consts(q)
        g, n0, p = self.decode_mu(mu)
        g, n0, p = g.to(q.dtype), n0.to(q.dtype), p.to(q.dtype)
        Gq = q @ G
        s = (q * Gq).sum(-1)
        return (k * q + n0.unsqueeze(-1) * Gq
                + (0.5 * g * s).unsqueeze(-1) * Gq - p.unsqueeze(-1) * f)

    def hess_V_full(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        k, G, _ = self._consts(q)
        g, n0, _ = self.decode_mu(mu)
        g, n0 = g.to(q.dtype), n0.to(q.dtype)
        Gq = q @ G
        s = (q * Gq).sum(-1)
        H = (torch.diag(k) + (n0 + 0.5 * g * s).reshape(-1, 1, 1) * G
             + g.reshape(-1, 1, 1) * Gq.unsqueeze(2) * Gq.unsqueeze(1))
        return H

    # -------- static condensation ---------------------------------------
    def condense(self, a: torch.Tensor, mu: torch.Tensor,
                 b0: torch.Tensor | None = None) -> torch.Tensor:
        """Batched Newton solve for b*(a;μ), returns (B, n_slave). b0 is a warm-start initial guess."""
        n, m = self.n_master, self.n_slave
        if b0 is None:
            cache = getattr(self, "_b_cache", None)
            if (cache is not None and cache.shape == (a.shape[0], m)
                    and cache.device == a.device and cache.dtype == a.dtype):
                b0 = cache      # warm start: a wrong guess only costs a few extra iterations; the residual criterion guarantees correctness
        b = torch.zeros(a.shape[0], m, dtype=a.dtype,
                        device=a.device) if b0 is None else b0.clone()
        k_s = self.k_diag[n:].to(a.device, a.dtype)
        for _ in range(self.NEWTON_MAX):
            q = torch.cat([a, b], dim=-1)
            r = self.grad_V_full(q, mu)[:, n:]
            if self._rel_residual(r, b, k_s) < self.NEWTON_RTOL:
                break
            J = self.hess_V_full(q, mu)[:, n:, n:]
            b = b - torch.linalg.solve(J, r.unsqueeze(-1)).squeeze(-1)
        self._b_cache = b.detach()
        return b

    def V_eff(self, a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        b = self.condense(a, mu)
        return self.V_full(torch.cat([a, b], dim=-1), mu)

    # -------- HamiltonianSystem interface -------------------------------
    def hamiltonian(self, q, p, mu):
        """H = ½|p|² + V_eff(q;μ) − V_eff(q_s(μ);μ).

        The potential zero is placed at the μ-dependent static equilibrium (convention
        V(q_eq;μ)=0), so H is the energy above static equilibrium: the normalized energy
        drift |H(t)−H(0)|/|H(0)| is well defined (otherwise H(0) may cross zero under large
        static loads). The constant offset does not affect the vector field; grad_q omits it.
        """
        q_s = self.static_equilibrium(mu).to(q.dtype)
        return 0.5 * (p ** 2).sum(-1) + self.V_eff(q, mu) - self.V_eff(q_s, mu)

    def grad_p(self, q, p, mu):
        return p

    def grad_q(self, q, p, mu):
        """Envelope theorem: ∂V_eff/∂a = ∂V_full/∂a |_{b=b*}."""
        b = self.condense(q, mu)
        return self.grad_V_full(torch.cat([q, b], dim=-1), mu)[:, :self.n_master]

    def equilibrium(self):
        """Reference equilibrium: static equilibrium at the midpoint μ of default_mu_ranges (where H=0).

        The base-class signature has no μ; the true equilibrium of this system moves with the
        static pressure, see static_equilibrium(mu) for the general case.
        """
        mr = torch.tensor(self.default_mu_ranges(), dtype=torch.float64)
        q_s = self.static_equilibrium(mr.mean(dim=1).unsqueeze(0))[0]
        return q_s, torch.zeros_like(q_s)

    def static_equilibrium(self, mu: torch.Tensor) -> torch.Tensor:
        """Master-mode components of the loaded equilibrium q_s(μ), (B, n_master).

        Solves ∂V_full/∂q = 0 for the full model; by the envelope theorem its master-mode
        components are the stationary point of V_eff.
        """
        B = mu.shape[0]
        q = torch.zeros(B, self.n_total, dtype=torch.float64, device=mu.device)
        k_all = self.k_diag.to(mu.device, torch.float64)
        for _ in range(self.NEWTON_MAX):
            r = self.grad_V_full(q, mu)
            if self._rel_residual(r, q, k_all) < self.NEWTON_RTOL:
                break
            H = self.hess_V_full(q, mu)
            q = q - torch.linalg.solve(H, r.unsqueeze(-1)).squeeze(-1)
        return q[:, :self.n_master]

    def default_mu_ranges(self):
        # μ₃ cap 0.5: static peak deflection (incl. slave modes) ≤0.90h over the training box and ≤2.09h incl. the 30% extrap shell,
        # leaving ≥0.9h dynamic margin at the worst extrap corner under the 3h absolute amplitude cap
        return [(-1.0, 1.0), (-0.5, 0.5), (0.2, 0.5)]

    # -------- amplitude <-> energy (energy window) ----------------------
    def peak_amplitude(self, a: torch.Tensor) -> torch.Tensor:
        """Peak deflection max_X |w|/h for given master-mode amplitudes, (B,).

        The slave-mode contribution to the deflection is ignored in the amplitude criterion
        (suppressed by β⁻⁴); the realized peak of full trajectories is checked separately.
        """
        pg = self.phi_grid[:self.n_master].to(a.device, a.dtype)
        return (a @ pg).abs().max(dim=-1).values / SQRT12

    def deflection(self, q_full: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """All N modal amplitudes (B,N) -> deflection w/h at positions x (n_x,), (B,n_x)."""
        phi = mode_shapes(self.beta, x.to(torch.float64), 0).to(q_full.device, q_full.dtype)
        return (q_full @ phi) / SQRT12

    def peak_deflection(self, a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Master amplitudes (B,n) -> peak deflection max_X|w|/h including the slave modes b*(a), (B,)."""
        b = self.condense(a, mu)
        pg = self.phi_grid.to(a.device, a.dtype)
        return (torch.cat([a, b], -1) @ pg).abs().max(dim=-1).values / SQRT12

    def amplitude_bounds(self, mu: torch.Tensor):
        """Usable total-peak-deflection interval [lo_eff, AMP_HI] per μ, lo_eff = max(AMP_LO, static + DYN_MIN).
        Returns (lo_eff, static, a_s). Raises if static + DYN_MIN > AMP_HI (misconfigured parameter box)."""
        a_s = self.static_equilibrium(mu)
        static = self.peak_deflection(a_s, mu)
        lo_eff = torch.maximum(torch.full_like(static, self.AMP_LO), static + self.DYN_MIN)
        if bool((lo_eff > self.AMP_HI).any()):
            i = int((lo_eff - self.AMP_HI).argmax())
            raise ValueError(f"static peak deflection {static[i]:.2f}h + dynamic margin {self.DYN_MIN}h exceeds "
                             f"the cap {self.AMP_HI}h: μ={mu[i].tolist()}; tighten the parameter box")
        return lo_eff, static, a_s

    def peak_coeff(self, mu: torch.Tensor, target):
        """Displace c·e₁ along the first mode from the static equilibrium a_s so that the peak deflection = target·h (scalar or (B,)). Returns (c, a_s)."""
        a_s = self.static_equilibrium(mu)
        # pressure p̃>0 and f₁>0 imply a_s[:,0]>0; displacing along +e₁ increases |w|_max monotonically (premise of the bisection)
        assert bool((a_s[:, 0] > 0).all()), "first static-equilibrium component must be positive (p̃>0)"
        e1 = torch.zeros_like(a_s)
        e1[:, 0] = 1.0
        lo = torch.zeros(mu.shape[0], dtype=torch.float64, device=mu.device)
        hi = torch.full_like(lo, 1.0)
        for _ in range(60):                      # grow the upper bound until it exceeds target (scalar or (B,))
            if bool((self.peak_deflection(a_s + hi.unsqueeze(-1) * e1, mu) > target).all()):
                break
            hi = hi * 2.0
        for _ in range(80):                      # bisection
            mid = 0.5 * (lo + hi)
            big = self.peak_deflection(a_s + mid.unsqueeze(-1) * e1, mu) > target
            lo = torch.where(big, lo, mid)
            hi = torch.where(big, mid, hi)
        return 0.5 * (lo + hi), a_s

    def _energy_at_peak(self, mu: torch.Tensor, target: float) -> torch.Tensor:
        """Potential energy (relative to static equilibrium) at peak deflection = target·h along the first mode, (B,)."""
        c, a_s = self.peak_coeff(mu, target)
        e1 = torch.zeros_like(a_s)
        e1[:, 0] = 1.0
        return self.V_eff(a_s + c.unsqueeze(-1) * e1, mu) - self.V_eff(a_s, mu)

    def energy_window(self, mu):
        """Initial-condition energy window (relative to static equilibrium, same zero as hamiltonian):
        the energy interval whose peak deflection lies in [AMP_LO, AMP_HI]·h."""
        lo_eff, _, _ = self.amplitude_bounds(mu)
        return self._energy_at_peak(mu, lo_eff), self._energy_at_peak(mu, self.AMP_HI)

    def sample_ic(self, mu_row, n_traj, g):
        """Two-stage initial-condition sampling:
        target total peak deflection A stratified-uniform on [max(AMP_LO, static+DYN_MIN), AMP_HI−margin];
        direction d=(d_q, 0): static release, zero initial velocity; modal energy shares w_i ∝ ω_i^{−α}·exp(σξ_i)
        (power law + jitter, MODAL_ALPHA/JITTER), d_q,i=±√w_i/ω_i, first component with the sign of the
        static deflection, the other signs random.
        Stage one: bisect c along the ray so that H = E(A) (energy of the first-mode turning point;
        V_eff is convex near q_s, hence monotone).
        Stage two: a random direction spreads energy over several modes and dilutes the first-mode
        amplitude (50-85% in practice), so c is bisected again on the realized total peak of a
        short-window (IC_SEARCH_T, ≥4 hardening periods) RK4 integration.
        Antisymmetric modes are excited naturally by the random direction.
        """
        n = self.n_master
        mu = mu_row.reshape(1, -1).to(torch.float64).expand(n_traj, -1)
        lo_eff, _, _ = self.amplitude_bounds(mu)
        hi_t = self.AMP_HI - self.IC_TARGET_MARGIN
        # stratified (jittered grid) rather than iid: uniform coverage of the target total peak over [lo_eff, hi_t]
        u = (torch.randperm(n_traj, generator=g).to(torch.float64)
             + torch.rand(n_traj, generator=g, dtype=torch.float64)) / n_traj
        lo_t = lo_eff.clamp(max=hi_t)            # lo_eff may already exceed the target cap (extreme extrap corner)
        A = lo_t + (hi_t - lo_t) * u
        c_pk, a_s = self.peak_coeff(mu, A)
        e1 = torch.zeros_like(a_s)
        e1[:, 0] = 1.0
        v_s = self.V_eff(a_s, mu)
        E = self.V_eff(a_s + c_pk.unsqueeze(-1) * e1, mu) - v_s
        # direction: static release (p=0); initial shape = static equilibrium + modal displacement with power-law spectrum, random signs,
        # first component with the sign of the static deflection (peak monotone in the scale factor, bisection reliable; the trajectory then crosses static equilibrium and covers the other side).
        # modal energy share w_i ∝ ω_i^{−α}·exp(σξ_i); under static release q_i = ±√(2e_i)/ω_i (linear approximation).
        omega = self.k_diag[:n].sqrt()                                   # ω_i = β_i²
        logw = -self.MODAL_ALPHA * torch.log(omega).unsqueeze(0) \
            + self.MODAL_JITTER * torch.randn(n_traj, n, generator=g, dtype=torch.float64)
        w = torch.softmax(logw, dim=-1)                                  # Σ_i w_i = 1
        sign = torch.where(torch.rand(n_traj, n, generator=g, dtype=torch.float64) < 0.5, -1.0, 1.0)
        sign[:, 0] = torch.sign(a_s[:, 0])
        dq = sign * torch.sqrt(w) / omega
        dp = torch.zeros_like(dq)

        def h_of(c):
            q = a_s + c.unsqueeze(-1) * dq
            return 0.5 * ((c.unsqueeze(-1) * dp) ** 2).sum(-1) + self.V_eff(q, mu) - v_s

        lo = torch.zeros(n_traj, dtype=torch.float64)
        hi = torch.ones(n_traj, dtype=torch.float64)
        for _ in range(60):
            over = h_of(hi) > E
            if bool(over.all()):
                break
            hi = torch.where(over, hi, hi * 2.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            big = h_of(mid) > E
            lo = torch.where(big, lo, mid)
            hi = torch.where(big, mid, hi)
        c0 = 0.5 * (lo + hi)                     # energy-matched solution (initial guess for stage two)

        # ---- stage two: scale c on the realized short-window total peak (corrects the first-mode dilution by the random direction) ----
        def realized_peak(c):
            q = a_s + c.unsqueeze(-1) * dq
            p = c.unsqueeze(-1) * dp
            return self._short_window_peak(q, p, mu)

        lo, hi = 0.6 * c0, 1.2 * c0
        for _ in range(12):                      # bracket: peak at lo < A < peak at hi
            pk_lo, pk_hi = realized_peak(lo), realized_peak(hi)
            need_lo, need_hi = pk_lo > A, pk_hi < A
            if not bool(need_lo.any() or need_hi.any()):
                break
            hi = torch.where(need_hi, hi * 1.5, hi)
            lo = torch.where(need_lo, lo * 0.6, lo)
        for _ in range(6):                       # bisection: relative precision 2⁻⁶ (1.6%, sufficient for peak scaling)
            mid = 0.5 * (lo + hi)
            big = realized_peak(mid) > A
            lo = torch.where(big, lo, mid)
            hi = torch.where(big, mid, hi)
        c = 0.5 * (lo + hi)
        return torch.cat([a_s + c.unsqueeze(-1) * dq, c.unsqueeze(-1) * dp], dim=-1)

    def _short_window_peak(self, q, p, mu):
        """RK4-integrate from (q,p) over IC_SEARCH_T; returns the total peak deflection max|w|/h within the window (incl. slave modes), (B,)."""
        dt, n_steps = self.IC_SEARCH_DT, int(round(self.IC_SEARCH_T / self.IC_SEARCH_DT))

        def f(qq, pp):
            return pp, -self.grad_q(qq, pp, mu)

        peak = self.peak_deflection(q, mu)
        for k in range(1, n_steps + 1):
            k1q, k1p = f(q, p)
            k2q, k2p = f(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
            k3q, k3p = f(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
            k4q, k4p = f(q + dt * k3q, p + dt * k3p)
            q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            if k % self.IC_SEARCH_EVERY == 0:
                peak = torch.maximum(peak, self.peak_deflection(q, mu))
        return peak

    def phase_space_box(self):
        """Proposal box for (q₁..qₙ, p₁..pₙ); the actual selection is done by energy-window rejection sampling."""
        pg = self.phi_grid[:self.n_master]
        qmax = (self.AMP_HI * SQRT12 / pg.abs().max(dim=-1).values * 1.5)
        pmax = float(self.beta[0] ** 2 * qmax[0])
        return ([(-float(v), float(v)) for v in qmax]
                + [(-pmax, pmax)] * self.n_master)


N_MASTER_DEFAULT = 5   # smallest n passing the modal convergence test


@register
class ClampedBeamDefault(ClampedBeamBase):
    """Final configuration: 5 master modes + 5 slave modes (N=2n=10).

    Convergence test (slave modes doubled, A/h=3, 8 corners + center of the parameter box): at n=5
    the static deflection changes by 5.4e-5 and the backbone frequency ω₁ by 5.2e-4, both <1e-3,
    the smallest passing value; Euler-Bernoulli fidelity ε₅=2.05% (L/h=50).
    """
    name = "clamped_beam"

    def __init__(self):
        super().__init__(n_master=N_MASTER_DEFAULT, n_slave=N_MASTER_DEFAULT)
