"""S0: parametrized FPU-alpha-beta chain (anchor example).

    V_plant(q;μ) = u₁(μ)·v₁(q) + u₂(μ)·v₂(q)
    u₁ = μ₁μ₂²μ₃ (total degree 4)   v₁ = Σᵢ (qᵢ₊₁−qᵢ)³
    u₂ = μ₂μ₃μ₄² (total degree 4)   v₂ = Σᵢ (qᵢ₊₁−qᵢ)⁴
    V = ½ qᵀq + ε·V_plant (ε found by the Hessian-margin search)

Note on u₂: with exponent 3 on μ₄ the total degree would be 5, outside the span of the
D_μ=4 dictionary (the d_φ=70 setting), so not every theoretical quantity would be exactly known.
This implementation uses μ₄² (total degree 4, exactly in the dictionary); for the degree-5 form use D_μ=5
(d_φ=126, G changes accordingly).

exact_C: u_k, v_k are projected onto φ, ψ respectively (Gauss-Legendre tensor quadrature, exact for
polynomials), C = Σ_k cu_k cv_kᵀ; planted spectrum = svd(C) (the confining channel is not in the spectrum).
"""
import torch

from .planted import PlantedFamily


def u1(mu: torch.Tensor) -> torch.Tensor:
    return mu[:, 0] * mu[:, 1] ** 2 * mu[:, 2]


def u2(mu: torch.Tensor) -> torch.Tensor:
    return mu[:, 1] * mu[:, 2] * mu[:, 3] ** 2


def v1(q: torch.Tensor) -> torch.Tensor:
    return ((q[:, 1:] - q[:, :-1]) ** 3).sum(-1)


def v2(q: torch.Tensor) -> torch.Tensor:
    return ((q[:, 1:] - q[:, :-1]) ** 4).sum(-1)


DEFAULT_CHANNELS = (((1, 2, 1, 0), 3), ((0, 1, 1, 2), 4))  # canonical S0-A


def make_fpu_family(n: int = 6, m: int = 4, D_mu: int = 4, D_q: int = 4,
                    mu_lo: float = 0.5, mu_hi: float = 1.5, q_a: float = 1.0,
                    extrap_frac: float = 0.3, seed: int = 0,
                    hess_margin: float = 0.25,
                    channels=DEFAULT_CHANNELS) -> PlantedFamily:
    """Build an FPU-type planted family: exact_C projection + eps search (scan box includes the extrapolation shell).

    channels: ((mu exponent tuple a_k, q-difference power e_k), ...), i.e.
    V_plant = sum_k (prod_j mu_j^{a_kj}) * sum_i (q_{i+1} - q_i)^{e_k}.
    Constraint: sum(a_k) <= D_mu and e_k <= D_q (otherwise outside the dictionary span and the factory rejects it).
    """
    for a, e in channels:
        assert len(a) == m and sum(a) <= D_mu and 2 <= e <= D_q, \
            f"channel ({a}, {e}) exceeds the dictionary span (D_mu={D_mu}, D_q={D_q})"
    mu_box = [(mu_lo, mu_hi)] * m
    q_box = [(-q_a, q_a)] * n
    w = (mu_hi - mu_lo) * extrap_frac
    scan_mu_box = [(mu_lo - w, mu_hi + w)] * m

    probe = PlantedFamily(mu_box, q_box, D_mu, D_q,
                          sigma=torch.ones(1), seed=seed, eps=0.0)

    def u_fn(a):
        exp = torch.tensor(a, dtype=torch.float64)
        return lambda mu: torch.prod(mu ** exp, dim=-1)

    def v_fn(e):
        return lambda q: ((q[:, 1:] - q[:, :-1]) ** e).sum(-1)

    cu = torch.stack([probe.phi.project(u_fn(a)) for a, _ in channels])
    cv = torch.stack([probe.psi.project(v_fn(e), n_quad=D_q + 2)
                      for _, e in channels])
    C = cu.T @ cv                                                # exact projection
    return PlantedFamily(mu_box, q_box, D_mu, D_q,
                         sigma=torch.ones(1),  # overwritten by the SVD of C
                         seed=seed, C=C, eps=None, hess_margin=hess_margin,
                         scan_mu_box=scan_mu_box, scan_q_box=q_box)


# Lever configurations (variants of the anchor family)
LEVER_CONFIGS = {
    "S0A": dict(),  # canonical
    "S0Ap": dict(D_mu=5, channels=(((1, 2, 1, 0), 3), ((0, 1, 1, 3), 4))),
    "S0Ap_steep": dict(D_mu=5,
                       channels=(((1, 0, 0, 4), 3), ((0, 2, 3, 0), 4))),
    "S0B": dict(n=8, m=6,
                channels=(((1, 2, 1, 0, 0, 0), 3), ((0, 0, 0, 1, 1, 2), 4))),
    "S3rich": dict(m=8, D_q=5,
                   channels=(((1, 2, 1, 0, 0, 0, 0, 0), 3),
                             ((0, 0, 1, 1, 2, 0, 0, 0), 4),
                             ((0, 0, 0, 0, 0, 1, 1, 2), 5))),
}
