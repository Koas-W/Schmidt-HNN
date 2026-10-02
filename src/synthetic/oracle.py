"""Oracle linear learner.

Observation model (derivative observations):
    y_b = J_ψ(q_b)ᵀ Cᵀ φ(μ_b) + ξ,   ξ ~ N(0, σ²I_n)
Error metric = ||C_hat - C*||_F^2 (orthonormal dictionary => exactly the L2 function error, directly comparable to the EYM floor).

- full-rank ridge regression: matrix-free CG (normal operator via block-wise einsum, never materializing the
  (d_phi d_psi)^2 matrix; configuration B has ~1e5 unknowns and the explicit normal equations would need ~40GB);
- reduced-rank regression (rank=R): ALS, C = A B^T, both sides solved alternately with the same CG machinery;
  R(d_phi or d_psi) unknowns per side, matrix-free matvec only O(B*(d_phi+d_psi+n)*R).

All float64 on CPU (multi-threaded BLAS); data accumulated in chunks, memory O(chunk*d_psi*n).
"""
import torch


def make_observations(family, n_states: int, sigma_noise: float,
                      seed: int, mu_box=None) -> dict:
    """Uniform i.i.d. sampling (the theoretical measure, uniform on the box); returns {phi, J, y}."""
    g = torch.Generator().manual_seed(seed)
    qb = family.psi.box
    mb = torch.tensor(mu_box, dtype=torch.float64) if mu_box is not None \
        else family.phi.box
    q = qb[:, 0] + (qb[:, 1] - qb[:, 0]) * torch.rand(
        n_states, family.dim, generator=g, dtype=torch.float64)
    mu = mb[:, 0] + (mb[:, 1] - mb[:, 0]) * torch.rand(
        n_states, family.mu_dim, generator=g, dtype=torch.float64)
    phi = family.phi.eval(mu)
    _, J = family.psi.eval_grad(q)                  # (B, d_ψ, n)
    y = torch.einsum("bf,fk,bkn->bn", phi, family.C, J)
    y = y + sigma_noise * torch.randn(y.shape, generator=g,
                                      dtype=torch.float64)
    return {"phi": phi, "J": J, "y": y}


def _cg(matvec, rhs: torch.Tensor, lam: float, iters: int = 200,
        tol: float = 1e-10) -> torch.Tensor:
    """Solve (N + lambda I) x = rhs, x has the shape of rhs (matrix-free)."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum()
    rhs_norm = rhs.norm() + 1e-300
    for _ in range(iters):
        Np = matvec(p) + lam * p
        alpha = rs / (p * Np).sum()
        x = x + alpha * p
        r = r - alpha * Np
        rs_new = (r * r).sum()
        if rs_new.sqrt() / rhs_norm < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _normal_full(phi, J, C):
    """Normal operator N(C) = sum_b phi phi^T C J J^T (per-sample structure, matrix-free)."""
    s = torch.einsum("bf,fk->bk", phi, C)
    p = torch.einsum("bk,bkn->bn", s, J)
    r = torch.einsum("bkn,bn->bk", J, p)
    return torch.einsum("bf,bk->fk", phi, r)


def fit_full_rank(obs: dict, lam: float = 1e-8,
                  iters: int = 300) -> torch.Tensor:
    """Full-rank ridge regression C_hat (d_phi, d_psi)."""
    phi, J, y = obs["phi"], obs["J"], obs["y"]
    rhs = torch.einsum("bf,bk->fk", phi, torch.einsum("bkn,bn->bk", J, y))
    return _cg(lambda C: _normal_full(phi, J, C), rhs, lam, iters)


def fit_reduced_rank(obs: dict, R: int, lam: float = 1e-8,
                     sweeps: int = 25, seed: int = 0,
                     inner_iters: int = 120) -> torch.Tensor:
    """Reduced-rank regression C_hat = A B^T, rank <= R (ALS, CG on both sides)."""
    phi, J, y = obs["phi"], obs["J"], obs["y"]
    d_phi, d_psi = phi.shape[1], J.shape[1]
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_phi, R, generator=g, dtype=torch.float64) / d_phi ** 0.5
    Bm = torch.randn(d_psi, R, generator=g, dtype=torch.float64) / d_psi ** 0.5

    for _ in range(sweeps):
        # --- fix B, solve for A: pred = (phi^T A).(B^T J) ------------------------
        Z = torch.einsum("kr,bkn->brn", Bm, J)         # (B,R,n)

        def nv_A(Av):
            s = torch.einsum("bf,fr->br", phi, Av)
            p = torch.einsum("br,brn->bn", s, Z)
            r = torch.einsum("brn,bn->br", Z, p)
            return torch.einsum("bf,br->fr", phi, r)

        rhs_A = torch.einsum("bf,br->fr", phi,
                             torch.einsum("brn,bn->br", Z, y))
        A = _cg(nv_A, rhs_A, lam, inner_iters)

        # --- fix A, solve for B: pred = sum_k B_{k.}.(A^T phi) J_{k.} -------------
        T = torch.einsum("fr,bf->br", A, phi)          # (B,R)

        def nv_B(Bv):
            u = torch.einsum("kr,br->bk", Bv, T)
            p = torch.einsum("bk,bkn->bn", u, J)
            r = torch.einsum("bkn,bn->bk", J, p)
            return torch.einsum("bk,br->kr", r, T)

        rhs_B = torch.einsum("bk,br->kr",
                             torch.einsum("bkn,bn->bk", J, y), T)
        Bm = _cg(nv_B, rhs_B, lam, inner_iters)
    return A @ Bm.T


def excess_risk(C_hat: torch.Tensor, C_true: torch.Tensor) -> float:
    """||C_hat - C*||_F^2 = L2 function error (orthonormal dictionary)."""
    return ((C_hat - C_true) ** 2).sum().item()


def eym_floor(sigma: torch.Tensor, R: int) -> float:
    return (sigma[R:] ** 2).sum().item()
