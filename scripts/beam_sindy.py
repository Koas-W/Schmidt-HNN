#!/usr/bin/env python3
"""Sparse-regression (SINDy-type) classical baseline on the beam data: same grid data, same evaluation tiers, same vector-field error definition.

Two variants (both linear least squares in the coefficients + sequential thresholded least squares STLSQ, Brunton et al. 2016):
  potential: V(q;μ) = Σ c_{αβ} q^α μ^β, regress ṗ = −∇_q V (the five components share coefficients; conservative structure built in);
  plain    : ṗ_j = Σ c_{jαβ} q^α μ^β, each of the five components regressed independently (standard SINDy, conservation not guaranteed).
Dictionary: monomials in q (degree <= dq) x monomials in μ (degree <= dmu). dq=4 with finite dmu is a "near-oracle" dictionary
(V_full is a quartic polynomial in q, but g(μ₁) is a rational function and V_eff is not polynomial because of static condensation).
The sparsity threshold λ is selected on the validation set (the same val-role trajectories as the networks). Everything runs on CPU by chunked accumulation of the normal equations.

Usage: python3 scripts/beam_sindy.py [--dq 4 6] [--dmu 2 3] [--threads 12]
Output: results/beam_sindy/sindy_beam.csv + console table
"""
import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import eval_views, load_dataset, training_views  # noqa: E402

LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]   # relative threshold grid: entries with |ξ_norm| < λ·max|ξ_norm| are zeroed (0 = plain least squares)
RIDGE = 1e-10


def exponents(n_var, max_deg, min_deg=0):
    out = []
    for deg in range(min_deg, max_deg + 1):
        for c in itertools.combinations_with_replacement(range(n_var), deg):
            e = [0] * n_var
            for i in c:
                e[i] += 1
            out.append(tuple(e))
    return out


def exact_mu_basis(mu):
    """Exact μ features for the solver self-check: the μ dependence of V_full is exactly a linear combination of 1, g(μ₁), Ñ₀(μ₂), p̃(μ₃)."""
    from src.systems import get_system
    g, N0, p = get_system("clamped_beam").decode_mu(mu)
    return torch.stack([torch.ones_like(g), g, N0, p], dim=1)


def monomials(X, exps):
    """X: (N, n_var) → (N, len(exps))"""
    cols = [torch.prod(X ** torch.tensor(e, dtype=X.dtype), dim=1) for e in exps]
    return torch.stack(cols, dim=1)


def features_potential(q, mu, q_exps, mu_exps):
    """Return (N, n, K): features of the j-th component equation, −ψ_β(μ)·∂_j q^α."""
    N, n = q.shape
    psi = exact_mu_basis(mu) if mu_exps == "exact" else monomials(mu, mu_exps)   # (N, M)
    cols = []
    for a in q_exps:
        grad = torch.zeros(N, n, dtype=q.dtype)
        for j in range(n):
            if a[j] > 0:
                e = list(a); e[j] -= 1
                grad[:, j] = -a[j] * torch.prod(q ** torch.tensor(e, dtype=q.dtype), dim=1)
        cols.append(grad.unsqueeze(-1) * psi.unsqueeze(1))   # (N, n, M)
    return torch.cat(cols, dim=-1)                      # (N, n, Q·M)


def features_plain(q, mu, q_exps, mu_exps):
    phi = monomials(q, q_exps); psi = monomials(mu, mu_exps)
    return (phi.unsqueeze(-1) * psi.unsqueeze(1)).reshape(q.shape[0], -1)   # (N, Q·M)


def accumulate(view, feat_fn, chunk, n):
    """Normal equations A=FᵀF, B=FᵀY (potential: Y is ṗ flattened; plain: Y is the (N, n) multi-target)."""
    x, mu, dx = view["x"].double(), view["mu"].double(), view["dx"].double()
    A = B = None
    for i in range(0, x.shape[0], chunk):
        q = x[i:i + chunk, :n]; m = mu[i:i + chunk]; y = dx[i:i + chunk, n:]
        F = feat_fn(q, m)
        if F.dim() == 3:                       # potential: (c, n, K) -> flatten rows
            F = F.reshape(-1, F.shape[-1]); y = y.reshape(-1, 1)
        A = F.T @ F if A is None else A + F.T @ F
        B = F.T @ y if B is None else B + F.T @ y
    return A, B


def stlsq(A, B, lam, iters=10):
    """Sequential thresholded least squares on RMS-normalized columns; returns coefficients (K, T) in the original scale."""
    s = A.diagonal().clamp_min(1e-300).sqrt()
    An = A / s[:, None] / s[None, :] + RIDGE * torch.eye(A.shape[0], dtype=A.dtype)
    Bn = B / s[:, None]
    K, T = An.shape[0], B.shape[1]
    xi = torch.zeros(K, T, dtype=A.dtype)
    for t in range(T):
        active = torch.ones(K, dtype=torch.bool)
        for _ in range(iters):
            idx = active.nonzero().squeeze(-1)
            sol = torch.linalg.solve(An[idx][:, idx], Bn[idx, t])
            new_active = active.clone(); new_active[idx] = sol.abs() >= lam * sol.abs().max()
            xi_t = torch.zeros(K, dtype=A.dtype); xi_t[idx] = sol
            if torch.equal(new_active, active):
                break
            active = new_active
            if active.sum() == 0:
                xi_t = torch.zeros(K, dtype=A.dtype); break
        xi[:, t] = xi_t
    return xi / s[:, None]


def predict_mse(view, feat_fn, xi, n, chunk):
    """Same definition as the networks: mean square over all 2n components (the q̇ components equal p exactly, error 0)."""
    x, mu, dx = view["x"].double(), view["mu"].double(), view["dx"].double()
    se = 0.0
    for i in range(0, x.shape[0], chunk):
        q = x[i:i + chunk, :n]; m = mu[i:i + chunk]; y = dx[i:i + chunk, n:]
        F = feat_fn(q, m)
        pred = (F @ xi[:, 0]) if F.dim() == 3 else (F @ xi)
        se += ((pred - y) ** 2).sum().item()
    return se / (x.shape[0] * 2 * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dq", type=int, nargs="*", default=[4, 6])
    ap.add_argument("--dmu", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--variants", nargs="*", default=["potential", "plain"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--mu-basis", default="poly", choices=["poly", "exact"],
                    help="exact = solver self-check (exact μ features, dmu ignored)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    data, man = load_dataset(Path("data_cache/clamped_beam"))
    tv = training_views(data, man); ev = eval_views(data, man)
    n = tv["train"]["x"].shape[1] // 2
    views = {"train": ev["train"], "interp": ev["interp"], "extrap": ev["extrap"]}
    out = Path("results/beam_sindy"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"train samples {tv['train']['x'].shape[0]}, val samples {tv['val']['x'].shape[0]}, threads {args.threads}")
    for variant, dq, dmu in itertools.product(args.variants, args.dq, args.dmu):
        q_exps = exponents(n, dq, min_deg=1 if variant == "potential" else 0)
        mu_exps = "exact" if args.mu_basis == "exact" else exponents(3, dmu)
        feat = (lambda q, m: features_potential(q, m, q_exps, mu_exps)) if variant == "potential" \
            else (lambda q, m: features_plain(q, m, q_exps, mu_exps))
        K = len(q_exps) * (4 if mu_exps == "exact" else len(mu_exps))
        t0 = time.time()
        A, B = accumulate(tv["train"], feat, args.chunk, n)
        t_acc = time.time() - t0
        best = None
        for lam in LAMBDAS:
            t1 = time.time()
            xi = stlsq(A, B, lam)
            val = predict_mse(tv["val"], feat, xi, n, args.chunk)
            nnz = int((xi != 0).sum())
            print(f"  {variant:9s} dq={dq} dmu={dmu if mu_exps != 'exact' else 'exact'} K={K:5d} λ={lam:<6g} nnz={nnz:5d} val={val:.4g} ({time.time()-t1:.0f}s)", flush=True)
            if best is None or val < best[1]:
                best = (lam, val, xi, nnz)
        lam, val, xi, nnz = best
        res = {tier: predict_mse(v, feat, xi, n, args.chunk) for tier, v in views.items()}
        print(f"* {variant:9s} dq={dq} dmu={dmu} K={K} selected λ={lam:g} nnz={nnz} | vf_mse train={res['train']:.4g} "
              f"interp={res['interp']:.4g} extrap={res['extrap']:.4g} | accumulate {t_acc:.0f}s total {time.time()-t0:.0f}s", flush=True)
        for tier, v in res.items():
            rows.append({"variant": variant, "dq": dq, "dmu": dmu, "n_cols": K, "lambda": lam, "n_nonzero": nnz,
                         "val_mse": f"{val:.6g}", "tier": tier, "vf_mse": f"{v:.6g}",
                         "t_accumulate_s": f"{t_acc:.0f}", "t_total_s": f"{time.time()-t0:.0f}"})
        tag = f"dq{dq}_dmu{dmu}" if mu_exps != "exact" else f"dq{dq}_muexact"
        torch.save({"xi": xi, "q_exps": q_exps, "mu_exps": mu_exps, "lambda": lam}, out / f"coef_{variant}_{tag}.pt")
    csv_name = "sindy_beam.csv" if args.mu_basis == "poly" else "sindy_beam_muexact.csv"
    with open(out / csv_name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"written {out/csv_name}")


if __name__ == "__main__":
    main()
