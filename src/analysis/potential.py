"""AHNN potential-energy metrics: <Delta V> relative potential error + Taylor coefficient recovery (paper Eq.4 / Eq.5).

Common convention ``ham_fn(q, p, mu) -> (B,)``, valid for both the system
(``system.hamiltonian``) and a model (``model.hamiltonian``); H values need no
graph, everything runs under no_grad.

Potential extraction (paper Fig.4 "extracting the Hamiltonian ... with constant momentum
input"): the HH kinetic energy 1/2|p|^2 is q-independent, so a fixed momentum p0 only adds
a constant to H; after min alignment the result is independent of p0. Default p0=0, giving
V_pred(q) = H(q,p0;mu) - min_domain H.

<Delta V> (Eq.4): area average over the reachable region Omega = {V_real < threshold}
    <Delta V> = mean_{q in Omega'} |V_pred(q) - V_real(q)| / V_real(q)
Omega' = Omega intersected with {max(V_pred,V_real) < threshold} (paper boundary criterion,
excludes diverging predictions) and with {V_real > floor} (excludes the 0/0 at the origin).
threshold = system.eval_domain_threshold(mu) = min(1/6, E_esc).
"""
import itertools

import torch

_DEFAULT_QBOX = (-1.2, 1.2)  # covers the {V<1/6} triangle for every lambda in [0.2,1]


def _q_grid(n, box, device, dtype):
    """(n*n, 2) configuration grid + axis coordinates. n is odd so the origin is included, keeping the constant alignment consistent."""
    xs = torch.linspace(box[0], box[1], n, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1), xs


def _chunked(fn, q, p, mu, chunk):
    """Chunked forward pass with bounded memory (grid x mu batches can reach millions of points)."""
    n = q.shape[0]
    if chunk is None or n <= chunk:
        return fn(q, p, mu)
    return torch.cat([fn(q[i:i + chunk], p[i:i + chunk], mu[i:i + chunk])
                      for i in range(0, n, chunk)])


def _central_well(domain, n):
    """4-connected component of {V_real<thr} containing the equilibrium (origin) = the paper's
    bounded region (boundary such that "almost all orbits are bounded"). For strongly
    asymmetric open angles {V<thr} is a disconnected union of central well and escape arms;
    only the well containing the origin is kept, excluding large-|q| escape arms that bounded
    orbits never reach and training never samples, which reproduces the AHNN integration domain.

    domain (M, n*n) bool; vectorised iterative dilation intersected with the domain up to a fixed point, returns bool of the same shape.
    """
    M = domain.shape[0]
    oi = n // 2                                  # origin cell (grid contains 0, n odd)
    dom = domain.reshape(M, n, n)
    well = torch.zeros_like(dom)
    well[:, oi, oi] = dom[:, oi, oi]
    for _ in range(n):                           # at most n dilations fill any connected domain
        nb = well.clone()
        nb[:, 1:, :] |= well[:, :-1, :]
        nb[:, :-1, :] |= well[:, 1:, :]
        nb[:, :, 1:] |= well[:, :, :-1]
        nb[:, :, :-1] |= well[:, :, 1:]
        nb &= dom
        if torch.equal(nb, well):
            break
        well = nb
    return well.reshape(M, n * n)


@torch.no_grad()
def _potential_on_grid(ham_fn, system, q_grid, mu_values, p_value, thr, chunk, n):
    """Batched extraction of V_pred, V_real ((M,G)); the integration domain is the bounded central well.

    Returns V_pred, V_real, well ((M,G) bool, connected component of {V_real<thr} containing the origin).
    """
    M, G = mu_values.shape[0], q_grid.shape[0]
    md = mu_values.shape[1]
    q_rep = q_grid.unsqueeze(0).expand(M, G, 2).reshape(M * G, 2)
    p_rep = torch.full_like(q_rep, float(p_value))
    mu_rep = mu_values.unsqueeze(1).expand(M, G, md).reshape(M * G, md)
    H_pred = _chunked(ham_fn, q_rep, p_rep, mu_rep, chunk).reshape(M, G)
    # ground-truth potential: p=0 => H=V (V(eq)=0, analytic min=0 at the origin)
    V_real = system.hamiltonian(q_rep, torch.zeros_like(q_rep), mu_rep).reshape(M, G)
    well = _central_well(V_real < thr.unsqueeze(1), n)
    # paper C=min(H_pred), taken over the bounded well (the in-well minimum is at the origin ~0, so this
    # equals equilibrium anchoring, but is not misled by V<0 pits in disconnected escape arms, which well excludes)
    big = torch.finfo(H_pred.dtype).max
    C = torch.where(well, H_pred, torch.full_like(H_pred, big)).min(dim=1, keepdim=True).values
    return H_pred - C, V_real, well


@torch.no_grad()
def deltaV_curve(ham_fn, system, mu_values, grid_n=129, q_box=_DEFAULT_QBOX,
                 p_value=0.0, v_floor_frac=1e-3, chunk=1 << 18,
                 metric="relative"):
    """<Delta V> vs mu (single-parameter curve / two-parameter grid, pointwise). mu_values (M, mu_dim).

    Integration domain = bounded central well (connected component, paper convention). metric:
      - "relative"  : original AHNN relative error |V_pred-V_real|/V_real (well-defined everywhere once the domain is the well);
      - "normalized": robust cross-check |V_pred-V_real|/E (E = fixed working energy).
    """
    device, dtype = mu_values.device, mu_values.dtype
    q_grid, _ = _q_grid(grid_n, q_box, device, dtype)
    thr = system.eval_domain_threshold(mu_values)                  # (M,)
    V_pred, V_real, well = _potential_on_grid(
        ham_fn, system, q_grid, mu_values, p_value, thr, chunk, grid_n)
    in_thr = torch.maximum(V_pred, V_real) < thr.unsqueeze(1)       # paper max criterion
    if metric == "normalized":
        e_work = system.working_energy() if hasattr(system, "working_energy") \
            else float(thr.max())
        err = (V_pred - V_real).abs() / e_work
        mask = well & in_thr
    else:
        v_floor = (v_floor_frac * thr).unsqueeze(1)                 # drop the 0/0 at the origin
        mask = well & in_thr & (V_real > v_floor)
        err = (V_pred - V_real).abs() / V_real.clamp_min(1e-30)
    return (err * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def potential_field_2d(ham_fn, system, mu_row, grid_n=201, q_box=_DEFAULT_QBOX,
                       p_value=0.0):
    """2-D potential field for one mu (contour plots). Returns xs, V_pred(n,n), V_real(n,n), domain(n,n)."""
    device, dtype = mu_row.device, mu_row.dtype
    q_grid, xs = _q_grid(grid_n, q_box, device, dtype)
    mu_values = mu_row.unsqueeze(0)
    thr = system.eval_domain_threshold(mu_values)
    V_pred, V_real, domain = _potential_on_grid(
        ham_fn, system, q_grid, mu_values, p_value, thr, 1 << 18, grid_n)
    n = grid_n
    return xs, V_pred.reshape(n, n), V_real.reshape(n, n), domain.reshape(n, n)


# ---- Taylor coefficients ---------------------------------------------------

def monomial_exponents(max_order=3, n_vars=4):
    """All (i1,i2,i3,i4) with 0<=sum(i)<=max_order (35 terms by default). Order (q1,q2,p1,p2)."""
    return [c for c in itertools.product(range(max_order + 1), repeat=n_vars)
            if sum(c) <= max_order]


def label(exp):
    """AHNN alpha notation: a{i1}{i2}{i3}{i4} (e.g. (2,1,0,0) -> 'a2100')."""
    return "a" + "".join(str(i) for i in exp)


# the 6 nonzero terms of the ground-truth HH Hamiltonian (coefficients in lambda or (lambda1, lambda2))
HH_TRUE_TERMS = {
    (2, 0, 0, 0): "½q₁²", (0, 2, 0, 0): "½q₂²",
    (0, 0, 2, 0): "½p₁²", (0, 0, 0, 2): "½p₂²",
    (2, 1, 0, 0): "λ q₁²q₂", (0, 3, 0, 0): "−λ/3 q₂³",
}


@torch.no_grad()
def taylor_coeffs(ham_fn, mu_values, radius=0.35, n_samples=6000,
                  max_order=3, seed=0):
    """Least-squares fit of H at the origin with monomials of order <= max_order (paper Eq.5).

    Returns (coeffs (M,K), exps (K 4-tuples)). For a truly polynomial Hamiltonian (HH) the fit
    recovers the coefficients exactly; for a black-box model it gives the best low-order
    approximation near the origin (numerical Taylor expansion).
    """
    device, dtype = mu_values.device, mu_values.dtype
    exps = monomial_exponents(max_order, 4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = ((torch.rand(n_samples, 4, generator=g, dtype=dtype) * 2 - 1) * radius).to(device)
    # design matrix built from plain products (avoids broadcast tensor**tensor JIT failures on new GPU architectures):
    # per-variable power table [1, x, x^2, ...], each monomial = product of the four matching powers.
    pw = [[torch.ones(n_samples, device=device, dtype=dtype)] for _ in range(4)]
    for j in range(4):
        for _ in range(max_order):
            pw[j].append(pw[j][-1] * x[:, j])
    Phi = torch.stack([pw[0][e[0]] * pw[1][e[1]] * pw[2][e[2]] * pw[3][e[3]]
                       for e in exps], dim=1)                        # (n_samples, K)
    q, p = x[:, :2], x[:, 2:]
    out = []
    for mu_row in mu_values:
        mu = mu_row.unsqueeze(0).expand(n_samples, -1)
        H = ham_fn(q, p, mu)
        out.append(torch.linalg.lstsq(Phi, H.unsqueeze(1)).solution.squeeze(1))
    return torch.stack(out), exps
