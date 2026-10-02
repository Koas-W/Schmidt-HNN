"""mu sampling design (AHNN-aligned protocol).

- maximin_lhs: standard Latin hypercube (stratified per dim, uniform within a
  stratum, random pairing), best-of-K by maximum minimum pairwise distance.
  The 1-D uniform grid of AHNN is its 1-D special case; this is the faithful
  generalisation to m>1.
- fixed_eval_sets: fixed master evaluation set (same test points across arms,
  tiers and seeds): interp = LHS inside the box; extrap = shell points
  stratified by L-inf excursion depth, maximum depth = 1/3 of the training-box
  width (AHNN: training [0.2,0.8], extrapolation alpha=1.0 => excursion
  0.2/0.6 = 1/3), split into three equal strata.
"""
import torch


def maximin_lhs(n: int, box: list[tuple[float, float]], seed: int,
                n_trials: int = 64) -> torch.Tensor:
    """(n, d) float64, maximin Latin hypercube on the box."""
    d = len(box)
    g = torch.Generator().manual_seed(seed)
    best, best_sep = None, -1.0
    for _ in range(n_trials):
        perm = torch.stack([torch.randperm(n, generator=g)
                            for _ in range(d)], dim=1).to(torch.float64)
        u = (perm + torch.rand(n, d, generator=g, dtype=torch.float64)) / n
        sep = torch.cdist(u, u).masked_fill(
            torch.eye(n, dtype=torch.bool), float("inf")).min().item()
        if sep > best_sep:
            best, best_sep = u, sep
    b = torch.tensor(box, dtype=torch.float64)
    return b[:, 0] + best * (b[:, 1] - b[:, 0])


def fixed_eval_sets(box: list[tuple[float, float]], n_interp: int = 24,
                    n_per_shell: int = 8, n_shells: int = 3,
                    max_depth: float = 1.0 / 3.0,
                    seed: int = 9999) -> dict[str, torch.Tensor]:
    """Fixed master evaluation set: {'interp': (n_interp,d), 'ex1'..'exK': (n_per_shell,d)}.

    Shells are stratified by L-inf excursion depth e(x) = max_j(excursion_j)/width_j:
    shell k has e in ((k-1)/K*D, k/K*D], D = max_depth.
    """
    b = torch.tensor(box, dtype=torch.float64)
    lo, hi = b[:, 0], b[:, 1]
    w = (hi - lo)
    out = {"interp": maximin_lhs(n_interp, box, seed)}
    g = torch.Generator().manual_seed(seed + 1)
    edges = [max_depth * k / n_shells for k in range(n_shells + 1)]
    for k in range(n_shells):
        pts = []
        while len(pts) < n_per_shell:
            prop = (lo - max_depth * w) + (1 + 2 * max_depth) * w * torch.rand(
                256, len(box), generator=g, dtype=torch.float64)
            excess = torch.maximum(lo - prop, prop - hi).clamp_min(0) / w
            e = excess.max(dim=-1).values
            ok = (e > edges[k]) & (e <= edges[k + 1])
            pts += list(prop[ok])
        out[f"ex{k + 1}"] = torch.stack(pts[:n_per_shell])
    return out


def full_grid(k: int, box: list[tuple[float, float]]) -> torch.Tensor:
    """Full tensor grid of the AHNN protocol: Cartesian product of k levels per dim (endpoints included), (k^d, d).

    AHNN (verified): 2-D alpha1, alpha2 in {0.2,0.4,0.6,0.8}, all 16 combinations.
    """
    ks = [k] * len(box) if isinstance(k, int) else list(k)   # int: same number of levels per dim; list: per-dim levels (e.g. [8,4,4])
    assert len(ks) == len(box), f"n_train_per_dim has {len(ks)} entries but box has {len(box)} dims"
    axes = [torch.linspace(lo, hi, kk, dtype=torch.float64) for (lo, hi), kk in zip(box, ks)]
    return torch.cartesian_prod(*axes).reshape(-1, len(box))
