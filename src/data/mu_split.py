"""Three-tier mu grid split: train / interp / extrap.

- train: Cartesian product of a few grid points per dim (default 4, AHNN protocol);
- interp: Cartesian product of midpoints between adjacent train grid points
  (inside the convex hull, unseen in training);
- extrap: deterministic grid points in the shell obtained by extending each dim
  outward by extrap_frac x interval width (at least one dim outside the train range).

The split is fixed at data generation time and written to the manifest; training
code only sees train-tier samples. Systems with mu_dim=0 get a single empty mu
(interp/extrap empty).
"""
import itertools

import torch


def maximin_lhs(n: int, d: int, seed: int, box: torch.Tensor | None = None,
                n_restarts: int = 64) -> torch.Tensor:
    """Maximin Latin hypercube design (standard in DoE/surrogate modelling, the
    native setting of the Loeppky 10d rule): one point per stratum in each of n
    strata per dim; restarts keep the design with the largest minimum pairwise distance.

    Seeded, so multi-seed statistics stay meaningful, while removing the "anchor hole"
    variance of purely random placement. With box (d,2) given, the design is mapped
    into the box, otherwise [0,1]^d is returned.
    """
    g = torch.Generator().manual_seed(seed)
    best, best_score = None, -1.0
    eye = torch.eye(n, dtype=torch.bool)
    for _ in range(n_restarts):
        strata = torch.stack([torch.randperm(n, generator=g)
                              for _ in range(d)], dim=-1).to(torch.float64)
        u = (strata + torch.rand(n, d, generator=g, dtype=torch.float64)) / n
        score = torch.cdist(u, u).masked_fill(eye, torch.inf).min().item()
        if score > best_score:
            best, best_score = u, score
    if box is not None:
        best = box[:, 0] + best * (box[:, 1] - box[:, 0])
    return best


def _per_dim_extrap_values(lo: float, hi: float, frac: float,
                           n_side: int) -> list[float]:
    """n_side equally spaced outward points on each side of the interval (endpoints excluded)."""
    w = hi - lo
    left = [lo - frac * w * (i + 1) / n_side for i in range(n_side)]
    right = [hi + frac * w * (i + 1) / n_side for i in range(n_side)]
    return left + right


def make_mu_split(ranges: list[tuple[float, float]],
                  n_train_per_dim: int = 4,
                  extrap_frac: float = 0.3,
                  n_extrap_side: int = 2) -> dict[str, torch.Tensor]:
    """Return {'train': (Nt,m), 'interp': (Ni,m), 'extrap': (Ne,m)} (float64)."""
    m = len(ranges)
    # mu is part of the physical ground truth and must be float64: mixing float32 mu
    # with float64 states limits H to ~1e-7 precision, which mimics energy drift
    if m == 0:
        empty = torch.zeros(0, 0, dtype=torch.float64)
        return {"train": torch.zeros(1, 0, dtype=torch.float64),
                "interp": empty, "extrap": empty}

    axes_train, axes_init, axes_extrap = [], [], []
    for lo, hi in ranges:
        grid = torch.linspace(lo, hi, n_train_per_dim, dtype=torch.float64)
        axes_train.append(grid.tolist())
        axes_init.append(((grid[:-1] + grid[1:]) / 2).tolist())
        axes_extrap.append(_per_dim_extrap_values(lo, hi, extrap_frac,
                                                  n_extrap_side))

    train = torch.tensor(list(itertools.product(*axes_train)),
                         dtype=torch.float64)
    interp = torch.tensor(list(itertools.product(*axes_init)),
                          dtype=torch.float64)

    # extrap: for each dim, take its outward values x the median train grid value of the
    # other dims, so each extrap point leaves the box along exactly one parameter direction (attributable).
    mids = [axes_train[j][len(axes_train[j]) // 2] for j in range(m)]
    extrap_pts = []
    for j in range(m):
        for v in axes_extrap[j]:
            pt = list(mids)
            pt[j] = v
            extrap_pts.append(pt)
    extrap = torch.tensor(extrap_pts, dtype=torch.float64)

    return {"train": train, "interp": interp, "extrap": extrap}
