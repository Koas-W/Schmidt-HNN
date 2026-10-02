"""Parameter-count matching, enforced in code (+-5%; width is tuned, not depth).

Parameter count is monotone in width, so a binary search finds the width closest to the target count;
a deviation beyond the tolerance raises (fairness is not left to discipline).
"""
from collections.abc import Callable

import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def match_width(build_fn: Callable[[int], nn.Module], target: int,
                tol: float = 0.05, lo: int = 4,
                hi: int = 4096) -> tuple[nn.Module, int, int]:
    """Return (model, width, n_params) with |n - target|/target <= tol guaranteed."""
    best_w, best_n = None, None
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        n = count_params(build_fn(mid))
        if best_n is None or abs(n - target) < abs(best_n - target):
            best_w, best_n = mid, n
        if n < target:
            a = mid + 1
        else:
            b = mid - 1
    rel = abs(best_n - target) / target
    if rel > tol:
        raise ValueError(f"parameter matching failed: width={best_w} gives {best_n}, "
                         f"target {target} (deviation {rel:.1%} > {tol:.0%})")
    return build_fn(best_w), best_w, best_n
