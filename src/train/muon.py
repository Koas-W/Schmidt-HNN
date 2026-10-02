"""Combined optimizer: Muon for matrix parameters + AdamW for the rest; pure tensor ops, CUDA Graph capturable.

Muon: the momentum gradient is orthogonalized by Newton-Schulz iteration (singular values flattened to 1) and used as the
update direction, i.e. steepest descent under the spectral norm (Jordan et al. 2024; Bernstein & Newhouse 2024). The update
magnitude uses the Kimi RMS matching (x0.2*sqrt(max(m,n))) so the same lr value matches the per-element RMS step of AdamW;
weight decay is decoupled. Non-2D parameters (biases, scalar heads) go through capturable AdamW, as in the existing protocol.

Motivation: stiff modes concentrate the loss curvature in a few singular directions, and diagonal preconditioning (Adam)
cannot equalize the anisotropy inside one matrix; spectral flattening targets this failure mode without touching the data scale.
"""
import math

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orth(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz approximation of G(G^T G)^{-1/2} (Jordan's quintic iteration coefficients)."""
    a, b, c = NS_COEFFS
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class MuonAdamW:
    """named_params: iterable of (name, Parameter). lr may be a float or a 0-dim tensor (modifiable in place after capture)."""

    def __init__(self, named_params, lr, weight_decay: float = 1e-4,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 capturable: bool = False):
        named_params = list(named_params)
        self.mat = [p for _, p in named_params if p.ndim == 2 and p.requires_grad]
        rest = [p for _, p in named_params if p.ndim != 2 and p.requires_grad]
        self.lr = lr
        self.wd, self.momentum, self.nesterov, self.ns_steps = weight_decay, momentum, nesterov, ns_steps
        self.buf = [torch.zeros_like(p) for p in self.mat]
        self.scale = [0.2 * math.sqrt(max(p.shape)) for p in self.mat]   # RMS matching
        self.adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=weight_decay,
                                       capturable=capturable) if rest else None

    # ---- interface used by the trainer ----
    @property
    def param_groups(self):
        return [{"lr": self.lr}] + (self.adamw.param_groups if self.adamw else [])

    def set_lr(self, value: float):
        if torch.is_tensor(self.lr):
            self.lr.fill_(value)
        else:
            self.lr = value
            if self.adamw:
                for g in self.adamw.param_groups:
                    g["lr"] = value

    def zero_grad(self, set_to_none: bool = True):
        for p in self.mat:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def reset_state(self):
        """Zero the state after warm-up (mirrors the AdamW state reset in the trainer)."""
        for b in self.buf:
            b.zero_()
        if self.adamw:
            for st in self.adamw.state.values():
                st["step"].zero_(); st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()

    @torch.no_grad()
    def step(self):
        lr = self.lr
        for p, buf, sc in zip(self.mat, self.buf, self.scale):
            g = p.grad
            if g is None:
                continue
            buf.mul_(self.momentum).add_(g)
            u = g.add(buf, alpha=self.momentum) if self.nesterov else buf
            O = newton_schulz_orth(u, self.ns_steps)
            # decoupled weight decay + orthogonalized direction; tensor arithmetic when lr is a tensor keeps it capturable
            p.mul_(1 - lr * self.wd).sub_(lr * sc * O)
        if self.adamw:
            self.adamw.step()
