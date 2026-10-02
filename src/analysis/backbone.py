"""Backbone-curve utilities: estimate the fundamental frequency omega_1 from first-mode vibration started at rest.

estimate_omega1: demean a_1(t), Hann window, rfft peak with log-spectrum parabolic
interpolation; more robust to quasi-periodic beating than zero-crossing period counting
(the two were cross-checked in scripts/beam_convergence.py).
"""
import math

import torch


def estimate_omega1(a1: torch.Tensor, dt: float) -> torch.Tensor:
    """a1 (T, B) -> omega_1 (B,)."""
    a1 = a1 - a1.mean(dim=0, keepdim=True)
    T = a1.shape[0]
    win = torch.hann_window(T, periodic=False, dtype=a1.dtype, device=a1.device)
    spec = torch.fft.rfft(a1 * win.unsqueeze(1), dim=0).abs()      # (F, B)
    freqs = torch.fft.rfftfreq(T, d=dt).to(a1.device)
    df = float(freqs[1] - freqs[0])
    out = torch.empty(a1.shape[1], dtype=torch.float64)
    for j in range(a1.shape[1]):
        s = spec[:, j]
        k = int(s[1:].argmax()) + 1
        if 1 <= k < s.numel() - 1:
            l0, l1, l2 = torch.log(s[k - 1:k + 2] + 1e-300).tolist()
            delta = 0.5 * (l0 - l2) / (l0 - 2 * l1 + l2)
        else:
            delta = 0.0
        out[j] = 2 * math.pi * (float(freqs[k]) + delta * df)
    return out


def rest_start_ic(system, mu: torch.Tensor, amp_over_h: torch.Tensor) -> torch.Tensor:
    """Backbone initial state from the ground-truth system: static equilibrium displaced along the first mode to a total peak deflection amp*h, zero momentum. (B, 2n)."""
    c, a_s = system.peak_coeff(mu, amp_over_h)
    q0 = a_s.clone()
    q0[:, 0] += c
    return torch.cat([q0, torch.zeros_like(q0)], dim=-1)
