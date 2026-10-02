"""Planted-spectrum synthetic family engine (shared by the S1/S2/S3 families).

V(q;mu) = 1/2 q^T q + eps * phi(mu)^T C psi(q), C = U diag(sigma) W^T (U/W with orthonormal columns).

- the SVE=SVD lemma requires phi, psi orthonormal (verified numerically in the tests);
- psi excludes degrees 0/1 => q=0 is always an equilibrium and V(0;mu)=0;
- eps is set by binary search: min eig Hess_q V >= margin on the scan box (including the extrapolation shell);
- kinetic term fixed at M^-1 = I; the confining channel is not counted in the planted spectrum.

(U, W, sigma, eps, boxes, degrees, seed) are all saved/loaded (npz) for bit-exact reproducibility.
"""
from pathlib import Path

import numpy as np
import torch

from .dictionaries import TensorLegendre, ZeroPointLegendre


def _orthonormal_columns(rows: int, cols: int,
                         g: torch.Generator) -> torch.Tensor:
    M = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    return Q[:, :cols]


class PlantedFamily:
    """One concrete planted-family instance (fixed U, W, sigma, eps)."""

    def __init__(self, mu_box, q_box, D_mu: int, D_q: int,
                 sigma: torch.Tensor, seed: int,
                 C: torch.Tensor | None = None, eps: float | None = None,
                 hess_margin: float = 0.25,
                 scan_mu_box=None, scan_q_box=None):
        self.phi = TensorLegendre(mu_box, D_mu)
        # psi: point-constrained orthonormal dictionary (f(0)=0 and grad f(0)=0 hold exactly)
        self.psi = ZeroPointLegendre(q_box, D_q)
        self.D_mu, self.D_q, self.seed = D_mu, D_q, seed
        self.hess_margin = hess_margin
        self.sigma = sigma.to(torch.float64)
        if C is None:
            K = sigma.numel()
            g = torch.Generator().manual_seed(seed)
            self.U = _orthonormal_columns(self.phi.n_basis, K, g)
            self.W = _orthonormal_columns(self.psi.n_basis, K, g)
            self.C = self.U @ torch.diag(self.sigma) @ self.W.T
        else:
            self.C = C.to(torch.float64)
            self.U, s, Wh = torch.linalg.svd(self.C, full_matrices=False)
            self.sigma, self.W = s, Wh.T
        # eps: guarantee a positive-definite Hessian margin on the scan box (including the extrapolation shell)
        self.scan_mu_box = scan_mu_box or mu_box
        self.scan_q_box = scan_q_box or q_box
        self.eps = self._fit_eps() if eps is None else float(eps)

    # ---- core evaluation (full batch float64) -----------------------------
    def v_plant(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bf,fk,bk->b", self.phi.eval(mu), self.C,
                            self.psi.eval(q))

    def grad_v_plant(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        _, J = self.psi.eval_grad(q)
        return torch.einsum("bf,fk,bkn->bn", self.phi.eval(mu), self.C, J)

    def hess_v(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Hess_q V (including the identity from 1/2 q^T q and eps), (B,n,n)."""
        coeff = torch.einsum("bf,fk->bk", self.phi.eval(mu), self.C)
        Hp = self.psi.eval_hess_contracted(q, coeff)
        eye = torch.eye(self.psi.d, dtype=torch.float64)
        return eye + self.eps * Hp

    def _scan_hess_plant(self, n_scan: int = 4096,
                         seed: int = 0) -> torch.Tensor:
        """Sample the scan box and return Hess V_plant (without eps and the identity), (B,n,n).

        The eps binary search reuses the same Hp (Hp is independent of eps) and only recomputes eigenvalues per step.
        """
        g = torch.Generator().manual_seed(seed)
        qb = torch.tensor(self.scan_q_box, dtype=torch.float64)
        mb = torch.tensor(self.scan_mu_box, dtype=torch.float64)
        q = qb[:, 0] + (qb[:, 1] - qb[:, 0]) * torch.rand(
            n_scan, self.psi.d, generator=g, dtype=torch.float64)
        mu = mb[:, 0] + (mb[:, 1] - mb[:, 0]) * torch.rand(
            n_scan, self.phi.d, generator=g, dtype=torch.float64)
        chunks = []
        for i in range(0, n_scan, 1024):
            coeff = torch.einsum("bf,fk->bk", self.phi.eval(mu[i:i + 1024]),
                                 self.C)
            chunks.append(self.psi.eval_hess_contracted(q[i:i + 1024], coeff))
        return torch.cat(chunks)

    def min_hess_eig(self, eps: float, Hp: torch.Tensor | None = None) -> float:
        Hp = self._scan_hess_plant() if Hp is None else Hp
        eye = torch.eye(self.psi.d, dtype=torch.float64)
        return torch.linalg.eigvalsh(eye + eps * Hp).min().item()

    def _fit_eps(self) -> float:
        Hp = self._scan_hess_plant()
        lo, hi = 0.0, 1.0
        while self.min_hess_eig(hi, Hp) >= self.hess_margin and hi < 64:
            hi *= 2
        for _ in range(30):
            mid = (lo + hi) / 2
            if self.min_hess_eig(mid, Hp) >= self.hess_margin:
                lo = mid
            else:
                hi = mid
        return lo

    # ---- system interface (same method names as src/systems, M^-1 = I) -------
    @property
    def dim(self):
        return self.psi.d

    @property
    def mu_dim(self):
        return self.phi.d

    def hamiltonian(self, q, p, mu):
        V = 0.5 * (q ** 2).sum(-1) + self.eps * self.v_plant(q, mu)
        return 0.5 * (p ** 2).sum(-1) + V

    def grad_q(self, q, p, mu):
        return q + self.eps * self.grad_v_plant(q, mu)

    def grad_p(self, q, p, mu):
        return p

    def equilibrium(self):
        return torch.zeros(self.dim), torch.zeros(self.dim)

    # ---- save/load ---------------------------------------------------------
    def save(self, path: str | Path):
        np.savez(path, C=self.C.numpy(), sigma=self.sigma.numpy(),
                 eps=self.eps, D_mu=self.D_mu, D_q=self.D_q, seed=self.seed,
                 mu_box=self.phi.box.numpy(), q_box=self.psi.box.numpy(),
                 scan_mu_box=np.asarray(self.scan_mu_box),
                 scan_q_box=np.asarray(self.scan_q_box),
                 hess_margin=self.hess_margin)

    @classmethod
    def load(cls, path: str | Path) -> "PlantedFamily":
        z = np.load(path)
        return cls(mu_box=z["mu_box"].tolist(), q_box=z["q_box"].tolist(),
                   D_mu=int(z["D_mu"]), D_q=int(z["D_q"]),
                   sigma=torch.from_numpy(z["sigma"]), seed=int(z["seed"]),
                   C=torch.from_numpy(z["C"]), eps=float(z["eps"]),
                   hess_margin=float(z["hess_margin"]),
                   scan_mu_box=z["scan_mu_box"].tolist(),
                   scan_q_box=z["scan_q_box"].tolist())


def geometric_sigma(K: int, rho: float) -> torch.Tensor:
    return rho ** torch.arange(K, dtype=torch.float64)


def algebraic_sigma(K: int, s: float) -> torch.Tensor:
    return torch.arange(1, K + 1, dtype=torch.float64) ** (-s)
