"""Tensor-product Legendre orthonormal dictionary.

- Orthonormal on the box Omega = prod_j [lo_j, hi_j] with respect to the uniform probability measure:
  1D P̃_k(t) = √(2k+1)·P_k(t), t ∈ [−1,1], ⟨P̃_a, P̃_b⟩_{dt/2} = δ_ab;
  multi-D takes tensor products truncated by total degree <= D, optionally dropping low degrees (the potential dictionary drops degrees 0/1).
- Fully vectorized torch float64: a per-dimension three-term recurrence builds a table (B, D+1, d), then gather+prod
  by multi-index; first/second derivatives use the Legendre derivative recurrence, no per-basis-function Python loop.
- Acceptance (tests/test_synthetic.py): orthonormality <1e-10; J_psi vs autograd.

Performance note: the evaluation core is a (B,D+1,d) table + a gather with (d_basis,d) indices,
memory O(B*d_basis*d); configuration B (d_psi=486, d=8) is processed in chunks.
"""
import itertools

import torch


def total_degree_indices(d: int, deg_max: int, deg_min: int = 0) -> torch.Tensor:
    """All multi-indices with |alpha| in [deg_min, deg_max], shape (n_basis, d)."""
    idx = [a for a in itertools.product(range(deg_max + 1), repeat=d)
           if deg_min <= sum(a) <= deg_max]
    idx.sort(key=lambda a: (sum(a), a))
    return torch.tensor(idx, dtype=torch.long)


def _legendre_tables(t: torch.Tensor, deg_max: int,
                     n_deriv: int) -> list[torch.Tensor]:
    """t (B,d) in [-1,1] -> [P, P', P''][:n_deriv+1], each (B, deg_max+1, d).

    Recurrences: (k+1)P_{k+1} = (2k+1) t P_k - k P_{k-1};
                 P'_{k+1} = P'_{k-1} + (2k+1) P_k; P''_{k+1} = P''_{k-1} + (2k+1) P'_k.
    """
    B, d = t.shape
    P = [torch.ones_like(t), t]
    dP = [torch.zeros_like(t), torch.ones_like(t)]
    d2P = [torch.zeros_like(t), torch.zeros_like(t)]
    for k in range(1, deg_max):
        P.append(((2 * k + 1) * t * P[k] - k * P[k - 1]) / (k + 1))
        dP.append(dP[k - 1] + (2 * k + 1) * P[k])
        d2P.append(d2P[k - 1] + (2 * k + 1) * dP[k])
    out = []
    for lst in (P, dP, d2P)[: n_deriv + 1]:
        out.append(torch.stack(lst[: deg_max + 1], dim=1))  # (B, D+1, d)
    return out


class TensorLegendre:
    """Orthonormal tensor-product Legendre dictionary on a box."""

    def __init__(self, box: list[tuple[float, float]], deg_max: int,
                 deg_min: int = 0):
        self.box = torch.tensor(box, dtype=torch.float64)  # (d, 2)
        self.d = self.box.shape[0]
        self.deg_max = deg_max
        self.idx = total_degree_indices(self.d, deg_max, deg_min)  # (nb, d)
        self.n_basis = self.idx.shape[0]
        # orthonormalization factor sqrt(2k+1) (uniform probability measure)
        self.norm = torch.sqrt(2.0 * torch.arange(deg_max + 1,
                                                  dtype=torch.float64) + 1.0)
        # dt/dx = 2/(hi−lo)
        self.scale = 2.0 / (self.box[:, 1] - self.box[:, 0])  # (d,)

    def _t(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.box[:, 0], self.box[:, 1]
        return (2.0 * (x - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0)

    def _gather(self, table: torch.Tensor) -> torch.Tensor:
        """(B, D+1, d) table -> per-dimension factors picked by multi-index, (B, nb, d)."""
        B = table.shape[0]
        idx = self.idx.to(table.device)  # (nb, d)
        cols = torch.arange(self.d, device=table.device)
        return table[:, idx, cols]  # broadcasting: (B, nb, d)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """psi(x), (B, n_basis)."""
        (P,) = _legendre_tables(self._t(x), self.deg_max, 0)
        P = P * self.norm.to(P.device).view(1, -1, 1)
        return self._gather(P).prod(dim=-1)

    def eval_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(psi, J); J = d psi/dx, (B, n_basis, d)."""
        P, dP = _legendre_tables(self._t(x), self.deg_max, 1)
        nrm = self.norm.to(P.device).view(1, -1, 1)
        P, dP = P * nrm, dP * nrm * self.scale.to(P.device).view(1, 1, -1)
        fP, fdP = self._gather(P), self._gather(dP)  # (B, nb, d)
        vals = fP.prod(dim=-1)
        # d_j psi = (prod_{i!=j} f_i) * f'_j: replace one factor per dimension (small loop over d)
        grads = []
        for j in range(self.d):
            g = torch.cat([fP[..., :j], fdP[..., j:j + 1],
                           fP[..., j + 1:]], dim=-1).prod(dim=-1)
            grads.append(g)
        return vals, torch.stack(grads, dim=-1)

    def eval_hess(self, x: torch.Tensor) -> torch.Tensor:
        """Hessian d^2 psi/dx^2, (B, n_basis, d, d)."""
        P, dP, d2P = _legendre_tables(self._t(x), self.deg_max, 2)
        dev = P.device
        nrm = self.norm.to(dev).view(1, -1, 1)
        s = self.scale.to(dev).view(1, 1, -1)
        P, dP, d2P = P * nrm, dP * nrm * s, d2P * nrm * s * s
        fP, fdP, fd2P = self._gather(P), self._gather(dP), self._gather(d2P)
        B, nb, d = fP.shape
        H = fP.new_zeros(B, nb, d, d)
        for i in range(d):
            for j in range(i, d):
                f = fP.clone()
                if i == j:
                    f[..., i] = fd2P[..., i]
                else:
                    f[..., i] = fdP[..., i]
                    f[..., j] = fdP[..., j]
                hij = f.prod(dim=-1)
                H[..., i, j] = hij
                H[..., j, i] = hij
        return H

    def eval_hess_contracted(self, x: torch.Tensor,
                             coeff: torch.Tensor) -> torch.Tensor:
        """sum_a coeff[b,a] * Hess psi_a(x_b), returns (B,d,d).

        Performance version: does not materialize (B,n_basis,d,d); contracts (B,n_basis) per (i,j) pair,
        memory O(B*n_basis).
        """
        P, dP, d2P = _legendre_tables(self._t(x), self.deg_max, 2)
        dev = P.device
        nrm = self.norm.to(dev).view(1, -1, 1)
        s = self.scale.to(dev).view(1, 1, -1)
        P, dP, d2P = P * nrm, dP * nrm * s, d2P * nrm * s * s
        fP, fdP, fd2P = self._gather(P), self._gather(dP), self._gather(d2P)
        B, nb, d = fP.shape
        H = fP.new_zeros(B, d, d)
        for i in range(d):
            for j in range(i, d):
                f = fP.clone()
                if i == j:
                    f[..., i] = fd2P[..., i]
                else:
                    f[..., i] = fdP[..., i]
                    f[..., j] = fdP[..., j]
                hij = torch.einsum("ba,ba->b", coeff, f.prod(dim=-1))
                H[:, i, j] = hij
                if i != j:
                    H[:, j, i] = hij
        return H

    def project(self, fn, n_quad: int | None = None) -> torch.Tensor:
        """Project a scalar function onto dictionary coefficients: c_a = <f, psi_a> (uniform probability measure).

        Gauss-Legendre tensor quadrature; fn: (Bq, d) -> (Bq,).
        Returns (n_basis,). For large dimension the quadrature nodes are processed in chunks to bound memory.
        """
        n_quad = n_quad or self.deg_max + 2
        nodes, weights = map(torch.from_numpy, (lambda r: (r[0], r[1]))(
            __import__("numpy").polynomial.legendre.leggauss(n_quad)))
        grids = torch.cartesian_prod(*([nodes] * self.d)).reshape(-1, self.d)
        w = torch.cartesian_prod(*([weights] * self.d)).reshape(-1, self.d)
        w = w.prod(dim=-1) / 2.0 ** self.d  # probability measure
        lo, hi = self.box[:, 0], self.box[:, 1]
        x = lo + (grids + 1.0) / 2.0 * (hi - lo)
        out = torch.zeros(self.n_basis, dtype=torch.float64)
        for i in range(0, x.shape[0], 65536):
            xb, wb = x[i:i + 65536], w[i:i + 65536]
            out += (self.eval(xb) * (fn(xb) * wb).unsqueeze(-1)).sum(dim=0)
        return out


class ZeroPointLegendre:
    """Point-constrained orthonormal dictionary: span{poly<=D} intersect {f(0)=0, grad f(0)=0}, orthonormal.

    Dropping the Legendre multi-indices of degree 0/1 would impose box-mean / first-moment conditions rather than
    conditions at the point q=0 (e.g. P̃_3 has nonzero gradient at 0), which would make the FPU v_k not exactly
    representable and q=0 not an exact equilibrium of the planted family. Instead, the (1+d) point-constraint
    functionals are imposed on the full Legendre basis and an orthonormal basis of their kernel is taken
    (orthonormal combinations stay orthonormal, so the SVE=SVD lemma is intact; dimension = C(d+D,D) - 1 - d).
    """

    def __init__(self, box: list[tuple[float, float]], deg_max: int):
        self.full = TensorLegendre(box, deg_max, deg_min=0)
        self.box, self.d, self.deg_max = self.full.box, self.full.d, deg_max
        zero = torch.zeros(1, self.d, dtype=torch.float64)
        vals, J = self.full.eval_grad(zero)            # (1,nf), (1,nf,d)
        A = torch.cat([vals, J[0].T], dim=0)           # (1+d, nf) constraint functionals
        _, S, Vh = torch.linalg.svd(A, full_matrices=True)
        rank = int((S > 1e-12 * S[0]).sum())
        assert rank == 1 + self.d, "point-constraint functionals should have full rank"
        self.Q = Vh[rank:].T.contiguous()              # (nf, nf-1-d) orthonormal basis of the kernel
        self.n_basis = self.Q.shape[1]

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.full.eval(x) @ self.Q

    def eval_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vals, J = self.full.eval_grad(x)
        return vals @ self.Q, torch.einsum("bad,ak->bkd", J, self.Q)

    def eval_hess(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("banm,ak->bknm", self.full.eval_hess(x), self.Q)

    def eval_hess_contracted(self, x: torch.Tensor,
                             coeff: torch.Tensor) -> torch.Tensor:
        return self.full.eval_hess_contracted(x, coeff @ self.Q.T)

    def project(self, fn, n_quad: int | None = None) -> torch.Tensor:
        return self.full.project(fn, n_quad) @ self.Q
