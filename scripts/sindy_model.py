"""Wraps the polynomial potential learned by beam_sindy.py as a model with the network interface (time_derivative), for reuse in rollout-type evaluation.

V(q;μ) = Σ_{α,β} c_{αβ} q^α μ^β; q̇ = p; ṗ_j = −∂V/∂q_j = −Σ_{α} C_α(μ) α_j q^{α−e_j},
where C_α(μ) = Σ_β c_{αβ} μ^β is computed once per given μ. Monomials are built by degree recursion (vectorized, no autograd).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.beam_sindy import monomials  # noqa: E402


class SindyPotentialModel(torch.nn.Module):
    def __init__(self, coef_path: str, n_dof: int):
        super().__init__()
        d = torch.load(coef_path, map_location="cpu")
        self.q_exps = [tuple(e) for e in d["q_exps"]]          # q monomials of degree >= 1
        self.mu_exps = d["mu_exps"]
        xi = d["xi"].reshape(len(self.q_exps), -1).double()     # (Q, M)
        self.register_buffer("xi", xi)
        self.n = n_dof
        # recursion table: all monomials up to max degree (constant plus degree >= 1) as columns; column c is obtained from (parent, k)
        allx = [tuple([0] * n_dof)] + self.q_exps
        index = {e: i for i, e in enumerate(allx)}
        parent, kvar = [0] * len(allx), [0] * len(allx)
        for i, e in enumerate(allx):
            if i == 0:
                continue
            k = max(j for j in range(n_dof) if e[j] > 0)
            pe = list(e); pe[k] -= 1
            parent[i], kvar[i] = index[tuple(pe)], k
        self.parent, self.kvar = parent, kvar
        # gradient table: ∂_j q^α = α_j · q^{α−e_j}; idx[j, a] = column index (points to the constant column with coefficient 0 when α_j=0)
        idx = torch.zeros(n_dof, len(self.q_exps), dtype=torch.long)
        coefj = torch.zeros(n_dof, len(self.q_exps), dtype=torch.float64)
        for a, e in enumerate(self.q_exps):
            for j in range(n_dof):
                if e[j] > 0:
                    pe = list(e); pe[j] -= 1
                    idx[j, a] = index[tuple(pe)]; coefj[j, a] = e[j]
        self.register_buffer("gidx", idx); self.register_buffer("gcoef", coefj)

    def _monos(self, q):
        cols = [torch.ones(q.shape[0], dtype=q.dtype, device=q.device)]
        for i in range(1, len(self.parent) + 0):
            if i >= len(self.parent):
                break
            cols.append(cols[self.parent[i]] * q[:, self.kvar[i]])
        return torch.stack(cols, dim=1)                          # (N, 1+Q)

    def _mu_recursion(self):
        """Recursion table for the μ monomials (column c = column parent x μ_k), same as for q; mu_exps starts with the constant term."""
        exps = [tuple(e) for e in self.mu_exps]
        assert exps[0] == (0,) * len(exps[0]), "mu_exps must start with the constant term"
        index = {e: i for i, e in enumerate(exps)}
        parent, kvar = [0] * len(exps), [0] * len(exps)
        for i, e in enumerate(exps):
            if i == 0:
                continue
            k = max(j for j in range(len(e)) if e[j] > 0)
            pe = list(e); pe[k] -= 1
            parent[i], kvar[i] = index[tuple(pe)], k
        return parent, kvar

    def _C(self, mu):
        """C_α(μ) = Σ_β c_{αβ} μ^β by pure multiplicative recursion (no pow kernel); capturable by CUDA Graph; μ may vary in time."""
        if self.mu_exps == "exact":
            raise NotImplementedError("exact μ basis is only for the solver self-check")
        if not hasattr(self, "_mu_tab"):
            self._mu_tab = self._mu_recursion()
        parent, kvar = self._mu_tab
        mu = mu.double()
        cols = [torch.ones(mu.shape[0], dtype=mu.dtype, device=mu.device)]
        for i in range(1, len(parent)):
            cols.append(cols[parent[i]] * mu[:, kvar[i]])
        psi = torch.stack(cols, dim=1)                                            # (N, M)
        return psi @ self.xi.T                                                    # (N, Q)

    def time_derivative(self, x, mu, create_graph=False):
        q, p = x[:, :self.n].double(), x[:, self.n:].double()
        C = self._C(mu)                                          # (N, Q)
        m = self._monos(q)                                       # (N, 1+Q)
        g = m[:, self.gidx]                                      # (N, n, Q)
        pdot = -(g * (C.unsqueeze(1) * self.gcoef.unsqueeze(0))).sum(-1)
        return torch.cat([p, pdot], dim=-1).to(x.dtype)


if __name__ == "__main__":
    # self-check: ṗ must match beam_sindy.features_potential
    from scripts.beam_sindy import features_potential
    from src.data.dataset import load_dataset, eval_views
    data, man = load_dataset(Path("data_cache/clamped_beam")); v = eval_views(data, man)["interp"]
    m = SindyPotentialModel("results/beam_sindy/coef_potential_dq4_dmu3.pt", 5)
    x, mu = v["x"][:2000].double(), v["mu"][:2000].double()
    F = features_potential(x[:, :5], mu, m.q_exps, m.mu_exps)
    ref = F @ m.xi.reshape(-1)
    out = m.time_derivative(x, mu)[:, 5:]
    print("max abs diff", float((out - ref).abs().max()), " ṗ magnitude", float(ref.abs().mean()))
