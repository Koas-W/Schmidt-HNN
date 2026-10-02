"""Schmidt-HNN (Parameter-Separable HNN, code name pshnn).

    u = Param-Net(mu) in R^{heads x R}, v = State-Net(q) in R^{heads x R}
    per-head channel-wise product and sum: s_h = sum_k u_{h,k} v_{h,k}
    V(q;mu) = s_0; lower-triangular slots of L(q;mu) = s_1..s_{n(n+1)/2}
    softplus on the diagonal of L is applied after the separable sum, a deliberate design
    decision (slightly breaks pure separability; SPD is guaranteed by L L^T); do not "fix" it.
    H = ½ pᵀ L Lᵀ p + V = ½ ‖Lᵀp‖² + V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hamiltonian_base import HamiltonianModel, kinetic_from_factor
from .param_net import build_param_net
from .state_net import StateNet


class PSHNN(HamiltonianModel):
    """R_V and R_L are independent hyperparameters (capacity allocation):
    rank R_V for the V head, rank R_L for each lower-triangular slot head of L; by physical prior the entries
    of L are simpler (five of six systems have M = I), so the default direction is R_L <= R_V. R_L defaults to R,
    in which case every head has the same rank."""

    def __init__(self, n_dof: int, mu_dim: int, width: int,
                 depth: int = 2, R: int = 8, R_L: int | list[int] | None = None,
                 block: str = "swiglu", learn_kinetic: bool = True,
                 bias_head: bool = False, width_mu: int | None = None,
                 mass_param: str = "inverse"):
        super().__init__()
        self.n_dof, self.mu_dim = n_dof, mu_dim
        self.R_V, self.R_L = R, (R if R_L is None else R_L)
        self.learn_kinetic = learn_kinetic
        self.mass_param = mass_param
        # learn_kinetic=False: potential-only mode (synthetic families, M^-1 = I known)
        # T = 1/2||p||^2 fixed analytically, no L heads: the theoretical object (the C matrix of V) is learned without confounding
        n_L = n_dof * (n_dof + 1) // 2 if learn_kinetic else 0
        # R_L may be a per-slot list (length n_L, lower-triangular row-major order; ranks allocated by spectrum)
        rl = list(self.R_L) if isinstance(self.R_L, (list, tuple)) else [self.R_L] * n_L
        assert len(rl) == n_L, f"R_L list has {len(rl)} entries but there are {n_L} lower-triangular slots"
        head_ranks = [self.R_V] + rl
        # mu-independent bias head: V_hat = sum u_k v_k + b(q), so that mu-constant components
        # (e.g. confining/harmonic terms) do not occupy product channels; applies to both modes
        self.bias_head = bias_head
        state_ranks = head_ranks + ([1] if self.bias_head else [])
        self.state_net = StateNet(n_dof, width, depth, state_ranks, block)
        # width_mu: independent width of the parameter trunk (defaults to the coordinate trunk width; allocate width by the complexity of each side)
        self.param_net = build_param_net(mu_dim, width if width_mu is None else width_mu,
                                         depth, head_ranks, block)
        tril = torch.tril_indices(n_dof, n_dof)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])
        self.register_buffer("diag_mask", tril[0] == tril[1])

    def V_and_L(self, q: torch.Tensor,
                mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return V (B,) and lower-triangular L (B,n,n)."""
        B = q.shape[0]
        v = self.state_net(q)                # (B, R_V + n_L·R_L [+1])
        if self.bias_head:
            v, b = v[:, :-1], v[:, -1]
        else:
            b = 0.0
        u = self.param_net(mu)
        prod = self.param_net.split(u * v)          # channel-wise product, sliced by head
        V = prod[0].sum(dim=-1) + b
        slots = torch.stack([h.sum(dim=-1) for h in prod[1:]], dim=-1)
        slots = torch.where(self.diag_mask, F.softplus(slots), slots)
        L = q.new_zeros(B, self.n_dof, self.n_dof)
        L[:, self.tril_rows, self.tril_cols] = slots
        return V, L

    def hamiltonian(self, q, p, mu):
        if not self.learn_kinetic:
            v = self.state_net(q)
            if self.bias_head:
                v, b = v[:, :-1], v[:, -1]
            else:
                b = 0.0
            u = self.param_net(mu)
            V = (u * v).sum(dim=-1) + b
            return 0.5 * (p ** 2).sum(dim=-1) + V
        V, L = self.V_and_L(q, mu)
        return kinetic_from_factor(L, p, self.mass_param) + V
