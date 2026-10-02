"""Model factory: build a model by name and match its parameter count automatically (baseline ~4e4, +-5%)."""
import torch.nn as nn

from .baselines.ahnn import AHNN
from .baselines.mlp_ode import MLPODE
from .baselines.symoden_mu import SymODENmu
from .known_kinetic import KnownKineticHNN, KnownKineticODE, TanhPotentialNet
from .param_match import count_params, match_width
from .pshnn import PSHNN
from .pshnn_concat import PSHNNConcat
from .sep_hnn import SepHNN

MODEL_NAMES = ("mlp_ode", "ahnn", "silu_hnn", "symoden_mu", "pshnn_concat",
               "sep_hnn", "pshnn_mlp", "pshnn_silu", "pshnn")


def build_model(name: str, n_dof: int, mu_dim: int, model_cfg: dict,
                target_params: int) -> tuple[nn.Module, dict]:
    """Return (model, info); info holds the width and the actual parameter count (written to the result snapshot)."""
    cfg = model_cfg.get(name, {})
    depth = cfg.get("depth", 2)
    # known_kinetic: T = 1/2|p|^2 is given to every model uniformly;
    # for the pshnn family this equals learn_kinetic=False; ahnn/mlp_ode/sep_hnn use a wrapper or a switch.
    known_kinetic = bool(model_cfg.get("known_kinetic", False))
    learn_kinetic = (not known_kinetic) and cfg.get("learn_kinetic", True)
    if name == "pshnn":
        R = cfg.get("R", 8)
        R_L = cfg.get("R_L")  # None -> R_L = R
        block = cfg.get("block", "swiglu")
        bias_head = cfg.get("bias_head", False)
        width_mu = cfg.get("width_mu")
        mass_param = cfg.get("mass_param", "inverse")

        def build_fn(w):
            return PSHNN(n_dof, mu_dim, w, depth=depth, R=R, R_L=R_L,
                         block=block, learn_kinetic=learn_kinetic,
                         bias_head=bias_head, width_mu=width_mu,
                         mass_param=mass_param)
    elif name in ("pshnn_mlp", "pshnn_silu"):
        # Ablation arms: same architecture as Schmidt-HNN, block replaced by an equal-parameter ungated MLP+Residual (pshnn_mlp: tanh; pshnn_silu: SiLU)
        R = cfg.get("R", 8)
        R_L = cfg.get("R_L")

        bias_head = cfg.get("bias_head", False)
        width_mu = cfg.get("width_mu")
        mass_param = cfg.get("mass_param", "inverse")
        _block = "mlp" if name == "pshnn_mlp" else "silu_mlp"

        def build_fn(w):
            return PSHNN(n_dof, mu_dim, w, depth=depth, R=R, R_L=R_L,
                         block=_block, learn_kinetic=learn_kinetic,
                         bias_head=bias_head, width_mu=width_mu,
                         mass_param=mass_param)
    elif name == "sep_hnn":
        # Ablation arm: separable injection without Cholesky (under known_kinetic the State-Net takes only q)
        R = cfg.get("R", 8)
        block = cfg.get("block", "swiglu")

        def build_fn(w):
            return SepHNN(n_dof, mu_dim, w, depth=depth, R=R, block=block,
                          learn_kinetic=learn_kinetic)
    elif name == "pshnn_concat":
        block = cfg.get("block", "swiglu")
        mass_param = cfg.get("mass_param", "inverse")

        def build_fn(w):
            return PSHNNConcat(n_dof, mu_dim, w, depth=depth, block=block,
                               learn_kinetic=learn_kinetic, mass_param=mass_param)
    elif name == "symoden_mu":
        def build_fn(w):
            return SymODENmu(n_dof, mu_dim, w, depth=depth)
    elif name == "ahnn":
        def build_fn(w):
            if known_kinetic:
                return KnownKineticHNN(n_dof, TanhPotentialNet(n_dof, mu_dim, w, depth))
            return AHNN(n_dof, mu_dim, w, depth=depth)
    elif name == "mlp_ode":
        def build_fn(w):
            if known_kinetic:
                return KnownKineticODE(n_dof, mu_dim, w, depth=depth)
            return MLPODE(n_dof, mu_dim, w, depth=depth)
    elif name == "silu_hnn":
        # Modern general-purpose MLP potential baseline: H_hat = 1/2|p|^2 + MLP_SiLU([q; mu]), defined only under known_kinetic
        # Without known_kinetic it is a full-H network identical to AHNN with SiLU activation (keeps the parameter-matching test usable)
        def build_fn(w):
            if known_kinetic:
                return KnownKineticHNN(n_dof, TanhPotentialNet(n_dof, mu_dim, w, depth, act="silu"))
            return AHNN(n_dof, mu_dim, w, depth=depth, act="silu")
    else:
        raise KeyError(f"unknown model '{name}', available: {MODEL_NAMES}")

    model, width, n_params = match_width(build_fn, target_params)
    assert count_params(model) == n_params
    return model, {"name": name, "width": width, "n_params": n_params,
                   "depth": depth, **({"R": cfg.get("R", 8)}
                                      if name == "pshnn" else {})}
