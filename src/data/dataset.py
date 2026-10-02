"""Dataset assembly, caching and loading.

Cache layout (data_cache/ is gitignored and fully reproducible from config + seed):
    data_cache/<system>/data.npz      # per-tier trajectories and analytic derivatives (float64)
    data_cache/<system>/manifest.json # mu split, role boundaries, generation parameter snapshot

npz keys (tier in {train, interp, extrap}):
    mu_<tier>    (n_mu, m)
    traj_<tier>  (n_mu, n_traj, T+1, 2n)   states (q||p)
    deriv_<tier> (n_mu, n_traj, T+1, 2n)   analytic (qdot||pdot) on the clean states

Trajectories of the train tier are ordered along dim 1 as [train | val | test]
(boundaries in the manifest, fixed at generation time and shared by all models and
training seeds); the interp/extrap tiers are entirely test role. Training code
reads only the train/val roles of the train tier.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ..systems import get_system
from .mu_split import make_mu_split
from .trajectories import add_state_noise, generate_tier

TIERS = ("train", "interp", "extrap")


def _make_split(system, cfg: dict) -> tuple[dict, dict]:
    """Return (split, extra manifest entries).

    mode='grid' (default): three-tier grid from make_mu_split;
    mode='fixed_eval': train = full grid with k points per dim, interp = maximin LHS
    inside the box, extrap = concatenated shells stratified by L-inf excursion depth
    (src/data/mu_design.py); shell labels go to manifest['extrap_shells']
    (1..n_shells, one per row of mu_extrap); TIERS is unchanged.
    """
    if not system.mu_dim:
        return make_mu_split([]), {}
    cfg = dict(cfg)
    mode = cfg.pop("mode", "grid")
    if mode == "grid":
        return make_mu_split(system.default_mu_ranges(), **cfg), {}
    if mode != "fixed_eval":
        raise ValueError(f"unknown mu_split.mode={mode!r}")
    from .mu_design import fixed_eval_sets, full_grid
    box = system.default_mu_ranges()
    n_shells = cfg.get("n_shells", 3)
    n_per = cfg.get("n_per_shell", 8)
    train = full_grid(cfg.get("n_train_per_dim", 4), box)
    ev = fixed_eval_sets(box, n_interp=cfg.get("n_interp", 24),
                         n_per_shell=n_per, n_shells=n_shells,
                         max_depth=cfg.get("extrap_frac", 0.3),
                         seed=cfg.get("eval_seed", 9999))
    extrap = torch.cat([ev[f"ex{k}"] for k in range(1, n_shells + 1)])
    shells = [k for k in range(1, n_shells + 1) for _ in range(n_per)]
    return ({"train": train, "interp": ev["interp"], "extrap": extrap},
            {"mu_split_mode": "fixed_eval", "extrap_shells": shells})


def generate_dataset(config: dict, out_dir: str | Path) -> dict:
    """Generate all three tiers per config, write them to disk and return the in-memory data dict."""
    sys_name = config["system"]
    system = get_system(sys_name)
    d = config["data"]
    roles = d["n_traj"]  # {train, val, test}
    n_total_train = roles["train"] + roles["val"] + roles["test"]

    split, extra_manifest = _make_split(system, d["mu_split"])

    data: dict[str, np.ndarray] = {}
    for k, tier in enumerate(TIERS):
        mu = split[tier]
        if tier != "train" and mu.shape[0] == 0:
            continue  # systems without parameters have no interp/extrap tier
        n_traj = n_total_train if tier == "train" else roles["test"]
        traj, deriv = generate_tier(
            system, mu, n_traj, d["dt"], d["t_span"], d["sample_every"],
            seed=d["seed"] + 1000 * k)
        if tier == "train" and d.get("noise_std", 0.0) > 0:
            # noise only on train/val roles; the test role stays clean as the rollout reference
            cut = roles["train"] + roles["val"]
            noisy = add_state_noise(traj[:, :cut], d["noise_std"],
                                    seed=d["seed"] + 77)
            traj = torch.cat([noisy, traj[:, cut:]], dim=1)
        data[f"mu_{tier}"] = mu.numpy()
        data[f"traj_{tier}"] = traj.numpy()
        data[f"deriv_{tier}"] = deriv.numpy()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **data)
    manifest = {
        "system": sys_name,
        "data_config": d,
        "roles": {"train": [0, roles["train"]],
                  "val": [roles["train"], roles["train"] + roles["val"]],
                  "test": [roles["train"] + roles["val"], n_total_train]},
        "tiers": {t: np.asarray(split[t]).tolist() for t in TIERS},
        "n_points_per_traj": int(round(d["t_span"] / d["dt"]))
                             // d["sample_every"] + 1,
        **extra_manifest,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return data


def load_dataset(out_dir: str | Path) -> tuple[dict, dict]:
    """Return (npz data dict as torch float64 tensors, manifest dict)."""
    out = Path(out_dir)
    with np.load(out / "data.npz") as z:
        data = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    manifest = json.loads((out / "manifest.json").read_text())
    return data, manifest


def flatten_pairs(traj: torch.Tensor, deriv: torch.Tensor,
                  mu: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(n_mu, n_traj, T, 2n) -> flat training pairs (x, mu_rep, dx), float32.

    x and dx have shape (N, 2n), mu_rep (N, m); N = n_mu*n_traj*T.
    """
    n_mu, n_traj, T, D = traj.shape
    N = n_mu * n_traj * T
    x = traj.reshape(N, D).to(torch.float32)
    dx = deriv.reshape(N, D).to(torch.float32)
    mu_rep = mu.unsqueeze(1).unsqueeze(1).expand(n_mu, n_traj, T, mu.shape[-1])
    mu_rep = mu_rep.reshape(N, mu.shape[-1]).to(torch.float32)
    return x, mu_rep, dx


def training_views(data: dict, manifest: dict) -> dict:
    """Training-visible views (train/val roles) cut from the train tier, flat float32 pairs.

    Training code may only obtain data through this function; interp/extrap keys are never exposed.
    """
    a, b = manifest["roles"]["train"]
    c, e = manifest["roles"]["val"]
    mu = data["mu_train"]
    out = {}
    for name, (i, j) in (("train", (a, b)), ("val", (c, e))):
        x, mu_rep, dx = flatten_pairs(data["traj_train"][:, i:j],
                                      data["deriv_train"][:, i:j], mu)
        out[name] = {"x": x, "mu": mu_rep, "dx": dx}
    return out


def eval_views(data: dict, manifest: dict) -> dict:
    """Evaluation views of the test role per tier: flat derivative pairs + trajectories (rollout reference)."""
    out = {}
    for tier in TIERS:
        if f"mu_{tier}" not in data:
            continue
        if tier == "train":
            i, j = manifest["roles"]["test"]
        else:
            i, j = 0, data[f"traj_{tier}"].shape[1]
        traj = data[f"traj_{tier}"][:, i:j]
        deriv = data[f"deriv_{tier}"][:, i:j]
        mu = data[f"mu_{tier}"]
        x, mu_rep, dx = flatten_pairs(traj, deriv, mu)
        out[tier] = {"x": x, "mu": mu_rep, "dx": dx,
                     "traj": traj, "mu_grid": mu}
    return out
