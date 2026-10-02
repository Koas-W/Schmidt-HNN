"""Estimate the parameter-coordinate singular value spectrum of ∇_q V from data (paper Sec. 4.1 "Parameter budget and ranks", table tab:rank-data, data route).

Method (matches the table caption): for each tested parameter value, regress the recorded ṗ on the training + validation
trajectories onto monomials in q of degree <= d; weight the coefficient vector by the Gram matrix of the monomials on the
recorded states (G^{1/2} c, i.e. change to an orthonormal basis), stack over parameter values, take singular values, normalize by the largest.
Rank rule: number of singular values above 10^{-4} of the largest.
State pool: the common box covered by every parameter value (common_box), columns scaled by their standard deviation; the FPU chain
has the same state box for every parameter value, so no common box is taken.

Usage: python3 scripts/rank_from_data.py [hh1p hh2p fpu beam]
Output: results/theory/rank_from_data.csv (one row per system: σ_k/σ_1 for k <= 8, and the rank given by the rule)
"""
import csv, itertools, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.data.dataset import load_dataset  # noqa: E402

THRESH = 1e-4


def monomials(q, d):
    n = q.shape[1]; cols = [np.ones(q.shape[0])]
    for deg in range(1, d + 1):
        for idx in itertools.combinations_with_replacement(range(n), deg):
            cols.append(np.prod(q[:, list(idx)], axis=1))
    return np.stack(cols, 1)


def data_rank(traj, deriv, d, pool_n=4000, seed=0, common_box=True):
    """traj, deriv: (M parameter values, k trajectories, T times, 2n). Returns normalized singular values and the common box."""
    M, k, T, D = traj.shape; n = D // 2
    q_all = traj[..., :n].reshape(M, -1, n)
    box = np.abs(q_all).max(1).min(0) if common_box else np.full(n, np.inf)
    scale = q_all.reshape(-1, n).std(0)
    rng = np.random.default_rng(seed); qa = q_all.reshape(-1, n); inb = (np.abs(qa) <= box).all(1)
    pool = qa[inb]; pool = pool[rng.choice(pool.shape[0], min(pool_n, pool.shape[0]), replace=False)]
    Phi_pool = monomials(pool / scale, d); G = Phi_pool.T @ Phi_pool / pool.shape[0]
    w, U = np.linalg.eigh(G); w = np.clip(w, 0, None); Gh = U @ np.diag(np.sqrt(w)) @ U.T
    C = []
    for i in range(M):
        q = q_all[i]; pd = deriv[i, ..., n:].reshape(-1, n)
        Phi = monomials(q / scale, d); coef, *_ = np.linalg.lstsq(Phi, pd, rcond=None)
        C.append((Gh @ coef).reshape(-1))
    s = np.linalg.svd(np.stack(C), compute_uv=False); return s / s[0], box


def cached(key, d, common_box=True, pool_n=4000):
    data, man = load_dataset(ROOT / "data_cache" / key); i, _ = man["roles"]["train"]; _, jv = man["roles"]["val"]
    tr = data["traj_train"][:, i:jv].numpy(); de = data["deriv_train"][:, i:jv].numpy()
    return data_rank(tr, de, d, pool_n=pool_n, common_box=common_box)[0]


def fpu(d=4):
    import run_fpu as rfs
    from src.data.mu_design import full_grid
    from src.synthetic.planted import PlantedFamily
    rfs.N_CHAIN, rfs.SAMPLE_EVERY = 8, 40
    fam = PlantedFamily.load(ROOT / "data_cache/fpu_family_n8.npz"); sys_ = rfs.PhysFPU(fam)
    mb = fam.phi.box; box = [(mb[i, 0].item(), mb[i, 1].item()) for i in range(4)]
    tv = rfs.gen_tier(sys_, full_grid(4, box), rfs.TRAJ_TRAIN, 10)[0]; vv = rfs.gen_tier(sys_, full_grid(4, box), rfs.TRAJ_VAL, 20)[0]
    N1, N2 = tv["x"].shape[0] // 256, vv["x"].shape[0] // 256
    x = np.concatenate([tv["x"].double().numpy().reshape(256, N1, 16), vv["x"].double().numpy().reshape(256, N2, 16)], 1)
    dx = np.concatenate([tv["dx"].double().numpy().reshape(256, N1, 16), vv["dx"].double().numpy().reshape(256, N2, 16)], 1)
    return data_rank(x[:, None], dx[:, None], d, pool_n=3000, common_box=False)[0]


JOBS = {"hh1p": lambda: cached("henon_heiles_ahnn", 4), "hh2p": lambda: cached("henon_heiles_2p", 4),
        "fpu": fpu, "beam": lambda: cached("clamped_beam", 4)}

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    names = sys.argv[1:] or list(JOBS); rows = []
    for nm in names:
        s = JOBS[nm](); rank = int((s > THRESH).sum())
        print(f"{nm:6s} σ_k/σ_1 = " + " ".join(f"{v:.1e}" for v in s[:8]) + f"   rank(>{THRESH:g}) = {rank}", flush=True)
        rows.append({"system": nm, **{f"sigma_{k+1}": float(s[k]) if k < len(s) else 0.0 for k in range(8)}, "rank": rank})
    out = ROOT / "results/theory/rank_from_data.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", out)
