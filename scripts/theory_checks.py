#!/usr/bin/env python3
"""Two numerical checks of the theory (no retraining).

1. `eta`: for each system, the interaction fraction η = ‖∇_qV − E_μ∇_qV‖² / ‖∇_qV‖², and the singular value spectrum
   σ_k/σ_1 of the (μ,q) kernel under the loss norm. Kernel matrix G[i, (j,c)] = ∂V/∂q_c (q_j; μ_i), μ_i = N_μ LHS points in the parameter box, q_j = state pool from the dataset.
2. `subspace`: principal angles between the subspace spanned by the trained Schmidt-HNN param-net outputs u_k(μ) and the subspace of the
   first K left singular vectors of the ground-truth kernel (a testable prediction of the theory); also the fraction of kernel energy captured by the learned subspace and a random-subspace baseline.
Usage: python3 scripts/theory_checks.py eta | subspace
Output: results/theory/eta_spectrum.csv, results/theory/subspace_angles.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.dataset import load_dataset  # noqa: E402
from src.data.mu_design import maximin_lhs  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402

OUT = Path("results/theory"); OUT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)
G = torch.Generator().manual_seed(2026)


def states_from_cache(system_name, n_q):
    data, man = load_dataset(Path("data_cache") / system_name)
    tr = data["traj_train"]                                   # (n_mu, n_traj, T, 2n)
    n = tr.shape[-1] // 2
    q = tr[..., :n].reshape(-1, n).double()
    idx = torch.randperm(q.shape[0], generator=G)[:n_q]
    return q[idx]


def fpu_setup(n_q):
    import run_fpu as rfs
    from src.data.mu_design import full_grid
    from src.synthetic.planted import PlantedFamily
    rfs.N_CHAIN, rfs.SAMPLE_EVERY = 8, 40
    fam = PlantedFamily.load(Path("data_cache/fpu_family_n8.npz")); sys_ = rfs.PhysFPU(fam)
    mb = fam.phi.box; box = [(mb[i, 0].item(), mb[i, 1].item()) for i in range(4)]
    _, (y0, mu_rep, traj), _ = rfs.gen_tier(sys_, full_grid(2, box), 4, 7777)
    q = traj[..., :8].reshape(-1, 8).double(); idx = torch.randperm(q.shape[0], generator=G)[:n_q]

    def grad_v(qq, mu):
        _, pd = sys_.vector_field(qq, torch.zeros_like(qq), mu)
        return -pd
    return grad_v, box, q[idx], 8, rfs


SYSTEMS = {  # name → (system key, data cache key)
    "E2 double pendulum": ("double_pendulum", "double_pendulum"),
    "E3 HH one parameter": ("henon_heiles_ahnn", "henon_heiles_ahnn"),
    "E4 HH two parameters": ("henon_heiles_2p", "henon_heiles_2p"),
    "E7 beam": ("clamped_beam", "clamped_beam"),
}


def kernel(grad_v, box, q, n_mu):
    mu = maximin_lhs(n_mu, box, 4242).double()                # (n_mu, m)
    rows = []
    for i in range(n_mu):
        rows.append(grad_v(q, mu[i].expand(q.shape[0], -1)).reshape(-1))
    Gm = torch.stack(rows)                                    # (n_mu, n_q·n)
    return mu, Gm


def cmd_eta(args):
    rows = []
    for label, (skey, ckey) in list(SYSTEMS.items()) + [("E6 FPU", ("fpu", None))]:
        if skey == "fpu":
            grad_v, box, q, n, _ = fpu_setup(args.n_q)
        else:
            system = get_system(skey); n = system.dim
            box = system.default_mu_ranges(); q = states_from_cache(ckey, args.n_q)
            grad_v = lambda qq, mu, s=system: s.grad_q(qq, torch.zeros_like(qq), mu)
        mu, Gm = kernel(grad_v, box, q, args.n_mu)
        C = Gm - Gm.mean(0, keepdim=True)
        eta = (C.norm() ** 2 / Gm.norm() ** 2).item()
        s_raw = torch.linalg.svdvals(Gm); s_c = torch.linalg.svdvals(C)
        r_raw = (s_raw / s_raw[0]).tolist()[:8]; r_c = (s_c / s_c[0]).tolist()[:8]
        print(f"{label:10s} n={n} m={mu.shape[1]}  η={eta:.3f}  σ_k/σ_1(raw)=" + " ".join(f"{v:.1e}" for v in r_raw)
              + "\n" + " " * 28 + "σ_k/σ_1(μ-centered)=" + " ".join(f"{v:.1e}" for v in r_c), flush=True)
        rows.append({"system": label, "n": n, "m": mu.shape[1], "eta": eta,
                     **{f"sig_raw_{k+1}": v for k, v in enumerate(r_raw)}, **{f"sig_centered_{k+1}": v for k, v in enumerate(r_c)}})
    with open(OUT / "eta_spectrum.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", OUT / "eta_spectrum.csv")


def principal_angles(A, B):
    """Principal angles (degrees) between the column spaces of A and B, ascending; A, B are (N, .)."""
    Qa, _ = torch.linalg.qr(A); Qb, _ = torch.linalg.qr(B)
    c = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(c))


def captured(C, U):
    Q, _ = torch.linalg.qr(U)
    return ((Q @ (Q.T @ C)).norm() ** 2 / C.norm() ** 2).item()


def cmd_subspace(args):
    jobs = []   # (label, seed, model builder, param-net getter, R_V, bias_head, grad_v, box, q)
    # E6 FPU k=4
    grad_v, box, q, n, rfs = fpu_setup(args.n_q)
    rfs.MODELS_CFG["pshnn"]["R"] = 2; rfs.MODELS_CFG["pshnn"]["bias_head"] = True
    for seed in range(5):
        ck = rfs.OUT / f"ckpt_grid_k4{'' if seed == 0 else f'_s{seed}'}_pshnn.pt"
        m, _ = build_model("pshnn", 8, 4, rfs.MODELS_CFG, 40000); m.load_state_dict(torch.load(ck, map_location="cpu"))
        jobs.append(("E6 FPU k=4", seed, m.double().eval(), 2, True, grad_v, box, q))
    # E7 beam
    ecfg = yaml.safe_load(open("configs/beam_main.yaml")); mcfg = ecfg["models"]
    system = get_system("clamped_beam"); qb = states_from_cache("clamped_beam", args.n_q)
    gv_b = lambda qq, mu, s=system: s.grad_q(qq, torch.zeros_like(qq), mu)
    for seed in range(5):
        m, _ = build_model("pshnn", system.dim, system.mu_dim, mcfg, mcfg["target_params"])
        m.load_state_dict(torch.load(f"results/beam_main/{seed}/pshnn.pt", map_location="cpu"))
        jobs.append(("E7 beam", seed, m.double().eval(), mcfg["pshnn"]["R"], bool(mcfg["pshnn"].get("bias_head", False)), gv_b, system.default_mu_ranges(), qb))
    # E3 HH one parameter (R=8, no bias head)
    hcfg = yaml.safe_load(open("configs/ahnn_hh1p.yaml"))["models"]
    hs = get_system("henon_heiles_ahnn"); qh = states_from_cache("henon_heiles_ahnn", args.n_q)
    gv_h = lambda qq, mu, s=hs: s.grad_q(qq, torch.zeros_like(qq), mu)
    for seed in range(5):
        m, _ = build_model("pshnn", hs.dim, hs.mu_dim, hcfg, hcfg["target_params"])
        m.load_state_dict(torch.load(f"results/ahnn_repro/hh1p/{seed}/pshnn.pt", map_location="cpu"))
        jobs.append(("E3 HH one parameter", seed, m.double().eval(), hcfg["pshnn"]["R"], False, gv_h, hs.default_mu_ranges(), qh))

    rows = []; cache = {}
    for label, seed, model, R, bias, gv, box, qq in jobs:
        if label not in cache:
            mu, Gm = kernel(gv, box, qq, args.n_mu)
            C = Gm - Gm.mean(0, keepdim=True) if bias else Gm
            U, S, _ = torch.linalg.svd(C, full_matrices=False)
            K = int(((S / S[0]) > 1e-3).sum().item()); K = min(K, R)
            cache[label] = (mu, C, U, S, K)
        mu, C, U, S, K = cache[label]
        with torch.no_grad():
            u = model.param_net(mu)[:, :R]                       # (n_mu, R)
        ang = principal_angles(U[:, :K], u)
        cap, cap_opt = captured(C, u), (S[:R] ** 2).sum().item() / (S ** 2).sum().item()
        rnd = [captured(C, torch.randn(mu.shape[0], R, generator=G)) for _ in range(20)]
        rnd_ang = [principal_angles(U[:, :K], torch.randn(mu.shape[0], R, generator=G)).max().item() for _ in range(20)]
        print(f"{label:12s} seed {seed}  K={K} R={R}  angles(deg)=" + " ".join(f"{a:.2f}" for a in ang.tolist())
              + f"  captured {cap:.4f} (optimal {cap_opt:.4f}, random {sum(rnd)/len(rnd):.4f})  random max angle {sum(rnd_ang)/len(rnd_ang):.1f}", flush=True)
        rows.append({"system": label, "seed": seed, "K_truth": K, "R": R, "max_angle_deg": ang.max().item(),
                     "angles_deg": " ".join(f"{a:.3f}" for a in ang.tolist()), "captured": cap, "captured_optimal": cap_opt,
                     "captured_random": sum(rnd) / len(rnd), "random_max_angle_deg": sum(rnd_ang) / len(rnd_ang),
                     "sigma_ratio": " ".join(f"{v:.2e}" for v in (S[:8] / S[0]).tolist())})
    with open(OUT / "subspace_angles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", OUT / "subspace_angles.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eta", "subspace"])
    ap.add_argument("--n-mu", type=int, default=128); ap.add_argument("--n-q", type=int, default=2000)
    a = ap.parse_args()
    {"eta": cmd_eta, "subspace": cmd_subspace}[a.cmd](a)
