#!/usr/bin/env python3
"""Separation-advantage demonstration: FPU chain physical trajectory pipeline + μ-starved protocol (5 training μ).

Design:
- System: FPU chain (n=6, m=4, ε from a positive-definiteness search) as the true mechanical system;
- Data: mirrors the e1/e2 trajectory pipeline: RK4 ground-truth trajectories, 0.1 sampling, analytic derivative targets,
  no noise; boundedness: Hess V ⪰ margin on [−1,1]⁶ (guaranteed by the family construction) + energy cap
  E ≤ 0.8·min_{∂box}V(μ) (convexity => orbits never leave the box);
- μ starvation: training sees only 5 random μ; interp = new μ inside the box; extrap = ±30% shell μ;
- Models: pshnn(R=8, R_L=1) vs pshnn_concat, only difference = injection scheme, 4e4±5% parameters.

Usage: --smoke (about 2-3 min, 1 seed, 600 steps, no rollout); default full run (2 seeds, 3000 steps + rollout)
"""
import argparse
import csv
import sys

import numpy as np
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Note: no global float64 (models must be float32); data generation uses explicit float64 throughout

from src.data.integrators import rollout_sampled  # noqa: E402
from src.data.mu_design import fixed_eval_sets, full_grid  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.synthetic.fpu import make_fpu_family  # noqa: E402
from src.train.trainer import evaluate_loss, train_with_lr_sweep  # noqa: E402

OUT = Path("results/e6_fpu")
N_CHAIN, M_MU = 6, 4
DT, T_SPAN, SAMPLE_EVERY = 0.01, 20.0, 10
Q_IC, P_IC = 0.45, 0.30          # IC proposal box (boundedness guaranteed by the energy cap)
N_MU_TRAIN, N_MU_EVAL = 5, 8
TRAJ_TRAIN, TRAJ_VAL, TRAJ_EVAL = 20, 4, 4
MODELS_CFG = {   # unified protocol: Schmidt-HNN family uses the M representation (mass_param direct) and a bias head; rank given by --pshnn-R (FPU V spectrum rank 3)
    "target_params": 40000,
    "pshnn": {"R": 8, "R_L": 1, "depth": 2, "block": "swiglu", "mass_param": "direct"},
    "pshnn_mlp": {"R": 8, "R_L": 1, "depth": 2, "mass_param": "direct"},
    "pshnn_silu": {"R": 8, "R_L": 1, "depth": 2, "mass_param": "direct"},
    "pshnn_concat": {"depth": 2, "block": "swiglu", "mass_param": "direct"},
    "sep_hnn": {"R": 8, "depth": 2, "block": "swiglu"},
    "symoden_mu": {"depth": 2},
    "ahnn": {"depth": 2},
    "mlp_ode": {"depth": 2},
}


class PhysFPU:
    """FPU chain physical system (M = I); the force field uses an analytic difference stencil (~30x faster than the dictionary path;
    fam only supplies ε and spectrum bookkeeping, numerical agreement of the two is covered by the family acceptance tests)."""

    U_EXP = ((1, 2, 1, 0), (0, 1, 1, 2))   # canonical coupling channels
    V_POW = (3, 4)

    def __init__(self, fam):
        self.fam, self.dim, self.mu_dim = fam, N_CHAIN, M_MU
        self.eps = fam.eps

    def _u(self, mu):
        return [torch.prod(mu ** torch.tensor(a, dtype=mu.dtype), -1)
                for a in self.U_EXP]

    def v_plant(self, q, mu):
        d = q[:, 1:] - q[:, :-1]
        u = self._u(mu)
        return sum(uk * (d ** e).sum(-1) for uk, e in zip(u, self.V_POW))

    def grad_v_plant(self, q, mu):
        d = q[:, 1:] - q[:, :-1]
        u = self._u(mu)
        g = torch.zeros_like(q)
        for uk, e in zip(u, self.V_POW):
            de = e * d ** (e - 1)            # (B, n-1)
            g[:, :-1] -= uk.unsqueeze(-1) * de
            g[:, 1:] += uk.unsqueeze(-1) * de
        return g

    def hamiltonian(self, q, p, mu):
        return (0.5 * (p ** 2).sum(-1) + 0.5 * (q ** 2).sum(-1)
                + self.eps * self.v_plant(q, mu))

    def vector_field(self, q, p, mu):
        return p, -(q + self.eps * self.grad_v_plant(q, mu))

    def e_max(self, mu_row):
        """0.8 × minimum potential on the boundary of [−1,1]⁶ (convexity => orbits with E below it stay inside the box)."""
        g = torch.Generator().manual_seed(0)
        x = torch.rand(4096, N_CHAIN, generator=g, dtype=torch.float64) * 2 - 1
        face = torch.randint(N_CHAIN, (4096,), generator=g)
        sign = (torch.randint(0, 2, (4096,), generator=g)
                .to(torch.float64) * 2.0 - 1.0)
        x[torch.arange(4096), face] = sign
        mu = mu_row.expand(4096, -1)
        V = 0.5 * (x ** 2).sum(-1) + self.eps * self.v_plant(x, mu)
        return 0.8 * V.min().item()


def gen_tier(sys_, mu_list, n_traj, seed):
    """Build one tier as a batch: IC rejection sampling under the energy cap (per μ) -> one batched RK4 -> analytic derivatives."""
    g = torch.Generator().manual_seed(seed)
    y0s, mu_rows = [], []
    for mu_row in mu_list:
        cap = sys_.e_max(mu_row)
        acc = []
        while sum(a.shape[0] for a in acc) < n_traj:
            q = (torch.rand(8192, N_CHAIN, generator=g, dtype=torch.float64)
                 * 2 - 1) * Q_IC
            p = (torch.rand(8192, N_CHAIN, generator=g, dtype=torch.float64)
                 * 2 - 1) * P_IC
            H = sys_.hamiltonian(q, p, mu_row.expand(8192, -1))
            ok = (H <= cap) & (H >= 0.05 * cap)
            acc.append(torch.cat([q[ok], p[ok]], -1))
        y0s.append(torch.cat(acc)[:n_traj])
    n_mu = len(mu_list)
    y0 = torch.cat(y0s)                                  # (n_mu·n_traj, 2n)
    mu_rep = torch.stack(list(mu_list)).repeat_interleave(n_traj, dim=0)

    def f(y):
        qd, pd = sys_.vector_field(y[:, :N_CHAIN], y[:, N_CHAIN:], mu_rep)
        return torch.cat([qd, pd], -1)

    with torch.no_grad():
        traj = rollout_sampled(f, y0, DT, int(T_SPAN / DT), SAMPLE_EVERY)
    T1, B, D = traj.shape                                # B = n_mu·n_traj
    flat = traj.permute(1, 0, 2).reshape(T1 * B, D)
    mu_flat = mu_rep.repeat_interleave(T1, dim=0)
    with torch.no_grad():
        qd, pd = sys_.vector_field(flat[:, :N_CHAIN], flat[:, N_CHAIN:],
                                   mu_flat)
    # ground-truth energy drift check (batched)
    Ht = sys_.hamiltonian(traj.reshape(T1 * B, D)[:, :N_CHAIN],
                          traj.reshape(T1 * B, D)[:, N_CHAIN:],
                          mu_rep.repeat(T1, 1)).reshape(T1, B)
    drift = ((Ht - Ht[0]).abs() / Ht[0].abs().clamp_min(1e-12)).max().item()
    view = {"x": flat.to(torch.float32), "mu": mu_flat.to(torch.float32),
            "dx": torch.cat([qd, pd], -1).to(torch.float32)}
    return view, (y0, mu_rep, traj), drift


def rollout_l2_tiers(model, packs: dict, device, t_short=10.0) -> dict:
    """All tiers concatenated into one batch and integrated once (rollout is latency bound and insensitive to batch size,
    batching saves ~3x over sequential calls); returns the short-horizon mean pointwise L2 per tier."""
    tiers = list(packs)
    y0 = torch.cat([packs[t][0] for t in tiers])
    mu = torch.cat([packs[t][1] for t in tiers])
    ref = torch.cat([packs[t][2] for t in tiers], dim=1)
    sizes = [packs[t][0].shape[0] for t in tiers]
    mu_dev = mu.to(torch.float32).to(device)

    def f(y):
        return model.time_derivative(y, mu_dev, create_graph=False)

    with torch.no_grad():
        pred = rollout_sampled(f, y0.to(torch.float32).to(device),
                               DT, int(t_short / DT), SAMPLE_EVERY)
    Ts = pred.shape[0]
    l2 = (pred.cpu().double() - ref[:Ts]).pow(2).sum(-1).sqrt()  # (Ts, B)
    out, off = {}, 0
    for t, n in zip(tiers, sizes):
        out[t] = l2[:, off:off + n].mean().item()
        off += n
    return out


def main():
    global N_MU_TRAIN, SAMPLE_EVERY, N_CHAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=["pshnn_concat", "pshnn"])
    ap.add_argument("--pshnn-R", type=int, default=8)
    ap.add_argument("--n-mu", type=int, default=N_MU_TRAIN,
                    help="number of training μ anchors")
    ap.add_argument("--sample-every", type=int, default=SAMPLE_EVERY,
                    help="storage thinning: store one point every this many integration steps")
    ap.add_argument("--n-chain", type=int, default=N_CHAIN,
                    help="FPU chain degrees of freedom n")
    ap.add_argument("--bias-head", action="store_true",
                    help="μ-independent bias head (absorbs the μ-constant channel of the harmonic term)")
    ap.add_argument("--mu-design", choices=["random", "grid"],
                    default="random")
    ap.add_argument("--grid-k", type=int, default=2,
                    help="levels per dimension of the grid design (N_μ=k^4)")
    ap.add_argument("--fixed-eval", action="store_true",
                    help="fixed evaluation sets (interp LHS24 + three shells at 1/3 depth)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override the seed list (default smoke:[0] / full:[0,1])")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the whole training step in a CUDA Graph (6.2x speedup)")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    for _m in ("pshnn", "pshnn_mlp", "pshnn_silu"):
        MODELS_CFG[_m]["R"] = args.pshnn_R
        MODELS_CFG[_m]["bias_head"] = args.bias_head
    N_MU_TRAIN, SAMPLE_EVERY = args.n_mu, args.sample_every
    N_CHAIN = args.n_chain
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.seeds if args.seeds is not None else ([0] if args.smoke else [0, 1])
    tcfg = {"lr_grid": [1e-3] if args.smoke else [1e-3, 3e-4, 1e-4],  # includes 1e-4 (paper protocol)
            "batch_size": 8192,
            "max_steps": 600 if args.smoke else args.max_steps,
            "eval_every": 100,
            "patience": 8 if args.smoke else args.patience,
            "weight_decay": 1e-4, "cuda_graph": args.cuda_graph}

    t0 = time.time()
    cache = Path("data_cache") / f"fpu_family_n{N_CHAIN}.npz"
    if cache.exists():
        from src.synthetic.planted import PlantedFamily
        fam = PlantedFamily.load(cache)
        print(f"[fpu] family cache hit {cache}", flush=True)
    else:
        fam = make_fpu_family(seed=0, n=N_CHAIN)  # canonical channels, n adjustable
        cache.parent.mkdir(exist_ok=True)
        fam.save(cache)
    sys_ = PhysFPU(fam)
    mb = fam.phi.box
    w = (mb[:, 1] - mb[:, 0]) * 0.3
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    box_mu = [(mb[i, 0].item(), mb[i, 1].item()) for i in range(M_MU)]
    for seed in seeds:
        t_seed = time.time()
        g = torch.Generator().manual_seed(100 + seed)
        if args.mu_design == "grid":
            mu_train = full_grid(args.grid_k, box_mu)   # deterministic anchors, no design randomness
        else:
            mu_train = mb[:, 0] + (mb[:, 1] - mb[:, 0]) * torch.rand(
                N_MU_TRAIN, M_MU, generator=g, dtype=torch.float64)
        if args.fixed_eval:
            fe = fixed_eval_sets(box_mu)   # master seed 9999, same sets across tiers and seeds
            eval_mu = {"train": mu_train, "interp": fe["interp"],
                       "ex1": fe["ex1"], "ex2": fe["ex2"], "ex3": fe["ex3"]}
            eval_seed = lambda tier: 7777  # evaluation trajectory ICs are fixed as well for the fixed evaluation sets
        else:
            mu_interp = mb[:, 0] + (mb[:, 1] - mb[:, 0]) * torch.rand(
                N_MU_EVAL, M_MU, generator=g, dtype=torch.float64)
            shell = []
            while len(shell) < N_MU_EVAL:
                prop = (mb[:, 0] - w) + (mb[:, 1] - mb[:, 0] + 2 * w) *                     torch.rand(64, M_MU, generator=g, dtype=torch.float64)
                out = ((prop < mb[:, 0]) | (prop > mb[:, 1])).any(-1)
                shell += list(prop[out])
            eval_mu = {"train": mu_train, "interp": mu_interp,
                       "extrap": torch.stack(shell[:N_MU_EVAL])}
            eval_seed = lambda tier: 30 + seed

        tv, _, dr_tr = gen_tier(sys_, mu_train, TRAJ_TRAIN, 10 + seed)
        vv, _, _ = gen_tier(sys_, mu_train, TRAJ_VAL, 20 + seed)
        evs = {tier: gen_tier(sys_, ml, TRAJ_EVAL, eval_seed(tier))
               for tier, ml in eval_mu.items()}
        pts = int(T_SPAN / DT) // SAMPLE_EVERY + 1
        print(f"[seed={seed}] data ready: train {tv['x'].shape[0]} state points "
              f"({mu_train.shape[0]}μ×{TRAJ_TRAIN}traj×{pts}pts), "
              f"truth drift={dr_tr:.1e} (this phase {time.time()-t_seed:.0f}s)",
              flush=True)

        for name in args.models:
            t1 = time.time()

            def build_fn(_n=name):
                return build_model(_n, N_CHAIN, M_MU, MODELS_CFG,
                                   MODELS_CFG["target_params"])[0]

            model, tinfo = train_with_lr_sweep(build_fn, tv, vv, tcfg,
                                               seed, device)
            hist = tinfo.get("history", [])
            # file names include the seed so that multi-seed runs under one tag do not
            # overwrite each other's checkpoints
            if args.tag:
                torch.save(model.state_dict(),
                           OUT / f"ckpt_{args.tag}_s{seed}_{name}.pt")
            if args.tag and tinfo.get("histories"):
                np.savez(OUT / f"history_{args.tag}_s{seed}_{name}.npz",
                         **{f"lr_{k}": np.array(v)
                            for k, v in tinfo["histories"].items()})
            r = {"seed": seed, "model": name, "val": tinfo["best_val"],
                 "stop_step": hist[-1][0] if hist else -1,
                 "lr": tinfo["chosen_lr"]}
            for tier, (view, pack, _) in evs.items():
                r[f"vf_{tier}"] = evaluate_loss(model, view, device)
            if not args.smoke:
                ro = rollout_l2_tiers(model,
                                      {t: p for t, (v, p, _) in evs.items()},
                                      device)
                for tier, v in ro.items():
                    r[f"rollout_{tier}"] = v
            rows.append(r)
            print(f"[seed={seed}] {name:13s} " + " ".join(
                f"{k}={v:.3e}" for k, v in r.items()
                if k not in ("seed", "model")) +
                f" ({time.time()-t1:.0f}s)", flush=True)
    import json, subprocess
    if args.tag:
        snap = {"args": vars(args), "eps": sys_.eps,
                "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True
                                      ).stdout.strip(),
                "n_mu_actual": int(mu_train.shape[0])}
        (OUT / f"config_{args.tag}.json").write_text(
            json.dumps(snap, indent=1, ensure_ascii=False))
    name_csv = ("smoke" if args.smoke else "metrics") + \
        (f"_{args.tag}" if args.tag else "") + ".csv"
    with open(OUT / name_csv, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[fpu] done, total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
