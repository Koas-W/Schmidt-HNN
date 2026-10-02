#!/usr/bin/env python3
"""Verification figures: one 2x3 figure per system.
Rows = {ablation: Schmidt-HNN + three variants; baselines: Schmidt-HNN + AHNN/SymODEN/MLP};
columns = (a) normalised trajectory error ||x_hat - x||_2 / A versus time (A = phase-space RMS amplitude of the reference trajectory),
     (b) relative energy error |H(x_hat(t)) - H(x_hat(0))| / |H(x_hat(0))| (true H) versus time,
     (c) view of a representative trajectory (phase portrait / coordinate time history / orbit).
Time is measured in linear periods T = 2*pi, 0-20T; (a) marks t = 3T with a vertical line (the instant used in the tables).
Solid line = median over non-extrapolated parameter values x all test trajectories x five seeds; band = min-max over the per-parameter-value medians (extrapolated values included).

Usage: python3 scripts/fig_verif_curves.py replay E1 E2 E3 E4 E6     # replay checkpoints, cache per-trajectory curves -> results/fig_verif/<id>/
       python3 scripts/fig_verif_curves.py plot   E1 E2 E3 E4 E6     # plot -> paper/figures/verif_<id>.pdf
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.dataset import eval_views, load_dataset          # noqa: E402
from src.data.integrators import rollout_sampled, rollout_sampled_graphed, system_field   # noqa: E402
from src.models.factory import build_model                     # noqa: E402
from src.systems import get_system                             # noqa: E402

PERIOD = 2 * math.pi
T_LONG = 20 * PERIOD
SEEDS = [0, 1, 2, 3, 4]
MODELS = ["pshnn", "pshnn_concat", "pshnn_mlp", "pshnn_silu", "sep_hnn", "ahnn", "symoden_mu", "mlp_ode"]
OUT = Path("results/fig_verif")
SPECS = {
    "E1": dict(config="configs/e1_pendulum.yaml", results="results/e1_pendulum", title="Pendulum", l2_ylim=(1e-5, 1e-1), view=("train", 0, 0), view_kind="coord", coord=0, window=(17, 20), hist_window=(18, 20)),
    "E2": dict(config="configs/e2_double_pendulum.yaml", results="results/e2_double_pendulum", title="Double pendulum", view=("interp", 0, 0), view_kind="coord", coord=1, window=(15, 20), hist_window=(18, 20), sweep=dict(axes=[("fix", 1, 1.0, 0), ("fix", 0, 0.5, 1)], center=[0.5, 1.0], half=[0.15, 0.3], train_xi=[-1, -0.5, 0, 0.5, 1])),
    "E3": dict(config="configs/ahnn_hh1p.yaml", results="results/ahnn_repro/hh1p", title="H\'enon--Heiles, one parameter", view=("interp", 2, 0), view_kind="coord", coord=1, window=(17, 20), hist_window=(18, 20), hist_plane="qp", hist_coords=[0], hist_scale=0.7, sweep=dict(axes=[("all", None, None, 0)], center=[0.5], half=[0.3], train_xi=[-1, -1/3, 1/3, 1])),
    "E4": dict(config="configs/ahnn_hh2p.yaml", results="results/ahnn_repro/hh2p", title="H\'enon--Heiles, two parameters", view=("interp", 8, 0), view_kind="coord", coord=1, window=(17, 20), hist_window=(18, 20), hist_plane="qp", hist_coords=[0], hist_scale=0.7, sweep=dict(axes=[("fix", 0, 0.6, 1), ("fix", 1, 0.6, 0)], center=[0.5, 0.5], half=[0.3, 0.3], train_xi=[-1, -1/3, 1/3, 1])),
    "E6": dict(results="results/e6_fpu", title="Fermi--Pasta--Ulam chain", view=("interp", 0, 0), view_kind="coord", coord=3, window=(17, 20), hist_window=(18, 20), hist_mode="phase", hist_backup=[("time", "_time")], k=4, sweep=dict(ray=True, train_xi=[1/3, 1])),
}


# ----------------------------------------------------------------------------- replay
def _batch_std(eid):
    """E1-E4: all test trajectories of every tier batched together -> (system, mcfg, dcfg, y0, mu, tier_of_traj, mu_index_of_traj, view_index, dt, se)"""
    sp = SPECS[eid]; cfg = yaml.safe_load(Path(sp["config"]).read_text())
    dcfg = yaml.safe_load(Path(cfg["data_config"]).read_text()) if "data_config" in cfg else cfg
    system = get_system(dcfg["system"]); data, manifest = load_dataset(Path("data_cache") / dcfg["system"])
    views = eval_views(data, manifest)
    y0s, mus, tiers, muidx, vidx = [], [], [], [], None; off = 0
    for tier, view in views.items():
        traj = view["traj"]; n_mu, k, T1, D = traj.shape
        y0s.append(traj[:, :, 0].reshape(n_mu * k, D)); mus.append(view["mu_grid"].repeat_interleave(k, 0))
        tiers += [tier] * (n_mu * k); muidx += [f"{tier}:{i}" for i in range(n_mu) for _ in range(k)]
        vt, vm, vk = sp["view"]
        if tier == vt: vidx = off + vm * k + vk
        off += n_mu * k
    return system, cfg["models"], dcfg["data"], torch.cat(y0s).double(), torch.cat(mus).double(), np.array(tiers), np.array(muidx), vidx, dcfg["data"]["dt"], dcfg["data"]["sample_every"]


def _batch_e6(eid):
    import run_fpu as rfs
    from src.data.mu_design import fixed_eval_sets, full_grid
    from src.synthetic.planted import PlantedFamily
    rfs.N_CHAIN, rfs.SAMPLE_EVERY = 8, 40
    for _m in ("pshnn", "pshnn_mlp", "pshnn_silu"): rfs.MODELS_CFG[_m]["R"] = 3; rfs.MODELS_CFG[_m]["bias_head"] = True
    fam = PlantedFamily.load(Path("data_cache") / "fpu_family_n8.npz"); sys_ = rfs.PhysFPU(fam)
    mb = fam.phi.box; box_mu = [(mb[i, 0].item(), mb[i, 1].item()) for i in range(rfs.M_MU)]; fe = fixed_eval_sets(box_mu)
    k = SPECS[eid]["k"]
    eval_mu = {"train": full_grid(k, box_mu), "interp": fe["interp"], "ex1": fe["ex1"], "ex2": fe["ex2"], "ex3": fe["ex3"]}
    y0s, mus, tiers, muidx, vidx = [], [], [], [], None; off = 0
    for tier, ml in eval_mu.items():
        y0, mu_rep, _ = rfs.gen_tier(sys_, ml, rfs.TRAJ_EVAL, 7777)[1]
        y0s.append(y0); mus.append(mu_rep); n_mu = ml.shape[0]; kk = rfs.TRAJ_EVAL
        tiers += [tier] * (n_mu * kk); muidx += [f"{tier}:{i}" for i in range(n_mu) for _ in range(kk)]
        vt, vm, vk = SPECS[eid]["view"]
        if tier == vt: vidx = off + vm * kk + vk
        off += n_mu * kk
    class Sys:   # adapter for the system_field / hamiltonian interface
        dim = 8
        def vector_field(self, q, p, mu): return sys_.vector_field(q, p, mu)
        def hamiltonian(self, q, p, mu): return sys_.hamiltonian(q, p, mu)
    return Sys(), rfs, None, torch.cat(y0s).double(), torch.cat(mus).double(), np.array(tiers), np.array(muidx), vidx, rfs.DT, rfs.SAMPLE_EVERY


def replay(eid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT / eid; out.mkdir(parents=True, exist_ok=True)
    if eid == "E6": system, rfs, _, y0, mu, tiers, muidx, vidx, dt, se = _batch_e6(eid)
    else: system, mcfg, _, y0, mu, tiers, muidx, vidx, dt, se = _batch_std(eid)
    n = system.dim; n_long = int(round(T_LONG / dt)) // se * se; t_axis = np.arange(n_long // se + 1) * dt * se
    truth_f = out / "truth.npz"
    if truth_f.exists():
        z = np.load(truth_f); truth = torch.from_numpy(z["truth"]); amp = torch.from_numpy(z["amp"])
    else:
        with torch.no_grad():
            truth = rollout_sampled(system_field(system, mu), y0, dt, n_long, se)      # (T,B,2n) float64 CPU
        amp = truth.pow(2).sum(-1).mean(0).sqrt()
        np.savez(truth_f, truth=truth.numpy(), amp=amp.numpy(), t=t_axis, tiers=tiers, muidx=muidx, vidx=vidx, mu=mu.numpy())
    print(f"[{eid}] {y0.shape[0]} trajectories, {len(t_axis)} stored points to {t_axis[-1]:.1f} = {t_axis[-1]/PERIOD:.0f}T; view index {vidx}", flush=True)
    Hx = lambda tr: system.hamiltonian(tr.reshape(-1, 2 * n)[:, :n], tr.reshape(-1, 2 * n)[:, n:], mu.repeat(tr.shape[0], 1)).reshape(tr.shape[0], -1)
    mu32 = mu.float().to(device); y032 = y0.float().to(device)
    vsel = np.where(muidx == muidx[vidx])[0]                                       # all test trajectories of the representative parameter value
    for seed in SEEDS:
        fn = out / f"curves_s{seed}.npz"
        if fn.exists(): print(f"  seed {seed} cached"); continue
        store = {}
        for name in MODELS:
            if eid == "E6":
                ck = rfs.OUT / f"ckpt_grid_k{SPECS[eid]['k']}_s{seed}_{name}.pt"
                model, _ = build_model(name, n, rfs.M_MU, rfs.MODELS_CFG, rfs.MODELS_CFG["target_params"])
            else:
                ck = Path(SPECS[eid]["results"]) / str(seed) / f"{name}.pt"
                model, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"])
            model.load_state_dict(torch.load(ck, map_location="cpu")); model.to(device).eval()
            with torch.no_grad():
                pred = rollout_sampled_graphed(lambda y: model.time_derivative(y, mu32, create_graph=False), y032, dt, n_long, se).cpu().double()
            pred = torch.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6)
            l2 = (pred - truth).pow(2).sum(-1).sqrt() / amp                          # (T,B) normalised
            Hp = Hx(pred); en = (Hp - Hp[0]).abs() / Hp[0].abs().clamp_min(1e-12)   # (T,B)
            store[f"{name}/l2"] = l2.numpy().astype(np.float32); store[f"{name}/en"] = en.numpy().astype(np.float32)
            store[f"{name}/view"] = pred[:, vsel].numpy()                             # (T,n_view,2n) all trajectories of the representative parameter value
            print(f"  seed {seed} {name:13s} l2@3T(in-range median)={np.median(l2[int(round(3*PERIOD/(dt*se))), tiers!='extrap'].numpy()) if eid!='E6' else np.median(l2[int(round(3*PERIOD/(dt*se))), np.isin(tiers,['train','interp'])].numpy()):.3g}", flush=True)
        np.savez(fn, **store)
    print(f"[{eid}] done", flush=True)


HIST_MODELS = ["pshnn", "ahnn", "symoden_mu", "mlp_ode"]
HIST_SCALE = 1.0   # default scale of the time-history/phase-plane figure relative to the 5.4 in text width; per-system override via hist_scale in SPECS
VIEW_PTS_PER_PERIOD = 128   # storage density of the time-history view (points per period); error curves still use the data step from curves_s<seed>.npz


def replay_view(eid):
    """Densely sampled replay restricted to the test trajectories of the representative parameter value (reference solution + the four models of the time-history figure); writes view_s<seed>.npz, leaves curves_s<seed>.npz untouched."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT / eid
    if eid == "E6": system, rfs, _, y0, mu, tiers, muidx, vidx, dt, _ = _batch_e6(eid)
    else: system, mcfg, _, y0, mu, tiers, muidx, vidx, dt, _ = _batch_std(eid)
    n = system.dim; se = max(1, int(round(PERIOD / VIEW_PTS_PER_PERIOD / dt))); n_long = int(round(T_LONG / dt)) // se * se
    t_axis = np.arange(n_long // se + 1) * dt * se
    vsel = np.where(muidx == muidx[vidx])[0]; y0v = y0[vsel]; muv = mu[vsel]
    with torch.no_grad(): truth = rollout_sampled(system_field(system, muv), y0v, dt, n_long, se)
    print(f"[{eid}] view config {muidx[vidx]}: {len(vsel)} trajectories, {len(t_axis)} stored points (every {dt*se:.3g})", flush=True)
    mu32 = muv.float().to(device); y032 = y0v.float().to(device)
    for seed in SEEDS:
        store = {"truth": truth.numpy(), "t": t_axis, "vsel": vsel}
        for name in HIST_MODELS:
            if eid == "E6":
                ck = rfs.OUT / f"ckpt_grid_k{SPECS[eid]['k']}_s{seed}_{name}.pt"
                model, _ = build_model(name, n, rfs.M_MU, rfs.MODELS_CFG, rfs.MODELS_CFG["target_params"])
            else:
                ck = Path(SPECS[eid]["results"]) / str(seed) / f"{name}.pt"
                model, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"])
            model.load_state_dict(torch.load(ck, map_location="cpu")); model.to(device).eval()
            with torch.no_grad():
                pred = rollout_sampled_graphed(lambda y: model.time_derivative(y, mu32, create_graph=False), y032, dt, n_long, se).cpu().double()
            store[f"{name}/view"] = torch.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6).numpy()
        np.savez(out / f"view_s{seed}.npz", **store); print(f"  seed {seed} written", flush=True)



# ----------------------------------------------------------------------------- sweep (panel b: error across the parameter domain)
def _cfg_views_std(eid):
    """E1-E4: per-parameter-value list of (x, mu, dx) views [(tier, mu_row, x, mu, dx)]"""
    sp = SPECS[eid]; cfg = yaml.safe_load(Path(sp["config"]).read_text())
    dcfg = yaml.safe_load(Path(cfg["data_config"]).read_text()) if "data_config" in cfg else cfg
    system = get_system(dcfg["system"]); data, manifest = load_dataset(Path("data_cache") / dcfg["system"])
    out = []
    for tier, view in eval_views(data, manifest).items():
        traj = view["traj"]; n_mu, k, T1, D = traj.shape; N = k * T1
        for c in range(n_mu):
            sl = slice(c * N, (c + 1) * N)
            out.append((tier, view["mu_grid"][c].numpy(), view["x"][sl], view["mu"][sl], view["dx"][sl]))
    return system, cfg["models"], out


def _cfg_views_e6(eid):
    import run_fpu as rfs
    from src.data.mu_design import fixed_eval_sets, full_grid
    from src.synthetic.planted import PlantedFamily
    rfs.N_CHAIN, rfs.SAMPLE_EVERY = 8, 40
    for _m in ("pshnn", "pshnn_mlp", "pshnn_silu"): rfs.MODELS_CFG[_m]["R"] = 3; rfs.MODELS_CFG[_m]["bias_head"] = True
    fam = PlantedFamily.load(Path("data_cache") / "fpu_family_n8.npz"); sys_ = rfs.PhysFPU(fam)
    mb = fam.phi.box; box_mu = [(mb[i, 0].item(), mb[i, 1].item()) for i in range(rfs.M_MU)]; fe = fixed_eval_sets(box_mu)
    k = SPECS[eid]["k"]; out = []
    eval_mu = {"train": full_grid(k, box_mu), "interp": fe["interp"], "ex1": fe["ex1"], "ex2": fe["ex2"], "ex3": fe["ex3"]}
    # Ray sweep (evaluation only): mu = mu_c + xi*h*d, mu_c = (1,1,1,1), h = 0.5 (half-width of the training box), d = L-inf unit direction;
    # direction set = all +-1 combinations of 1/2/3/4 parameters varied jointly (8 + 24 + 32 + 16 = 80, in +- pairs), xi in [0, 1.6] on 13 equispaced points (1.6 = 30% beyond the box)
    import itertools
    dirs = []
    for r in (1, 2, 3, 4):
        for idx in itertools.combinations(range(4), r):
            for signs in itertools.product((-1.0, 1.0), repeat=r):
                d = [0.0] * 4
                for i_, s_ in zip(idx, signs): d[i_] = s_
                dirs.append(d)
    dirs = torch.tensor(dirs, dtype=torch.float64); xis = torch.linspace(0, 1.6, 13, dtype=torch.float64)
    rays = [torch.ones(4, dtype=torch.float64)] + [1.0 + xi * 0.5 * d for xi in xis[1:] for d in dirs]
    eval_mu["ray"] = torch.stack(rays)
    for tier, ml in eval_mu.items():
        view = rfs.gen_tier(sys_, ml, rfs.TRAJ_EVAL, 7777)[0]; N = view["x"].shape[0] // ml.shape[0]
        for c in range(ml.shape[0]):
            sl = slice(c * N, (c + 1) * N)
            out.append((tier, ml[c].numpy(), view["x"][sl], view["mu"][sl], view["dx"][sl]))
    return rfs, out


def sweep(eid):
    """Per-parameter-value relative vector-field residual sum||f_hat - xdot||^2 / sum||xdot||^2 (over all evaluation states of that value) -> results/fig_verif/<id>/sweep.npz"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT / eid; out.mkdir(parents=True, exist_ok=True)
    if eid == "E6": rfs, cfgs = _cfg_views_e6(eid)
    else: system, mcfg, cfgs = _cfg_views_std(eid)
    res = np.full((len(SEEDS), len(MODELS), len(cfgs)), np.nan)
    for si, seed in enumerate(SEEDS):
        for mi, name in enumerate(MODELS):
            if eid == "E6":
                ck = rfs.OUT / f"ckpt_grid_k{SPECS[eid]['k']}_s{seed}_{name}.pt"
                model, _ = build_model(name, 8, rfs.M_MU, rfs.MODELS_CFG, rfs.MODELS_CFG["target_params"])
            else:
                ck = Path(SPECS[eid]["results"]) / str(seed) / f"{name}.pt"
                model, _ = build_model(name, system.dim, system.mu_dim, mcfg, mcfg["target_params"])
            model.load_state_dict(torch.load(ck, map_location="cpu")); model.to(device).eval()
            for ci, (tier, mu_row, x, mu, dx) in enumerate(cfgs):
                num = den = 0.0
                for i in range(0, x.shape[0], 65536):
                    xb, mub, dxb = x[i:i+65536].to(device), mu[i:i+65536].to(device), dx[i:i+65536].to(device)
                    pred = model.time_derivative(xb, mub, create_graph=False)
                    num += (pred - dxb).pow(2).sum().item(); den += dxb.pow(2).sum().item()
                res[si, mi, ci] = num / den
        print(f"  [{eid}] seed {seed} sweep done", flush=True)
    np.savez(out / "sweep.npz", res=res, tiers=np.array([c[0] for c in cfgs]), mu=np.stack([c[1] for c in cfgs]), models=np.array(MODELS))
    print(f"[{eid}] sweep written ({len(cfgs)} configurations)", flush=True)


# ----------------------------------------------------------------------------- plot
# Colours: the Elsevier author guide ("In the eye of the beholder") recommends a colour-blind-safe palette plus line styles/markers; baselines use the Okabe-Ito palette,
# the proposed model its blue #0072B2, the three ablation variants desaturated lighter steps of the same blue family (one family at a glance), thinner lines, 80% opacity, distinct line styles;
# baselines also thin lines, 80% opacity, fixed markers (triangle/diamond/square); MLP as the lowest baseline in plain grey dashed.
COLOR = {"truth": "#000000", "pshnn": "#0072B2", "pshnn_concat": "#2F6F9F", "pshnn_mlp": "#4C5FA6", "pshnn_silu": "#4C5FA6", "sep_hnn": "#2E7F8C",
         "ahnn": "#C13E3D", "symoden_mu": "#008662", "mlp_ode": "#7F7F7F"}   # ablations: cool dark colours of similar lightness, each shifted in one channel; baselines: red #e34948 and Okabe-Ito blue-green #009E73, each darkened by 15%
STYLE = {"pshnn": "-", "pshnn_concat": "--", "pshnn_mlp": "-.", "pshnn_silu": (0, (4, 1, 1, 1, 1, 1)), "sep_hnn": ":", "ahnn": "-", "symoden_mu": "-", "mlp_ode": "--"}
ZORDER = {"pshnn": 10, "pshnn_concat": 9, "pshnn_mlp": 8, "pshnn_silu": 7.5, "sep_hnn": 7, "ahnn": 9, "symoden_mu": 8, "mlp_ode": 7}   # stacking order: proposed model on top, MLP at the bottom
LW = {"pshnn": 0.8, "pshnn_concat": 0.7, "pshnn_mlp": 0.7, "pshnn_silu": 0.7, "sep_hnn": 0.7, "ahnn": 0.7, "symoden_mu": 0.7, "mlp_ode": 0.7}
ALPHA = {"pshnn": 1.0, "pshnn_concat": 0.92, "pshnn_mlp": 0.92, "pshnn_silu": 0.92, "sep_hnn": 0.92, "ahnn": 0.92, "symoden_mu": 0.92, "mlp_ode": 0.92}
MARKER = {"pshnn": "o", "pshnn_concat": "o", "pshnn_mlp": "o", "pshnn_silu": "o", "sep_hnn": "o", "ahnn": "^", "symoden_mu": "D", "mlp_ode": "s"}
MFC = {m: "white" for m in ("pshnn_concat", "pshnn_mlp", "pshnn_silu", "sep_hnn")}   # ablation variants: hollow circles
LABEL = {"truth": "reference", "pshnn": "Schmidt-HNN (present)", "pshnn_concat": "w/o separation", "pshnn_mlp": "w/o gating (tanh)", "pshnn_silu": "w/o gating (SiLU)", "sep_hnn": "w/o Cholesky",
         "ahnn": "AHNN (Han et al. 2021)", "symoden_mu": "SymODEN (Zhong et al. 2020)", "mlp_ode": "MLP (Greydanus et al. 2019)"}
LEG = dict(fontsize=5.6, handlelength=1.4, borderpad=0.35, labelspacing=0.25, handletextpad=0.5, frameon=True, facecolor="white", framealpha=1.0, edgecolor="none", fancybox=False)
ROWS = [("Ablation", ["pshnn", "pshnn_concat", "pshnn_mlp", "pshnn_silu", "sep_hnn"]), ("Baselines", ["pshnn", "ahnn", "symoden_mu", "mlp_ode"])]


def _agg(arr, tiers, muidx):
    """arr (T,B) -> median line (non-extrapolated) and band (min-max over per-parameter-value medians, extrapolated included)."""
    inr = ~np.char.startswith(tiers.astype(str), "ex")
    line = np.median(arr[:, inr], axis=1)
    cfg_med = np.stack([np.median(arr[:, muidx == c], axis=1) for c in np.unique(muidx)], axis=1)
    return line, cfg_med.min(1), cfg_med.max(1)



def _panel_sweep(ax, sw, spec, models, xlabel):
    """(b) relative vector-field residual versus normalised parameter coordinate xi = (mu_j - c_j)/h_j (training interval -> [-1, 1]).
    Per model: first axis line dark, second axis line light (two-parameter systems); FPU: median over the four axis lines as solid line, min-max as light band."""
    res = sw["res"]; tiers = sw["tiers"]; mu = sw["mu"]; names = list(sw["models"])
    med = np.median(res, axis=0)                                                  # (model, cfg)
    if spec.get("ray"):   # FPU: grouped by normalised L-inf distance xi to the box centre; solid line = median over the 80 directions, band = min-max
        sel = tiers == "ray"; xi_all = np.round(np.abs(mu[sel] - 1.0).max(1) / 0.5, 3); xs = np.unique(xi_all)
        for m in models:
            y = med[names.index(m)][sel]
            q = np.array([[np.median(y[xi_all == v]), y[xi_all == v].min(), y[xi_all == v].max()] for v in xs])
            ax.fill_between(xs, q[:, 1], q[:, 2], color=COLOR[m], alpha=0.15, linewidth=0)
            ax.plot(xs, q[:, 0], linestyle=STYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], marker=MARKER[m], markersize=2.2, markerfacecolor=MFC.get(m, COLOR[m]), markeredgewidth=0.6, label=LABEL[m], zorder=ZORDER[m], clip_on=False)
        ax.set_xlim(0, xs.max() * 1.02)   # distance axis starts at 0 (xi >= 0); left-end marker not clipped
        ax.axvspan(0, 1, color="#000000", alpha=0.06, linewidth=0, zorder=0.55)
        for v in spec["train_xi"]: ax.axvline(v, color="#000000", linewidth=0.4, alpha=0.25, zorder=0.6)
        ax.set_yscale("log"); ax.set_ylabel("Relative vector-field residual")
        if xlabel: ax.set_xlabel(r"Normalised parameter distance $\|\tilde{\boldsymbol{\mu}}\|_\infty$")
        ax.set_axisbelow(True); ax.grid(True, linewidth=0.3, color="#dddddd"); return
    curves = []
    for kind, k, val, j in spec["axes"]:
        if kind == "fix": sel = np.abs(mu[:, k] - val) < 1e-6
        elif kind == "tier": sel = tiers == k
        else: sel = np.ones(len(tiers), bool)
        xi = (mu[sel, j] - spec["center"][j]) / spec["half"][j]; order = np.argsort(xi)
        curves.append((xi[order], {m: med[names.index(m)][sel][order] for m in models}))
    if spec.get("band"):
        xi = curves[0][0]
        for m in models:
            Y = np.stack([c[1][m] for c in curves]); ax.fill_between(xi, Y.min(0), Y.max(0), color=COLOR[m], alpha=0.15, linewidth=0)
            ax.plot(xi, np.median(Y, 0), linestyle=STYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], marker=MARKER[m], markersize=2.2, markerfacecolor=MFC.get(m, COLOR[m]), markeredgewidth=0.6, label=LABEL[m], zorder=ZORDER[m])
    else:
        for a_i, (xi, ys) in enumerate(curves):
            for m in models:
                ax.plot(xi, ys[m], linestyle=STYLE[m], color=COLOR[m], linewidth=LW[m], marker=MARKER[m], markersize=2.2, markerfacecolor=MFC.get(m, COLOR[m]), markeredgewidth=0.6, alpha=ALPHA[m] if a_i == 0 else 0.35, label=LABEL[m] if a_i == 0 else None, zorder=ZORDER[m] - (0 if a_i == 0 else 0.5))
    ax.axvspan(-1, 1, color="#000000", alpha=0.06, linewidth=0, zorder=0.55)
    for v in spec["train_xi"]: ax.axvline(v, color="#000000", linewidth=0.4, alpha=0.25, zorder=0.6)
    ax.set_yscale("log"); ax.set_ylabel("Relative vector-field residual")
    if xlabel: ax.set_xlabel(r"Normalised parameter $\tilde{\mu}_j$")
    ax.set_axisbelow(True); ax.grid(True, linewidth=0.3, color="#dddddd")


def _load_curves(eid):
    sp = SPECS[eid]; d = OUT / eid; z = np.load(d / "truth.npz", allow_pickle=True)
    S = {s_: np.load(d / f"curves_s{s_}.npz") for s_ in SEEDS}
    sw = np.load(d / "sweep.npz") if (d / "sweep.npz").exists() else None
    return sp, z, S, sw


def _legend(fig, entries, y=0.0, ncol=4):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[m], linestyle=STYLE.get(m, "-"), linewidth=1.2 if m == "truth" else LW[m], alpha=ALPHA.get(m, 1.0)) for m in entries]
    fig.legend(handles, [LABEL[m] for m in entries], loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.2, handletextpad=0.6)

HSTYLE = {"pshnn": (0, (5.5, 2.8)), "ahnn": (0, (4, 1.5)), "symoden_mu": (0, (2.5, 1.2)), "mlp_ode": (0, (1, 1.2))}   # time histories: proposed model and baselines all dashed styles, reference solution solid


def _hist_data(eid, sp, S, z, truth, seed_view, jv, vidx):
    """Reference solution and the four model predictions for the representative trajectory (densely sampled replay_view output preferred), plus the time-history window."""
    vf = OUT / eid / f"view_s{seed_view}.npz"
    if vf.exists(): zv = np.load(vf); t = zv["t"] / PERIOD; tr = zv["truth"][:, jv]; V = {m: zv[f"{m}/view"][:, jv] for m in HIST_MODELS}
    else: t = z["t"] / PERIOD; tr = truth[:, vidx]; V = {m: S[seed_view][f"{m}/view"][:, jv] for m in HIST_MODELS}
    w0, w1 = sp.get("hist_window", sp["window"]); sl = (t >= w0) & (t <= w1)
    return t, tr, V, w0, w1, sl


def _hist_rows_figure(rows, name, sc=1.0):
    """Each row = (sp, t, tr, V, w0, w1, sl, j): left = time history of coordinate j, right = phase plane (q_j, p_j) or configuration plane; sc = scale relative to 5.4 in, font size unchanged."""
    import matplotlib.pyplot as plt
    nr = len(rows); W, H = 5.4 * sc, 2.3 * nr * sc
    HLEG = dict(LEG, fontsize=LEG["fontsize"] * 0.66, handlelength=1.1, borderpad=0.3, labelspacing=0.2, handletextpad=0.4)   # legend: about 35% smaller, lower-right corner of (a)
    fig, axes = plt.subplots(nr, 2, figsize=(W, H), gridspec_kw={"width_ratios": [3, 2]}, squeeze=False)
    fig.subplots_adjust(left=0.486 / W, right=1 - 0.108 / W, top=1 - 0.23 / H, bottom=0.46 / H, wspace=0.32, hspace=0.27 / sc)   # margins and row spacing given in inches, so label space is unchanged under scaling
    for r, (sp, t, tr, V, w0, w1, sl, j) in enumerate(rows):
        axL, axR = axes[r]; n = tr.shape[-1] // 2
        axL.plot(t[sl], tr[sl, j], color=COLOR["truth"], linewidth=LW["pshnn"], label=LABEL["truth"], zorder=5)
        for m in HIST_MODELS:
            v = V[m]; axL.plot(t[sl], v[sl, j], linestyle=HSTYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], label=LABEL[m], zorder=ZORDER[m])
        axL.set_xlim(w0, w1); axL.set_ylabel(rf"$q_{{{j+1}}}$" if n > 1 else r"$q$")
        if r == nr - 1: axL.set_xlabel(r"Dimensionless time $\tilde{t}$")
        if n > 1 and sp.get("hist_plane", "qq") == "qq": a, b = min(j, (j + 1) % n), max(j, (j + 1) % n); la, lb = rf"$q_{{{a+1}}}$", rf"$q_{{{b+1}}}$"
        elif n > 1: a, b = j, n + j; la, lb = rf"$q_{{{j+1}}}$", rf"$p_{{{j+1}}}$"   # hist_plane="qp": phase plane (q_j, p_j) of the plotted coordinate
        else: a, b = 0, 1; la, lb = r"$q$", r"$p$"
        axR.plot(tr[sl, a], tr[sl, b], color=COLOR["truth"], linewidth=LW["pshnn"], zorder=5)
        for m in HIST_MODELS:
            v = V[m]; axR.plot(v[sl, a], v[sl, b], linestyle=HSTYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], zorder=ZORDER[m])
        axR.set_xlabel(la); axR.set_ylabel(lb)
        for ax_, lab in ((axL, f"({chr(97 + 2 * r)})"), (axR, f"({chr(98 + 2 * r)})")): ax_.set_title(lab, loc="left", pad=3)
    axes[0][0].legend(loc="lower right", ncol=1, **HLEG).set_zorder(20)
    fig.savefig(Path("paper/figures") / f"{name}.pdf"); fig.savefig(Path("paper/figures") / f"{name}.png"); plt.close(fig)


def _hist_row_of(eid):
    """Representative trajectory of one system as a row of the merged figure (same selection rule as plot: median seed, test trajectory of median amplitude, coord from SPECS)."""
    sp, z, S, _ = _load_curves(eid)
    t = z["t"] / PERIOD; tiers = z["tiers"]; muidx = z["muidx"]; truth = z["truth"]; vidx = int(z["vidx"])
    i3 = int(np.argmin(np.abs(t - 3))); inr = ~np.char.startswith(tiers.astype(str), "ex")
    seed_view = sorted(SEEDS, key=lambda s_: np.median(S[s_]["pshnn/l2"][i3, inr]))[2]
    vsel = np.where(muidx == muidx[vidx])[0]; jv = int(np.argsort(z["amp"][vsel])[len(vsel) // 2]); vidx = int(vsel[jv])
    t, tr, V, w0, w1, sl = _hist_data(eid, sp, S, z, truth, seed_view, jv, vidx)
    return (sp, t, tr, V, w0, w1, sl, sp.get("hist_coords", [sp["coord"]])[0])


def hist_merged(eids, name, sc):
    """Merged multi-system time-history figure: one row per system (e.g. E3 and E4 merged into hist_hh)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, os, scienceplots  # noqa: F401
    os.environ["PATH"] = str(Path(__file__).resolve().parents[1] / ".texlive" / "bin" / "x86_64-linux") + os.pathsep + os.environ["PATH"]
    plt.style.use(["science", str(Path(__file__).resolve().parent / "figstyle" / "science.mplstyle")])
    _hist_rows_figure([_hist_row_of(e) for e in eids], name, sc); print(f"[{'+'.join(eids)}] {name} written (scale {sc})")


def plot(eid):
    """Figure A: verif_<id>: 2x2 (rows = ablation / baselines; columns = normalised trajectory error versus time, relative residual versus parameter).
       Figure B: hist_<id>: single figure, coordinate time histories of the representative trajectory, reference solution + Schmidt-HNN + three baselines only."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os, scienceplots  # noqa: F401
    os.environ["PATH"] = str(Path(__file__).resolve().parents[1] / ".texlive" / "bin" / "x86_64-linux") + os.pathsep + os.environ["PATH"]   # usetex uses the project-local TeX Live (latex + dvipng)
    plt.style.use(["science", str(Path(__file__).resolve().parent / "figstyle" / "science.mplstyle")])
    sp, z, S, sw = _load_curves(eid)
    t = z["t"] / PERIOD; tiers = z["tiers"]; muidx = z["muidx"]; truth = z["truth"]; vidx = int(z["vidx"])
    i3 = int(np.argmin(np.abs(t - 3))); inr = ~np.char.startswith(tiers.astype(str), "ex")
    med_by_seed = {s_: np.median(S[s_]["pshnn/l2"][i3, inr]) for s_ in SEEDS}
    seed_view = sorted(SEEDS, key=lambda s_: med_by_seed[s_])[2]
    vsel = np.where(muidx == muidx[vidx])[0]
    jv = int(np.argsort(z["amp"][vsel])[len(vsel) // 2]); vidx = int(vsel[jv])   # representative trajectory = test trajectory of median amplitude for that parameter value

    # ---------------- Figure A ----------------
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.11, wspace=0.35, hspace=0.35)
    for r, (rname, models) in enumerate(ROWS):
        ax = axes[r, 0]
        for m in models:
            arr = np.concatenate([S[s_][f"{m}/l2"] for s_ in SEEDS], axis=1)
            tt = np.tile(tiers, len(SEEDS)); mm = np.tile(muidx, len(SEEDS))
            line, lo, hi = _agg(arr, tt, mm)
            ax.fill_between(t, np.maximum(lo, 1e-12), hi, color=COLOR[m], alpha=0.12, linewidth=0)
            ax.plot(t, line, linestyle=STYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], label=LABEL[m], zorder=ZORDER[m])
        ax.set_yscale("log"); ax.set_xlim(0, 20); ax.set_ylim(*sp.get("l2_ylim", (1e-4, 2)))
        ax.axvline(3, color="#888888", linewidth=0.5, zorder=0.6); ax.set_ylabel(r"Normalised $L_2$ error")
        if r == 1: ax.set_xlabel(r"Dimensionless time $\tilde{t}$")
        ax.legend(loc="lower right", **LEG)
        ax = axes[r, 1]
        if sw is not None and "sweep" in sp:
            _panel_sweep(ax, sw, sp["sweep"], models, r == 1)
        else:   # pendulum: no parameter, show relative energy error versus time instead
            for m in models:
                arr = np.concatenate([S[s_][f"{m}/en"] for s_ in SEEDS], axis=1)
                tt = np.tile(tiers, len(SEEDS)); mm = np.tile(muidx, len(SEEDS))
                line, lo, hi = _agg(arr, tt, mm)
                ax.fill_between(t, np.maximum(lo, 1e-12), hi, color=COLOR[m], alpha=0.12, linewidth=0)
                ax.plot(t, line, linestyle=STYLE[m], color=COLOR[m], linewidth=0.6, alpha=ALPHA[m], label=LABEL[m], zorder=ZORDER[m])
            ax.set_yscale("log"); ax.set_xlim(0, 20); ax.set_ylim(1e-6, 1e-1); ax.set_ylabel(r"Relative energy error $|\Delta E|/E_0$")
            if r == 1: ax.set_xlabel(r"Dimensionless time $\tilde{t}$")
        for c in range(2):
            axes[r, c].set_title(f"({'abcd'[2 * r + c]})" + (f" {rname.lower()}" if c == 0 else ""), loc="left", pad=3)
    if sw is not None and "sweep" in sp:
        lo = min(axes[r, 1].get_ylim()[0] for r in range(2)); hi = max(axes[r, 1].get_ylim()[1] for r in range(2))
        for r in range(2): axes[r, 1].set_ylim(lo, hi)
    fig.savefig(Path("paper/figures") / f"verif_{eid}.pdf"); fig.savefig(Path("paper/figures") / f"verif_{eid}.png"); plt.close(fig)

    # ---------------- Figure B: each row = one coordinate (left = time history, right = phase plane (q_j, p_j) or configuration plane (q_a, q_b)); hist_coords gives the rows; reference solution + proposed model + three baselines only ----------------
    models = HIST_MODELS
    n = truth.shape[-1] // 2; coords = sp.get("hist_coords", [sp["coord"]]); nr = len(coords)
    t, tr, V, w0, w1, sl = _hist_data(eid, sp, S, z, truth, seed_view, jv, vidx)
    HLEG = dict(LEG, fontsize=LEG["fontsize"] * 0.66, handlelength=1.1, borderpad=0.3, labelspacing=0.2, handletextpad=0.4)   # Figure B legend: about 35% smaller, lower-right corner of (a)
    mode = sp.get("hist_mode", "rows")   # rows: one row per coordinate (time history + plane); time: grid of all coordinate time histories; phase: grid of all phase planes
    grid_modes = [(mode, "")] + list(sp.get("hist_backup", [])) if mode in ("time", "phase") else []   # hist_backup: extra outputs as (mode, file suffix)
    for mode, suffix in grid_modes:
        coords = list(range(n)); nc = 2 if mode == "time" else 4; nr = math.ceil(n / nc); rh = 1.35 if mode == "time" else 1.3; H = rh * nr + 0.25
        fig, axes = plt.subplots(nr, nc, figsize=(5.4, H), squeeze=False, sharex=(mode == "time"))
        fig.subplots_adjust(left=0.09, right=0.98, top=1 - 0.32 / H, bottom=(0.42 if mode == "time" else 0.45) / H, wspace=0.3 if mode == "time" else 0.5, hspace=0.35 if mode == "time" else 0.5)
        for i, j in enumerate(coords):
            ax = axes[i // nc][i % nc]
            if mode == "time":
                ax.plot(t[sl], tr[sl, j], color=COLOR["truth"], linewidth=LW["pshnn"], label=LABEL["truth"], zorder=5)
                for m in models:
                    v = V[m]; ax.plot(t[sl], v[sl, j], linestyle=HSTYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], label=LABEL[m], zorder=ZORDER[m])
                ax.set_xlim(w0, w1); ax.set_ylabel(rf"$q_{{{j+1}}}$")
                if i // nc == nr - 1: ax.set_xlabel(r"Dimensionless time $\tilde{t}$")
            else:
                ax.plot(tr[sl, j], tr[sl, n + j], color=COLOR["truth"], linewidth=LW["pshnn"], label=LABEL["truth"], zorder=5)
                for m in models:
                    v = V[m]; ax.plot(v[sl, j], v[sl, n + j], linestyle=HSTYLE[m], color=COLOR[m], linewidth=LW[m], alpha=ALPHA[m], label=LABEL[m], zorder=ZORDER[m])
                ax.set_xlabel(rf"$q_{{{j+1}}}$"); ax.set_ylabel(rf"$p_{{{j+1}}}$"); ax.set_box_aspect(1)   # phase-plane panels square
            ax.set_title(f"({chr(97 + i)})", loc="left", pad=3)
        hd, lb = axes[0][0].get_legend_handles_labels()   # panel grid: shared one-row legend centred above the grid, no frame
        fig.legend(hd, lb, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(lb), frameon=False, fontsize=HLEG["fontsize"], handlelength=HLEG["handlelength"], handletextpad=HLEG["handletextpad"], columnspacing=1.2)
        fig.savefig(Path("paper/figures") / f"hist_{eid}{suffix}.pdf"); fig.savefig(Path("paper/figures") / f"hist_{eid}{suffix}.png"); plt.close(fig)
        print(f"[{eid}] hist{suffix} written ({mode}; view seed {seed_view}, config {muidx[vidx]})")
    if grid_modes: return
    _hist_rows_figure([(sp, t, tr, V, w0, w1, sl, j) for j in coords], f"hist_{eid}", sp.get("hist_scale", HIST_SCALE))
    print(f"[{eid}] figures written (view seed {seed_view}, config {muidx[vidx]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["replay", "view", "sweep", "plot", "histmerge"]); ap.add_argument("ids", nargs="+")
    ap.add_argument("--name", default="hist_hh", help="output file name for histmerge"); ap.add_argument("--scale", type=float, default=0.7, help="histmerge scale relative to 5.4 in"); a = ap.parse_args()
    if a.cmd == "histmerge": hist_merged(a.ids, a.name, a.scale)
    else:
        for eid in a.ids: {"replay": replay, "view": replay_view, "sweep": sweep, "plot": plot}[a.cmd](eid)
