"""Beam dynamic load cases (non-autonomous Hamiltonian H = ½|p|² + V̂(q; μ(t)), or impact initial states): reference solution vs identified models.

Cases (--cases):
  res    near-resonant harmonic pressure: μ₃ = μ̄₃ + A sin(Ω t), Ω = ratio·ω₁,lin; initial state = static equilibrium at μ̄₃, at rest
  sweep  frequency sweep through the primary resonance: μ₃ = μ̄₃ + A sin θ(t), θ' = Ω(t) rising linearly from r₀·ω₁,lin to r₁·ω₁,lin; initial state as in res
  param  parametric excitation of the preload: μ₂ = μ̄₂ + Δ sin(2 ω₁,lin t), μ₃ = μ̄₃; initial state = static equilibrium + small first-mode displacement (0.02h)
  impact impact: constant μ (μ̄₃), initial state = static equilibrium + initial momentum I·φ(x₀) (impulse / initial-velocity equivalence)
  blast  uniform pressure impulse: initial state = static equilibrium + initial momentum I·f/f₁ (f = modal load vector of the pressure), mainly excites the first mode
Reference: the same non-autonomous RK4 as the ground truth (data dt, float64), cached; models: pshnn/silu_hnn of the given seed + sparse regression, CUDA Graph.
Staged output: peak and RMS errors are printed as soon as each model finishes, so separation can be judged early.
Usage: python3 scripts/beam_dyn_cases.py --mu12 -0.761 -0.025 --cases res sweep param impact --tag typical
"""
import argparse, csv, math, sys
from pathlib import Path
import torch, yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.models.factory import build_model  # noqa: E402
from src.systems import get_system  # noqa: E402
from src.systems.clamped_beam import SQRT12, mode_shapes  # noqa: E402


def rk4_timedep(f, y0, t0, dt, n_steps, sample_every, graphed=False):
    """Non-autonomous RK4: f(y, t); stages evaluated at t, t+dt/2, t+dt. With graphed=True the whole step is captured as a CUDA Graph."""
    y_s = y0.clone(); t_s = torch.full((y0.shape[0],), t0, dtype=y0.dtype, device=y0.device)

    def body():
        k1 = f(y_s, t_s)
        k2 = f(y_s + 0.5 * dt * k1, t_s + 0.5 * dt)
        k3 = f(y_s + 0.5 * dt * k2, t_s + 0.5 * dt)
        k4 = f(y_s + dt * k3, t_s + dt)
        y_s.copy_(y_s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)); t_s.add_(dt)

    ys = [y0.clone()]
    if graphed and y0.device.type == "cuda":
        y_bak, t_bak = y_s.clone(), t_s.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        y_s.copy_(y_bak); t_s.copy_(t_bak)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        y_s.copy_(y_bak); t_s.copy_(t_bak)
        step = g.replay
    else:
        step = body
    for i in range(1, n_steps + 1):
        step()
        if i % sample_every == 0:
            ys.append(y_s.clone())
    return torch.stack(ys, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="beam_main"); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=["pshnn", "silu_hnn"])
    ap.add_argument("--sindy", default="results/beam_sindy/coef_potential_dq4_dmu3.pt")
    ap.add_argument("--mu12", type=float, nargs=2, default=[-0.761, -0.025]); ap.add_argument("--mu3", type=float, default=0.35, help="base pressure μ̄₃")
    ap.add_argument("--amp3", type=float, default=0.15, help="μ₃ amplitude A for the harmonic/sweep cases")
    ap.add_argument("--cases", nargs="*", default=["res", "sweep", "param", "impact"])
    ap.add_argument("--omega-ratio", type=float, default=0.98); ap.add_argument("--sweep", type=float, nargs=2, default=[0.9, 1.3])
    ap.add_argument("--delta2", type=float, default=0.15, help="μ₂ amplitude Δ for parametric excitation"); ap.add_argument("--param-ratio", type=float, default=2.0, help="parametric excitation Ω/ω₁,lin")
    ap.add_argument("--impulse", type=float, default=60.0); ap.add_argument("--x0", type=float, default=0.3, help="impact location x₀/L")
    ap.add_argument("--n-periods", type=float, default=20.0); ap.add_argument("--tag", default="typical")
    ap.add_argument("--only-ref", action="store_true", help="reference solution only (to try out amplitudes)"); ap.add_argument("--ref-device", default="cpu")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    ecfg = yaml.safe_load(open(ROOT / f"configs/{a.exp}.yaml")); dcfg = yaml.safe_load(open(ROOT / ecfg["data_config"]))
    system = get_system(dcfg["system"]); n = system.dim; dt = dcfg["data"]["dt"]; se = 100
    T1 = 2 * math.pi / float(system.beta[0] ** 2); t_end = a.n_periods * T1; n_steps = int(round(t_end / dt)) // se * se
    m1, m2 = a.mu12; mu_base = torch.tensor([[m1, m2, a.mu3]])
    # first linearized frequency at the base static equilibrium (slave modes included)
    q_full = torch.zeros(1, system.n_total); q_full[:, :n] = system.static_equilibrium(mu_base); q_full[:, n:] = system.condense(q_full[:, :n], mu_base)
    w1 = float(torch.linalg.eigvalsh(system.hess_V_full(q_full, mu_base)[0]).min().sqrt())
    a_s = system.static_equilibrium(mu_base)[0]
    phi_mid = mode_shapes(system.beta, torch.tensor([0.5]), 0)[:n, 0]; phi_x0 = mode_shapes(system.beta, torch.tensor([a.x0]), 0)[:n, 0]
    w = lambda tr: (tr[..., :n] @ phi_mid) / SQRT12
    r0, r1 = a.sweep
    def mu_case(c, t):   # t: (B,) → (B,3)
        one = torch.ones_like(t)
        if c == "res": m3 = a.mu3 + a.amp3 * torch.sin(a.omega_ratio * w1 * t); m2t = m2 * one
        elif c == "sweep": th = w1 * (r0 * t + 0.5 * (r1 - r0) * t ** 2 / t_end); m3 = a.mu3 + a.amp3 * torch.sin(th); m2t = m2 * one
        elif c == "param": m3 = a.mu3 * one; m2t = m2 + a.delta2 * torch.sin(a.param_ratio * w1 * t)
        else: m3 = a.mu3 * one; m2t = m2 * one
        return torch.stack([m1 * one, m2t, m3], -1)
    def y0_case(c):
        q0 = a_s.clone(); p0 = torch.zeros(n)
        if c == "param": q0[0] += 0.02 * SQRT12 / float(phi_mid[0])      # seed displacement of 0.02h in the first mode
        if c == "impact": p0 = a.impulse * phi_x0
        if c == "blast": p0 = a.impulse * system.f[:n].to(torch.float64) / float(system.f[0])   # uniform-pressure impulse: p(0) ∝ modal load vector f, normalized by f₁
        return torch.cat([q0, p0])
    cases = list(a.cases); B = len(cases)
    y0 = torch.stack([y0_case(c) for c in cases])
    mu_of_t = lambda t: torch.stack([mu_case(c, t[j:j + 1])[0] for j, c in enumerate(cases)])   # t (B,) → (B,3)
    print(f"line ({m1:g}, {m2:g}), base μ₃={a.mu3:g}, ω₁,lin={w1:.3f}, T₁={T1:.4f}, {a.n_periods:g} periods; cases {cases}", flush=True)

    out = ROOT / "results" / a.exp / "dyn_cases"; out.mkdir(parents=True, exist_ok=True)
    key = f"{a.tag}_{'-'.join(cases)}_{a.n_periods:g}T1"
    rdev = torch.device(a.ref_device)
    def f_ref(y, t):
        mu = mu_of_t(t); qd, pd = system.vector_field(y[:, :n], y[:, n:], mu); return torch.cat([qd, pd], -1)
    cache = out / f"ref_{key}.pt"
    if cache.exists(): ref = torch.load(cache)
    else:
        with torch.no_grad(): ref = rk4_timedep(f_ref, y0.to(rdev), 0.0, dt, n_steps, se).cpu()
        torch.save(ref, cache)
    tt = torch.arange(ref.shape[0]) * dt * se; w_ref = w(ref)
    for j, c in enumerate(cases):
        wr = w_ref[:, j]; ipk = int(wr.abs().argmax()); tail = tt >= tt[-1] - 5 * T1
        print(f"[{c}] reference: peak w/h = {float(wr[ipk]):.3f} at {float(tt[ipk]) / T1:.2f} T1; last-5-period amplitude {float((wr[tail] - wr[tail].mean()).abs().max()):.3f}h", flush=True)
    if a.only_ref: return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); mcfg = ecfg["models"]; rows = []
    jobs = ["sindy"] + list(a.models)   # sparse regression first, to see separation early
    for name in jobs:
        if name == "sindy":
            from scripts.sindy_model import SindyPotentialModel
            m = SindyPotentialModel(str(ROOT / a.sindy), n)
        else:
            m, _ = build_model(name, n, system.mu_dim, mcfg, mcfg["target_params"]); m.load_state_dict(torch.load(ROOT / f"results/{a.exp}/{a.seed}/{name}.pt", map_location="cpu"))
        pc = out / f"pred_{key}_{name}_s{a.seed}.pt"
        if pc.exists(): pred = torch.load(pc)
        else:
            m = m.double().to(dev).eval()
            with torch.no_grad(): pred = rk4_timedep(lambda y, t, m=m: m.time_derivative(y, mu_of_t(t), create_graph=False), y0.to(dev), 0.0, dt, n_steps, se, graphed=True).cpu()
            torch.save(pred, pc)
        wp_all = w(pred)
        for j, c in enumerate(cases):
            wr, wp = w_ref[:, j], wp_all[:, j]; e = wp - wr; ipk = int(wr.abs().argmax()); ipm = int(wp.abs().argmax())
            base = float(w(torch.cat([a_s, torch.zeros(n)])[None]))
            pk_err = abs(float(wp[ipm]) - float(wr[ipk])) / abs(float(wr[ipk]) - base)
            tail = tt >= tt[-1] - 5 * T1; amp_r = float((wr[tail] - wr[tail].mean()).abs().max()); amp_p = float((wp[tail] - wp[tail].mean()).abs().max())
            rows.append({"line": a.tag, "case": c, "model": name, "rms_err_h": float(e.pow(2).mean().sqrt()), "max_err_h": float(e.abs().max()), "peak_ref_h": float(wr[ipk]), "peak_model_h": float(wp[ipm]),
                         "peak_rel_err_dyn": pk_err, "t_peak_ref_T1": float(tt[ipk]) / T1, "t_peak_model_T1": float(tt[ipm]) / T1, "tail_amp_ref_h": amp_r, "tail_amp_model_h": amp_p})
            print(f"  {name:9s} [{c}] RMS {float(e.pow(2).mean().sqrt()):.4f}h  max {float(e.abs().max()):.4f}h  peak {float(wp[ipm]):.3f}h vs {float(wr[ipk]):.3f}h ({pk_err:.2%})  tail amp {amp_p:.3f}/{amp_r:.3f}h", flush=True)
    with open(out / f"dyn_cases_{key}_s{a.seed}.csv", "w", newline="") as fh:
        wr_ = csv.DictWriter(fh, fieldnames=list(rows[0])); wr_.writeheader(); wr_.writerows(rows)
    import numpy as np
    np.savez(out / f"traj_{key}_s{a.seed}.npz", t=tt.numpy(), T1=T1, w1=w1, w_ref=w_ref.numpy(), mu=torch.stack([mu_of_t(tt[i:i + 1].expand(B)) for i in range(len(tt))]).numpy(),
             **{f"w_{name}": w(torch.load(out / f"pred_{key}_{name}_s{a.seed}.pt")).numpy() for name in jobs})
    print("written", out / f"dyn_cases_{key}_s{a.seed}.csv")


if __name__ == "__main__":
    main()
