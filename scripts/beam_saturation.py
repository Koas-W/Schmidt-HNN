"""Beam: scale and saturation diagnostics. Does not modify any existing result file.

1. RMS scale A_i, B_i of every modal coordinate and momentum in the training data, and the max/min ratio of the action scales A_i B_i:
   a diagonal scaling preserves the canonical form only if a_i b_i is a common constant, so coordinate and momentum ranges can only be
   traded against each other; both are O(1) simultaneously iff the modal action scales are equal.
2. For each model (tanh baselines ahnn / symoden_mu / mlp_ode, and silu_hnn), on the interp-tier stored test states, the fraction of
   units per hidden layer with pre-activation |z|>2 (tanh'(2)=0.07); median and range over five seeds.
Output: results/beam_main/saturation.csv (scale: one row per mode; saturation: one row per model and layer).
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.models.factory import build_model  # noqa: E402
from src.data.dataset import load_dataset  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/beam_main.yaml")); mcfg = cfg["models"]
    data, _ = load_dataset(ROOT / "data_cache/clamped_beam")
    tr = np.asarray(data["traj_train"]); n = tr.shape[-1] // 2; X = tr.reshape(-1, 2 * n)
    Aq = np.sqrt((X[:, :n] ** 2).mean(0)); Ap = np.sqrt((X[:, n:] ** 2).mean(0)); J = Aq * Ap
    F = np.sqrt((np.asarray(data["deriv_train"]).reshape(-1, 2 * n)[:, n:] ** 2).mean(0))
    rows = [{"kind": "scale", "name": f"mode{i + 1}", "layer": "", "rms_q": Aq[i], "rms_p": Ap[i], "action": J[i], "rms_force": F[i]} for i in range(n)]
    print("RMS q:", np.round(Aq, 4), "\nRMS p:", np.round(Ap, 3), "\naction A_i B_i:", np.array2string(J, precision=3), f"  max/min = {J.max() / J.min():.3g}")
    print("RMS restoring force per mode:", np.round(F, 2))

    ti = torch.as_tensor(np.asarray(data["traj_interp"]), dtype=torch.float32); mu = torch.as_tensor(np.asarray(data["mu_interp"]), dtype=torch.float32)
    x = ti.reshape(ti.shape[0], -1, 2 * n)[:, ::7].reshape(-1, 2 * n); m = mu[:, None, :].expand(mu.shape[0], ti.shape[1] * ti.shape[2], mu.shape[-1])[:, ::7].reshape(-1, mu.shape[-1])
    print("states used for saturation:", x.shape[0])
    for name in ["ahnn", "symoden_mu", "mlp_ode", "silu_hnn"]:
        fr = []
        for s in range(5):
            model, _ = build_model(name, n, mu.shape[-1], mcfg, mcfg["target_params"])
            model.load_state_dict(torch.load(ROOT / f"results/beam_main/{s}/{name}.pt", map_location="cpu")); model.eval()
            pre = []; hooks = [mod.register_forward_hook(lambda mod, i, o: pre.append(o.detach())) for mod in model.modules() if isinstance(mod, torch.nn.Linear)]
            with torch.enable_grad(): model.time_derivative(x, m, create_graph=False)
            for h in hooks: h.remove()
            fr.append([float((p.abs() > 2).float().mean()) for p in pre if p.shape[-1] > 8])   # exclude the output layer
        fr = np.array(fr); med, lo, hi = np.median(fr, 0), fr.min(0), fr.max(0)
        print(f"{name:10s} hidden layers {fr.shape[1]}: fraction |z|>2 per layer, median over seeds = {np.round(med, 3)} (min {np.round(lo, 3)}, max {np.round(hi, 3)})")
        rows += [{"kind": "saturation", "name": name, "layer": l + 1, "frac_median": med[l], "frac_min": lo[l], "frac_max": hi[l]} for l in range(fr.shape[1])]
    out = ROOT / "results/beam_main/saturation.csv"; keys = sorted({k for r in rows for k in r}, key=lambda k: ["kind", "name", "layer"].index(k) if k in ("kind", "name", "layer") else 9)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
