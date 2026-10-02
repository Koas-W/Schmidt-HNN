"""Training loop: minibatch AdamW + validation early stopping (one protocol, applied to all models alike).

Fairness ingredients:
- same optimizer (AdamW), same learning-rate grid (outer sweep picks the best on validation),
  same training budget (max_steps) and early-stopping rule, same data and seed set;
- the validation set is the val-role trajectories fixed at generation time, shared by all models.
"""
import copy
import math
import random

import numpy as np
import torch

from .loss import derivative_matching_loss
from .muon import MuonAdamW


def build_optimizer(model, tcfg: dict, lr, capturable: bool):
    """tcfg["optimizer"]: adamw (default) or muon (Muon for 2D parameters + AdamW for the rest)."""
    wd = tcfg.get("weight_decay", 1e-4)
    if tcfg.get("optimizer", "adamw") == "muon":
        return MuonAdamW(model.named_parameters(), lr=lr, weight_decay=wd,
                         momentum=tcfg.get("muon_momentum", 0.95), capturable=capturable)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                             capturable=capturable)


def _reset_opt_state(opt):
    if hasattr(opt, "reset_state"):
        opt.reset_state(); return
    for st in opt.state.values():
        st["step"].zero_(); st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()


def _set_lr(opt, value: float):
    if hasattr(opt, "set_lr"):
        opt.set_lr(value); return
    for grp in opt.param_groups:
        grp["lr"] = value


def lr_at(step: int, tcfg: dict) -> float:
    """Learning-rate schedule (identical for all models; an optimizer setting, does not touch data/physical coordinates).
    constant (default): always tcfg["lr"];
    cosine: cosine decay from lr to lr_min (default lr*1e-2), optional linear warmup_steps."""
    lr = tcfg["lr"]
    sched = tcfg.get("lr_schedule", "constant")
    if sched == "constant":
        return lr
    if sched != "cosine":
        raise ValueError(f"unknown lr_schedule: {sched}")
    warm = int(tcfg.get("warmup_steps", 0))
    if step < warm:
        return lr * (step + 1) / warm
    lr_min = tcfg.get("lr_min", lr * 1e-2)
    T = max(1, tcfg["max_steps"] - warm)
    frac = min(1.0, (step - warm) / T)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * frac))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _to(device, *tensors):
    return [t.to(device) for t in tensors]


def evaluate_loss(model, view: dict, device, chunk: int = 65536) -> float:
    """Validation/test derivative MSE (chunked to avoid building a huge graph)."""
    x, mu, dx = view["x"], view["mu"], view["dx"]
    total, count = 0.0, 0
    for i in range(0, x.shape[0], chunk):
        xb, mub, dxb = _to(device, x[i:i + chunk], mu[i:i + chunk],
                           dx[i:i + chunk])
        pred = model.time_derivative(xb, mub, create_graph=False)
        total += torch.nn.functional.mse_loss(
            pred, dxb, reduction="sum").item()
        count += dxb.numel()
    return total / count


def _train_model_cuda_graph(model, train_view: dict, val_view: dict,
                            tcfg: dict, seed: int,
                            device: torch.device) -> tuple[dict, list]:
    """Whole-step CUDA Graph captured training (measured 6.2x speed-up; per-step loss differs from
    eager by a relative <1e-6).

    Capture requirements and trade-offs: fixed batch size (drop-last, which differs from the tail-batch
    semantics of the eager version; a protocol difference the caller must be aware of); AdamW(capturable=True);
    warm-up contamination is removed by reloading parameters and zeroing momentum/step counters; the loss is read
    from a static buffer and synchronized to CPU only at validation points. The evaluation path stays eager.
    """
    set_seed(seed)
    model.to(device)
    bs = min(tcfg.get("batch_size", 8192), train_view["x"].shape[0])
    x = train_view["x"].to(device)
    mu = train_view["mu"].to(device)
    dx = train_view["dx"].to(device)
    N = x.shape[0]
    lr_t = torch.tensor(tcfg["lr"], device=device)   # tensor lr: modifiable in place after capture (scheduling)
    opt = build_optimizer(model, tcfg, lr_t, capturable=True)
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}
    sx = torch.zeros(bs, x.shape[1], device=device)
    smu = torch.zeros(bs, mu.shape[1], device=device)
    sdx = torch.zeros(bs, dx.shape[1], device=device)
    sloss = torch.zeros((), device=device)

    def body():
        opt.zero_grad(set_to_none=False)
        pred = model.time_derivative(sx, smu, create_graph=True)
        loss = torch.nn.functional.mse_loss(pred, sdx)
        loss.backward()
        opt.step()
        sloss.copy_(loss.detach())

    g = torch.Generator().manual_seed(seed)
    warm_idx = torch.randperm(N, generator=g)[:bs].to(device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sx.copy_(x[warm_idx]); smu.copy_(mu[warm_idx])
            sdx.copy_(dx[warm_idx])
            body()
    torch.cuda.current_stream().wait_stream(stream)
    model.load_state_dict(ref_state)
    _reset_opt_state(opt)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    # training loop: same early-stopping protocol as the eager version; batch sequence is drop-last
    g = torch.Generator().manual_seed(seed)  # sharing the seed sequence with warm-up is harmless: indices only
    best_val, best_state, bad, history = math.inf, None, 0, []
    eval_every = tcfg.get("eval_every", 100)
    patience = tcfg.get("patience", 10)
    early_stop = tcfg.get("early_stop", True)  # False = run the full budget (no early stop); the best-val checkpoint is still kept
    step, max_steps = 0, tcfg["max_steps"]
    while step < max_steps:
        perm = torch.randperm(N, generator=g)
        for i in range(0, N - bs + 1, bs):
            idx = perm[i:i + bs].to(device)
            sx.copy_(x[idx]); smu.copy_(mu[idx]); sdx.copy_(dx[idx])
            lr_t.fill_(lr_at(step, tcfg))
            graph.replay()
            step += 1
            if step % eval_every == 0 or step >= max_steps:
                model.eval()
                val = evaluate_loss(model, val_view, device)
                model.train()
                history.append((step, sloss.item(), val))
                if val < best_val * (1 - 1e-4):
                    best_val, bad = val, 0
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in
                         model.state_dict().items()})
                else:
                    bad += 1
                if (early_stop and bad >= patience) or step >= max_steps:
                    if best_state is None:
                        best_state = copy.deepcopy(
                            {k: v.detach().cpu() for k, v in
                             model.state_dict().items()})
                    return best_state, history
    if best_state is None:
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in
                                    model.state_dict().items()})
    return best_state, history


def train_model(model, train_view: dict, val_view: dict, tcfg: dict,
                seed: int, device: torch.device) -> tuple[dict, list]:
    """Single training run (fixed lr); returns (state_dict at best val, history).

    history rows: (step, train_loss_minibatch, val_loss).
    With tcfg["cuda_graph"]=True on a CUDA device the whole-step captured path is used (6.2x).
    """
    if tcfg.get("cuda_graph") and device.type == "cuda":
        return _train_model_cuda_graph(model, train_view, val_view, tcfg,
                                       seed, device)
    set_seed(seed)
    model.to(device)
    opt = build_optimizer(model, tcfg, tcfg["lr"], capturable=False)
    x, mu, dx = train_view["x"], train_view["mu"], train_view["dx"]
    N = x.shape[0]
    bs = min(tcfg.get("batch_size", 8192), N)
    g = torch.Generator().manual_seed(seed)

    best_val, best_state, bad, history = math.inf, None, 0, []
    eval_every = tcfg.get("eval_every", 100)
    patience = tcfg.get("patience", 10)
    early_stop = tcfg.get("early_stop", True)  # False = run the full budget (no early stop); the best-val checkpoint is still kept
    step, max_steps = 0, tcfg["max_steps"]
    while step < max_steps:
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb, mub, dxb = _to(device, x[idx], mu[idx], dx[idx])
            _set_lr(opt, lr_at(step, tcfg))
            opt.zero_grad(set_to_none=True)
            loss = derivative_matching_loss(model, xb, mub, dxb)
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step >= max_steps:
                model.eval()
                val = evaluate_loss(model, val_view, device)
                model.train()
                history.append((step, loss.item(), val))
                if val < best_val * (1 - 1e-4):
                    best_val, bad = val, 0
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in
                         model.state_dict().items()})
                else:
                    bad += 1
                if (early_stop and bad >= patience) or step >= max_steps:
                    if best_state is None:
                        best_state = copy.deepcopy(
                            {k: v.detach().cpu() for k, v in
                             model.state_dict().items()})
                    return best_state, history
    if best_state is None:
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in
                                    model.state_dict().items()})
    return best_state, history


def train_with_lr_sweep(build_fn, train_view, val_view, tcfg: dict,
                        seed: int, device) -> tuple[object, dict]:
    """Train once per lr in lr_grid (same seed and data) and keep the best on validation (common protocol).

    build_fn() -> freshly initialized model (set_seed is called first so every lr gets the same init).
    Returns (best_model, info).
    """
    best = None
    histories = {}
    for lr in tcfg["lr_grid"]:
        set_seed(seed)
        model = build_fn()
        state, history = train_model(model, train_view, val_view,
                                     {**tcfg, "lr": lr}, seed, device)
        histories[f"{lr:g}"] = history
        model.load_state_dict(state)
        model.to(device).eval()
        val = evaluate_loss(model, val_view, device)
        if best is None or val < best[1]:
            best = (model, val, lr, history)
    model, val, lr, history = best
    return model, {"chosen_lr": lr, "best_val": val,
                   "n_evals": len(history), "history": history,
                   "histories": histories}
