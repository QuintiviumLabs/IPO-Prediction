"""Training and evaluation of the three-arm model over purged walk-forward folds.

Per fold: fit a FoldScaler on the training window, train one model per seed
with early stopping on the (purged) validation loss, average the quantile
forecasts across seeds (ensemble), evaluate the ensemble on the test block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.data.preprocess import FoldScaler
from ipo_model.data.splits import Fold, purged_walk_forward
from ipo_model.models.model import ThreeArmModel
from ipo_model.training.losses import multi_horizon_loss
from ipo_model.training.metrics import aggregate_folds, evaluate


@dataclass
class FoldResult:
    fold: int
    test_idx: np.ndarray
    q_pred: np.ndarray          # (n_test, Q) ensemble quantile forecasts, main horizon
    per_seed_val_loss: list[float]
    metrics: dict[str, float]


@dataclass
class RunResult:
    fold_results: list[FoldResult]
    summary: dict[str, tuple[float, float]]   # metric -> (mean, std) across folds
    pooled: dict[str, float]                  # metrics on all OOS rows pooled


def _tensors(fs: FeatureSet, scaler: FoldScaler, cfg: Config,
             device: str) -> dict[str, torch.Tensor]:
    binary, bk = scaler.static_matrix(fs)
    mom = scaler.momentum_matrix(fs)
    groups = cfg.model.momentum_groups
    keep = [i for i, n in enumerate(fs.momentum_names) if n.split("_")[0] in groups]
    t = {
        "static_binary": torch.tensor(binary),
        "static_bk": torch.tensor(bk),
        "momentum": torch.tensor(mom[:, keep]),
        "gpr_seq": torch.tensor(scaler.gpr_sequences(fs)),
        "gpr_feats": torch.tensor(scaler.gpr_features(fs)),
    }
    return {k: v.to(device) for k, v in t.items()}


def _slice(tensors: dict[str, torch.Tensor], idx: np.ndarray) -> dict[str, torch.Tensor]:
    ix = torch.as_tensor(idx, dtype=torch.long, device=next(iter(tensors.values())).device)
    return {k: v[ix] for k, v in tensors.items()}


def _build_model(cfg: Config, tensors: dict[str, torch.Tensor]) -> ThreeArmModel:
    return ThreeArmModel(
        cfg.model,
        n_binary=tensors["static_binary"].shape[1],
        n_bookrunners=tensors["static_bk"].shape[1],
        n_momentum=tensors["momentum"].shape[1],
        n_gpr_feats=tensors["gpr_feats"].shape[1],
        horizons=tuple(sorted(cfg.data.horizons)),
    )


def _wandb_or_none(cfg: Config):
    """Lazy wandb import, only when train.wandb=True. On a locked-down
    machine: set WANDB_MODE=offline and `wandb sync` later."""
    if not cfg.train.wandb:
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "train.wandb=True but wandb is not installed — "
            "pip install wandb (it's in requirements-extras.txt), "
            "or set train.wandb=False") from e
    return wandb


def train_one(cfg: Config, tensors: dict[str, torch.Tensor],
              targets: dict[int, torch.Tensor], fold: Fold,
              seed: int, epoch_log=None) -> tuple[ThreeArmModel, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.train.device
    model = _build_model(cfg, tensors).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    train_t = _slice(tensors, fold.train_idx)
    val_t = _slice(tensors, fold.val_idx)
    y_train = {h: t[fold.train_idx] for h, t in targets.items()}
    y_val = {h: t[fold.val_idx] for h, t in targets.items()}

    n = len(fold.train_idx)
    best_val, best_state, patience_left = float("inf"), None, cfg.train.patience
    for _epoch in range(cfg.train.max_epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, cfg.train.batch_size):
            b = perm[start: start + cfg.train.batch_size]
            bt = torch.as_tensor(b, dtype=torch.long, device=device)
            preds = model({k: v[bt] for k, v in train_t.items()})
            loss = multi_horizon_loss(
                preds, {h: y[bt] for h, y in y_train.items()},
                cfg.model.quantiles, cfg.main_horizon, cfg.model.aux_weight,
            )
            if cfg.model.l1_input > 0:  # structured feature pruning
                loss = loss + cfg.model.l1_input * model.input_l1()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(multi_horizon_loss(
                model(val_t), y_val,
                cfg.model.quantiles, cfg.main_horizon, cfg.model.aux_weight,
            ))
        if epoch_log is not None:
            epoch_log(_epoch, val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.train.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def run(cfg: Config, fs: FeatureSet, verbose: bool = True) -> RunResult:
    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    device = cfg.train.device
    main_h = cfg.main_horizon
    results: list[FoldResult] = []

    wb = _wandb_or_none(cfg)
    if wb is not None:
        import dataclasses as _dc
        wb.init(project=cfg.train.wandb_project, config=_dc.asdict(cfg),
                reinit=True)

    for k, fold in enumerate(folds):
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        tensors = _tensors(fs, scaler, cfg, device)
        # Optional rank-based label transform: fit on the TRAINING fold only,
        # train in Gaussian-score space (outlier magnitudes cannot pull the
        # body), back-map predictions to return space before evaluation.
        transform = None
        if cfg.train.label_transform == "normal_score":
            from ipo_model.training.transforms import NormalScore
            transform = {h: NormalScore().fit(y[fold.train_idx])
                         for h, y in fs.y.items()}
            targets = {h: torch.tensor(transform[h].transform(y),
                                       dtype=torch.float32, device=device)
                       for h, y in fs.y.items()}
        elif cfg.train.label_transform != "none":
            raise ValueError(f"Unknown label_transform: {cfg.train.label_transform}")
        else:
            targets = {h: torch.tensor(y, dtype=torch.float32, device=device)
                       for h, y in fs.y.items()}

        preds_per_seed, val_losses = [], []
        for seed in cfg.train.seeds:
            cb = None
            if wb is not None:
                def cb(e, v, _k=k, _s=seed):
                    wb.log({f"fold{_k}/seed{_s}/val_loss": v, "epoch": e})
            model, val_loss = train_one(cfg, tensors, targets, fold, seed,
                                        epoch_log=cb)
            with torch.no_grad():
                q = model(_slice(tensors, fold.test_idx))[main_h].cpu().numpy()
            preds_per_seed.append(q)
            val_losses.append(val_loss)

        q_ens = np.mean(preds_per_seed, axis=0)  # score space if transformed
        if transform is not None:
            q_ens = transform[main_h].inverse(q_ens)  # monotone -> no crossing
        y_test = fs.y[main_h][fold.test_idx]
        if cfg.model.l1_input > 0 and cfg.model.use_momentum and verbose:
            norms = model.momentum_input_norms()  # last seed's model
            names = [n for n in fs.momentum_names
                     if n.split("_")[0] in cfg.model.momentum_groups]
            dead = [n for n, v in zip(names, norms) if v < 0.02 * norms.max()]
            print(f"    l1_input pruned {len(dead)}/{len(names)} momentum "
                  f"features{': ' + ', '.join(dead) if dead else ''}")
        m = evaluate(y_test, q_ens, cfg.model.quantiles)
        results.append(FoldResult(fold=k, test_idx=fold.test_idx, q_pred=q_ens,
                                  per_seed_val_loss=val_losses, metrics=m))
        if wb is not None:
            wb.log({f"fold{k}/{name}": v for name, v in m.items()
                    if np.isfinite(v)} | {f"fold{k}/n_test": len(fold.test_idx)})
        if verbose:
            print(f"  fold {k}: n_test={len(fold.test_idx)} "
                  f"IC={m['rank_ic']:+.3f} hit={m['hit_rate']:.3f} "
                  f"spread={m['decile_spread']:+.4f} MAE={m['mae']:.4f} "
                  f"cov={m['coverage']:.2f}")

    pooled_y = np.concatenate([fs.y[main_h][r.test_idx] for r in results])
    pooled_q = np.concatenate([r.q_pred for r in results])
    out = RunResult(
        fold_results=results,
        summary=aggregate_folds([r.metrics for r in results]),
        pooled=evaluate(pooled_y, pooled_q, cfg.model.quantiles),
    )
    if wb is not None:
        for name, (mu, sd) in out.summary.items():
            if np.isfinite(mu):
                wb.run.summary[f"{name}_mean"] = mu
                wb.run.summary[f"{name}_std"] = sd
        for name, v in out.pooled.items():
            if np.isfinite(v):
                wb.run.summary[f"pooled_{name}"] = v
        wb.finish()
    return out
