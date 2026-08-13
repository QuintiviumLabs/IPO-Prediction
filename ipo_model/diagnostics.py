"""Feature and failure diagnostics.

Three complementary instruments, all on the same purged walk-forward folds:

- deep_permutation_importance: train the deep model per fold, then shuffle one
  named feature (or feature group) across the test block and measure the drop
  in rank IC. Near-zero drop = dead feature; negative drop (IC improves when
  shuffled) = the feature is actively harmful — likely fitted noise.
- tree_gain_importance: LightGBM gain importance per engineered feature,
  averaged across folds. Cheap; disagreement with permutation importance is
  itself informative (information present but badly encoded by the net).
- slice_metrics: out-of-sample rank IC / MAE by market, sector, GPR regime
  and momentum-drought status — turns "underperforms" into "underperforms
  WHERE", which is what makes it fixable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.data.preprocess import FoldScaler, engineered_table
from ipo_model.data.splits import purged_walk_forward
from ipo_model.training.loop import RunResult, _slice, _tensors, train_one
from ipo_model.training.metrics import evaluate


def _feature_groups(cfg: Config, fs: FeatureSet) -> dict[str, tuple]:
    """name -> (tensor_key, column indices | None for the whole tensor)."""
    groups: dict[str, tuple] = {}
    if cfg.model.use_momentum:
        mom_names = [n for n in fs.momentum_names
                     if n.split("_")[0] in cfg.model.momentum_groups]
        for j, n in enumerate(mom_names):
            groups[n] = ("momentum", [j])
    binary_names = list(fs.sector_names)
    if cfg.data.include_market_onehot:
        binary_names += [f"mkt_{m}" for m in fs.market_names]
    binary_names += ["n_bookrunners"]
    for j, n in enumerate(binary_names):
        groups[n] = ("static_binary", [j])
    groups["bookrunners"] = ("static_bk", None)
    if cfg.model.use_gpr:
        groups["gpr"] = ("__gpr__", None)  # gpr_seq + gpr_feats jointly
    return groups


def _permute(test_t: dict[str, torch.Tensor], key: str, cols,
             perm: torch.Tensor) -> dict[str, torch.Tensor]:
    pt = {k: v.clone() for k, v in test_t.items()}
    if key == "__gpr__":
        pt["gpr_seq"] = pt["gpr_seq"][perm]
        pt["gpr_feats"] = pt["gpr_feats"][perm]
    elif cols is None:
        pt[key] = pt[key][perm]
    else:
        for c in cols:
            pt[key][:, c] = pt[key][perm, c]
    return pt


def deep_permutation_importance(cfg: Config, fs: FeatureSet, n_repeats: int = 3,
                                seed: int | None = None,
                                verbose: bool = True) -> pd.DataFrame:
    """Mean rank-IC drop (base − permuted) per feature across folds.
    Positive = the model relies on it; ~0 = dead; negative = harmful."""
    seed = cfg.train.seeds[0] if seed is None else seed
    device = cfg.train.device
    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    main_h = cfg.main_horizon
    groups = _feature_groups(cfg, fs)
    deltas: dict[str, list[float]] = {n: [] for n in groups}

    for k, fold in enumerate(folds):
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        tensors = _tensors(fs, scaler, cfg, device)
        targets = {h: torch.tensor(y, dtype=torch.float32, device=device)
                   for h, y in fs.y.items()}
        model, _ = train_one(cfg, tensors, targets, fold, seed)
        test_t = _slice(tensors, fold.test_idx)
        y_test = fs.y[main_h][fold.test_idx]
        with torch.no_grad():
            base = evaluate(y_test, model(test_t)[main_h].cpu().numpy(),
                            cfg.model.quantiles)["rank_ic"]
        gen = torch.Generator().manual_seed(seed + 1000 + k)
        for name, (key, cols) in groups.items():
            ics = []
            for _ in range(n_repeats):
                perm = torch.randperm(len(fold.test_idx), generator=gen)
                with torch.no_grad():
                    q = model(_permute(test_t, key, cols, perm))[main_h]
                ics.append(evaluate(y_test, q.cpu().numpy(),
                                    cfg.model.quantiles)["rank_ic"])
            deltas[name].append(base - float(np.mean(ics)))
        if verbose:
            print(f"  fold {k}: base IC {base:+.3f}, {len(groups)} features permuted")

    df = pd.DataFrame({
        "feature": list(deltas),
        "delta_ic": [float(np.mean(v)) for v in deltas.values()],
        "delta_ic_std": [float(np.std(v)) for v in deltas.values()],
    }).sort_values("delta_ic", ascending=False).reset_index(drop=True)
    return df


def tree_gain_importance(cfg: Config, fs: FeatureSet, seed: int = 0) -> pd.DataFrame:
    """LightGBM gain importance (share of total gain) averaged across folds."""
    import lightgbm as lgb

    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    y = fs.y[cfg.main_horizon]
    shares = []
    for fold in folds:
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        X = engineered_table(fs, scaler)
        booster = lgb.train(
            dict(cfg.lgbm.params, seed=seed),
            lgb.Dataset(X.iloc[fold.train_idx], label=y[fold.train_idx]),
            num_boost_round=cfg.lgbm.num_boost_round,
            valid_sets=[lgb.Dataset(X.iloc[fold.val_idx], label=y[fold.val_idx])],
            callbacks=[lgb.early_stopping(cfg.lgbm.early_stopping_rounds,
                                          verbose=False)],
        )
        gain = booster.feature_importance(importance_type="gain")
        shares.append(pd.Series(gain / max(gain.sum(), 1e-12), index=X.columns))
    imp = pd.concat(shares, axis=1)
    return pd.DataFrame({
        "feature": imp.index,
        "gain_share": imp.mean(axis=1).to_numpy(),
        "gain_share_std": imp.std(axis=1).to_numpy(),
    }).sort_values("gain_share", ascending=False).reset_index(drop=True)


def slice_metrics_from_predictions(cfg: Config, fs: FeatureSet,
                                   preds: pd.DataFrame) -> pd.DataFrame:
    """Slice metrics from a saved predictions CSV — no retraining.

    Rows are matched back to the feature set by ipo_id, so this works on any
    run whose predictions were persisted, including ones produced by a
    different model or an earlier session.
    """
    from ipo_model.results_io import quantile_columns

    pos = {i: k for k, i in enumerate(fs.ids)}
    keep = preds["ipo_id"].isin(pos).to_numpy()
    if not keep.all():
        preds = preds[keep]
    if preds.empty:
        raise ValueError("no saved predictions match the current feature set — "
                         "are these predictions from a different dataset?")
    idx = np.array([pos[i] for i in preds["ipo_id"]])
    q, levels = quantile_columns(preds)
    y = preds["y_true"].to_numpy(float)
    return _slices(cfg, fs, idx, q, y, levels)


def slice_metrics(cfg: Config, fs: FeatureSet, res: RunResult) -> pd.DataFrame:
    """Pooled OOS rank IC / MAE by slice, from an existing RunResult."""
    main_h = cfg.main_horizon
    idx = np.concatenate([r.test_idx for r in res.fold_results])
    q = np.concatenate([r.q_pred for r in res.fold_results])
    y = fs.y[main_h][idx]
    return _slices(cfg, fs, idx, q, y, cfg.model.quantiles)


def _slices(cfg: Config, fs: FeatureSet, idx: np.ndarray, q: np.ndarray,
            y: np.ndarray, quantiles) -> pd.DataFrame:

    slices: dict[str, np.ndarray] = {}
    for j, m in enumerate(fs.market_names):
        slices[f"market={m}"] = fs.market_onehot[idx, j] > 0
    tmt = fs.sector[idx, 0] > 0 if fs.sector.shape[1] > 0 else None
    hc = fs.sector[idx, 1] > 0 if fs.sector.shape[1] > 1 else None
    if tmt is not None:
        slices["sector=tmt"] = tmt
    if hc is not None:
        slices["sector=healthcare"] = hc
    if tmt is not None and hc is not None:
        slices["sector=other"] = ~tmt & ~hc
    level = fs.gpr_feats[idx, 0]
    terciles = np.quantile(level, [1 / 3, 2 / 3])
    slices["gpr=low"] = level <= terciles[0]
    slices["gpr=mid"] = (level > terciles[0]) & (level <= terciles[1])
    slices["gpr=high"] = level > terciles[1]
    if "f1_n_1d" in fs.momentum_names:
        n_deals = fs.momentum[idx, fs.momentum_names.index("f1_n_1d")]
        slices["momentum=drought(n=0)"] = n_deals == 0
        slices["momentum=active(n>0)"] = n_deals > 0

    rows = []
    for name, mask in slices.items():
        if mask.sum() < 20:
            continue
        m = evaluate(y[mask], q[mask], quantiles)
        rows.append({"slice": name, "n": int(mask.sum()), "rank_ic": m["rank_ic"],
                     "quintile_spread": m["quintile_spread"], "mae": m["mae"],
                     "coverage": m["coverage"], "coverage_error": m["coverage_error"]})
    return pd.DataFrame(rows)
