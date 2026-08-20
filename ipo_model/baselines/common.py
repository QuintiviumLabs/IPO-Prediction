"""Shared fold loop for tree-model baselines.

Both baselines (LightGBM, XGBoost) run the same purged walk-forward folds,
per-fold scalers/bucketing, engineered feature table, and metrics as the deep
model — an engine plugs in a single fit-and-predict-quantiles function.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.data.preprocess import FoldScaler, engineered_table, raw_table
from ipo_model.data.splits import purged_walk_forward
from ipo_model.training.loop import FoldResult, RunResult
from ipo_model.training.metrics import aggregate_folds, evaluate, evaluate_binary

# (X_train, y_train, X_val, y_val, X_test, quantiles, mid_index) -> (n_test, Q)
QuantilePredictor = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame,
     list[float], int],
    np.ndarray,
]


TABLES = {"engineered": engineered_table, "raw": raw_table}

# Random-search space for the tree baselines, sized for ~1k training rows.
# A baseline left on library defaults is not a baseline — if the deep model
# only wins against an untuned competitor, it has not been shown to win.
LGBM_SPACE = {
    "num_leaves": [3, 7, 15, 31],
    "min_data_in_leaf": [5, 10, 20, 40],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "feature_fraction": [0.5, 0.7, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 20.0],
}
XGB_SPACE = {
    "max_leaves": [3, 7, 15, 31],
    "min_child_weight": [1, 5, 20, 50],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "subsample": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
}


def sample_configs(space: dict, n: int, seed: int = 0) -> list[dict]:
    """n distinct random draws from a discrete hyperparameter space."""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for _ in range(n * 20):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
        if len(out) == n:
            break
    return out


def run_folds(cfg: Config, fs: FeatureSet, predict_quantiles: QuantilePredictor,
              features: str = "engineered", verbose: bool = True) -> RunResult:
    """head='quantile': the predictor returns an (n_test, Q) quantile matrix.
    head='binary': the predictor returns an (n_test, 1) probability of
    outperformance (it still receives the CONTINUOUS y and derives y > 0
    itself, so it can also use magnitudes for tuning if it wants)."""
    binary = cfg.model.head == "binary"
    table_fn = TABLES[features]
    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    y = fs.y[cfg.main_horizon]
    qs = [0.5] if binary else sorted(cfg.model.quantiles)
    mid = qs.index(0.5)
    results: list[FoldResult] = []

    for k, fold in enumerate(folds):
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        X = table_fn(fs, scaler)
        q_pred = predict_quantiles(
            X.iloc[fold.train_idx], y[fold.train_idx],
            X.iloc[fold.val_idx], y[fold.val_idx],
            X.iloc[fold.test_idx], qs, mid,
        )
        q_pred = np.atleast_2d(np.asarray(q_pred, float))
        if not binary:
            q_pred = np.sort(q_pred, axis=1)  # enforce monotone quantiles

        m = (evaluate_binary(y[fold.test_idx], q_pred[:, 0]) if binary
             else evaluate(y[fold.test_idx], q_pred, cfg.model.quantiles))
        results.append(FoldResult(fold=k, test_idx=fold.test_idx, q_pred=q_pred,
                                  per_seed_val_loss=[], metrics=m))
        if verbose and binary:
            print(f"  fold {k}: n_test={len(fold.test_idx)} "
                  f"IC={m['rank_ic']:+.3f} AUC={m['auc']:.3f} "
                  f"acc={m['accuracy']:.3f} brier={m['brier']:.4f} "
                  f"base={m['base_rate']:.2f}")
        elif verbose:
            print(f"  fold {k}: n_test={len(fold.test_idx)} "
                  f"IC={m['rank_ic']:+.3f} hit={m['hit_rate']:.3f} "
                  f"spread={m['decile_spread']:+.4f} MAE={m['mae']:.4f} "
                  f"cov={m['coverage']:.2f}")

    pooled_y = np.concatenate([y[r.test_idx] for r in results])
    pooled_q = np.concatenate([r.q_pred for r in results])
    return RunResult(
        fold_results=results,
        summary=aggregate_folds([r.metrics for r in results]),
        pooled=(evaluate_binary(pooled_y, pooled_q[:, 0]) if binary
                else evaluate(pooled_y, pooled_q, cfg.model.quantiles)),
    )
