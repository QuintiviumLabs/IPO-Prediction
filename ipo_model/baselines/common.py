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
from ipo_model.data.preprocess import FoldScaler, engineered_table
from ipo_model.data.splits import purged_walk_forward
from ipo_model.training.loop import FoldResult, RunResult
from ipo_model.training.metrics import aggregate_folds, evaluate

# (X_train, y_train, X_val, y_val, X_test, quantiles, mid_index) -> (n_test, Q)
QuantilePredictor = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame,
     list[float], int],
    np.ndarray,
]


def run_folds(cfg: Config, fs: FeatureSet, predict_quantiles: QuantilePredictor,
              verbose: bool = True) -> RunResult:
    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    y = fs.y[cfg.main_horizon]
    qs = sorted(cfg.model.quantiles)
    mid = qs.index(0.5)
    results: list[FoldResult] = []

    for k, fold in enumerate(folds):
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        X = engineered_table(fs, scaler)
        q_pred = predict_quantiles(
            X.iloc[fold.train_idx], y[fold.train_idx],
            X.iloc[fold.val_idx], y[fold.val_idx],
            X.iloc[fold.test_idx], qs, mid,
        )
        q_pred = np.sort(q_pred, axis=1)  # enforce monotone quantiles

        m = evaluate(y[fold.test_idx], q_pred, cfg.model.quantiles)
        results.append(FoldResult(fold=k, test_idx=fold.test_idx, q_pred=q_pred,
                                  per_seed_val_loss=[], metrics=m))
        if verbose:
            print(f"  fold {k}: n_test={len(fold.test_idx)} "
                  f"IC={m['rank_ic']:+.3f} hit={m['hit_rate']:.3f} "
                  f"spread={m['decile_spread']:+.4f} MAE={m['mae']:.4f} "
                  f"cov={m['coverage']:.2f}")

    pooled_y = np.concatenate([y[r.test_idx] for r in results])
    pooled_q = np.concatenate([r.q_pred for r in results])
    return RunResult(
        fold_results=results,
        summary=aggregate_folds([r.metrics for r in results]),
        pooled=evaluate(pooled_y, pooled_q, cfg.model.quantiles),
    )
