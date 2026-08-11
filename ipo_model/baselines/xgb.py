"""Ablation 0 (second engine): XGBoost on the same engineered features.

A twin of the LightGBM baseline — identical folds, feature table, and
metrics — as a robustness check that baseline conclusions aren't an artifact
of one tree implementation. Point forecasts use a pseudo-Huber objective;
the 10/90 interval uses XGBoost's native quantile objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_model.baselines.common import run_folds
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult


def run(cfg: Config, fs: FeatureSet, seed: int = 0, verbose: bool = True) -> RunResult:
    import xgboost as xgb

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        q_pred = np.empty((len(X_test), len(qs)))
        for qi, q in enumerate(qs):
            params = dict(cfg.xgb.params, seed=seed)
            if qi != mid:  # median slot uses the configured point objective
                params.update({"objective": "reg:quantileerror", "quantile_alpha": q})
            booster = xgb.train(
                params, dtrain,
                num_boost_round=cfg.xgb.num_boost_round,
                evals=[(dval, "val")],
                early_stopping_rounds=cfg.xgb.early_stopping_rounds,
                verbose_eval=False,
            )
            q_pred[:, qi] = booster.predict(
                dtest, iteration_range=(0, booster.best_iteration + 1))
        return q_pred

    return run_folds(cfg, fs, predict_quantiles, verbose=verbose)
