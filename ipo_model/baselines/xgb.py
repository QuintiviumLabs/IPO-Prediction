"""Ablation 0 (second engine): XGBoost on the same engineered features.

A twin of the LightGBM baseline — identical folds, feature table, and
metrics — as a robustness check that baseline conclusions aren't an artifact
of one tree implementation. Point forecasts use a pseudo-Huber objective;
the 10/90 interval uses XGBoost's native quantile objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_model.baselines.common import XGB_SPACE, run_folds, sample_configs
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult


def run(cfg: Config, fs: FeatureSet, seed: int = 0, features: str = "engineered",
        tune: int = 0, verbose: bool = True) -> RunResult:
    """tune=N runs an N-config random search per fold, selected on the purged
    validation slice (see lgbm.run for why this matters)."""
    import xgboost as xgb

    def predict_binary(X_train: pd.DataFrame, y_train: np.ndarray,
                       X_val: pd.DataFrame, y_val: np.ndarray,
                       X_test: pd.DataFrame, qs: list[float],
                       mid: int) -> np.ndarray:
        """Binary head: objective binary:logistic, probability out."""
        lt, lv = (y_train > 0).astype(float), (y_val > 0).astype(float)
        dtrain = xgb.DMatrix(X_train, label=lt)
        dval = xgb.DMatrix(X_val, label=lv)
        dtest = xgb.DMatrix(X_test)

        def _fit(params):
            return xgb.train(params, dtrain,
                             num_boost_round=cfg.xgb.num_boost_round,
                             evals=[(dval, "val")],
                             early_stopping_rounds=cfg.xgb.early_stopping_rounds,
                             verbose_eval=False)

        base = dict(cfg.xgb.params, seed=seed, objective="binary:logistic",
                    eval_metric="logloss")
        if cfg.train.class_weight == "balanced":
            pos = max(float(lt.sum()), 1.0)
            base["scale_pos_weight"] = (len(lt) - lt.sum()) / pos
        base.pop("huber_slope", None)
        if tune:
            best_err, best = np.inf, base
            for cand in sample_configs(XGB_SPACE, tune, seed=seed):
                params = dict(base, **cand)
                b = _fit(params)
                p = np.clip(b.predict(dval, iteration_range=(0, b.best_iteration + 1)),
                            1e-7, 1 - 1e-7)
                err = float(-(lv * np.log(p) + (1 - lv) * np.log(1 - p)).mean())
                if err < best_err:
                    best_err, best = err, params
            base = best
            if verbose:
                shown = {k: base[k] for k in XGB_SPACE if k in base}
                print(f"    tuned ({tune} configs) val logloss={best_err:.4f}: {shown}")
        booster = _fit(base)
        return booster.predict(
            dtest, iteration_range=(0, booster.best_iteration + 1))[:, None]

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        def _fit(params):
            return xgb.train(params, dtrain,
                             num_boost_round=cfg.xgb.num_boost_round,
                             evals=[(dval, "val")],
                             early_stopping_rounds=cfg.xgb.early_stopping_rounds,
                             verbose_eval=False)

        base = dict(cfg.xgb.params, seed=seed)
        if tune:
            best_err, best = np.inf, base
            for cand in sample_configs(XGB_SPACE, tune, seed=seed):
                params = dict(base, **cand)
                b = _fit(params)
                pred = b.predict(dval, iteration_range=(0, b.best_iteration + 1))
                err = float(np.abs(pred - y_val).mean())
                if err < best_err:
                    best_err, best = err, params
            base = best
            if verbose:
                shown = {k: base[k] for k in XGB_SPACE if k in base}
                print(f"    tuned ({tune} configs) val MAE={best_err:.4f}: {shown}")

        q_pred = np.empty((len(X_test), len(qs)))
        for qi, q in enumerate(qs):
            params = dict(base)
            if qi != mid:  # median slot uses the configured point objective
                params.update({"objective": "reg:quantileerror", "quantile_alpha": q})
            booster = _fit(params)
            q_pred[:, qi] = booster.predict(
                dtest, iteration_range=(0, booster.best_iteration + 1))
        return q_pred

    predictor = (predict_binary if cfg.model.head == "binary"
                 else predict_quantiles)
    return run_folds(cfg, fs, predictor, features=features, verbose=verbose)
