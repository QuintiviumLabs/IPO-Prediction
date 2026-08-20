"""Ablation 0: LightGBM on engineered features.

Same purged walk-forward folds, targets and metrics as the deep model. This
is the bar the three-arm architecture has to clear: if the LSTM arms can't
beat hand-built summaries of the same information, the sequences are not
carrying incremental signal.

Point forecasts come from a Huber-objective model; the 10/90 interval from
quantile-objective models, so the baseline is comparable on pinball loss and
coverage too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_model.baselines.common import LGBM_SPACE, run_folds, sample_configs
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult


def run(cfg: Config, fs: FeatureSet, seed: int = 0, features: str = "engineered",
        tune: int = 0, verbose: bool = True) -> RunResult:
    """tune=N runs an N-config random search per fold, selected on the purged
    validation slice. Use it before claiming the deep model wins: beating an
    untuned baseline proves nothing."""
    import lightgbm as lgb

    def _fit(params: dict, X_train, y_train, X_val, y_val):
        return lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=cfg.lgbm.num_boost_round,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[lgb.early_stopping(cfg.lgbm.early_stopping_rounds,
                                          verbose=False)],
        )

    def predict_binary(X_train: pd.DataFrame, y_train: np.ndarray,
                       X_val: pd.DataFrame, y_val: np.ndarray,
                       X_test: pd.DataFrame, qs: list[float],
                       mid: int) -> np.ndarray:
        """Binary head: same engine, objective='binary', probability out."""
        lt, lv = (y_train > 0).astype(float), (y_val > 0).astype(float)
        base = dict(cfg.lgbm.params, seed=seed, objective="binary",
                    metric="binary_logloss")
        base.pop("alpha", None)
        base.pop("huber_slope", None)
        if cfg.train.class_weight == "balanced":
            base["is_unbalance"] = True   # pop-heavy base rate
        if tune:
            best_err, best = np.inf, base
            for cand in sample_configs(LGBM_SPACE, tune, seed=seed):
                params = dict(base, **cand)
                b = _fit(params, X_train, lt, X_val, lv)
                p = b.predict(X_val, num_iteration=b.best_iteration)
                p = np.clip(p, 1e-7, 1 - 1e-7)
                err = float(-(lv * np.log(p) + (1 - lv) * np.log(1 - p)).mean())
                if err < best_err:
                    best_err, best = err, params
            base = best
            if verbose:
                shown = {k: base[k] for k in LGBM_SPACE if k in base}
                print(f"    tuned ({tune} configs) val logloss={best_err:.4f}: {shown}")
        booster = _fit(base, X_train, lt, X_val, lv)
        return booster.predict(X_test,
                               num_iteration=booster.best_iteration)[:, None]

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        base = dict(cfg.lgbm.params, seed=seed)
        if tune:
            best_err, best = np.inf, base
            for cand in sample_configs(LGBM_SPACE, tune, seed=seed):
                params = dict(base, **cand)
                b = _fit(params, X_train, y_train, X_val, y_val)
                err = float(np.abs(b.predict(X_val, num_iteration=b.best_iteration)
                                   - y_val).mean())
                if err < best_err:
                    best_err, best = err, params
            base = best
            if verbose:
                shown = {k: base[k] for k in LGBM_SPACE if k in base}
                print(f"    tuned ({tune} configs) val MAE={best_err:.4f}: {shown}")

        q_pred = np.empty((len(X_test), len(qs)))
        for qi, q in enumerate(qs):
            params = dict(base)
            if qi != mid:  # median slot uses the configured point objective (Huber)
                params.update({"objective": "quantile", "alpha": q})
            booster = _fit(params, X_train, y_train, X_val, y_val)
            q_pred[:, qi] = booster.predict(X_test, num_iteration=booster.best_iteration)
        return q_pred

    predictor = (predict_binary if cfg.model.head == "binary"
                 else predict_quantiles)
    return run_folds(cfg, fs, predictor, features=features, verbose=verbose)
