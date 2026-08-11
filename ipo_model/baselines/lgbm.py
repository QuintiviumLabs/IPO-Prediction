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

from ipo_model.baselines.common import run_folds
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult


def run(cfg: Config, fs: FeatureSet, seed: int = 0, features: str = "engineered",
        verbose: bool = True) -> RunResult:
    import lightgbm as lgb

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        q_pred = np.empty((len(X_test), len(qs)))
        for qi, q in enumerate(qs):
            params = dict(cfg.lgbm.params, seed=seed)
            if qi != mid:  # median slot uses the configured point objective (Huber)
                params.update({"objective": "quantile", "alpha": q})
            booster = lgb.train(
                params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=cfg.lgbm.num_boost_round,
                valid_sets=[lgb.Dataset(X_val, label=y_val)],
                callbacks=[lgb.early_stopping(cfg.lgbm.early_stopping_rounds,
                                              verbose=False)],
            )
            q_pred[:, qi] = booster.predict(X_test, num_iteration=booster.best_iteration)
        return q_pred

    return run_folds(cfg, fs, predict_quantiles, features=features, verbose=verbose)
