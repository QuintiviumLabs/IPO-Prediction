"""Naive floor rungs — the level any real model must clear.

Without these, "rank IC 0.08" has no reference point. Two strategies:

  constant   predict the training-fold median outperformance for every deal.
             Has no ordering at all (rank IC is undefined by construction),
             so it exists to floor the DISTRIBUTION metrics: any pinball loss
             or interval score worse than this means the model is adding
             nothing but noise to the unconditional distribution.

  feature    a univariate regression of the target on ONE engineered factor
             (default f1_med_1w, recent same-market IPO performance). This is
             the floor that actually matters: does the whole three-arm model
             beat simply extrapolating the single most obvious signal? A
             reviewer will ask; better to have the number.

Both run through the shared fold harness, so they are directly comparable to
every other rung.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_model.baselines.common import run_folds
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult

DEFAULT_FEATURE = "f1_med_1w"


def _univariate_logistic(x: np.ndarray, label: np.ndarray,
                         iters: int = 25) -> tuple[float, float]:
    """(alpha, beta) of P(label=1) = sigmoid(alpha + beta*x) via Newton/IRLS."""
    a, b = 0.0, 0.0
    for _ in range(iters):
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = np.clip(p * (1 - p), 1e-6, None)
        g0, g1 = float((p - label).sum()), float(((p - label) * x).sum())
        h00, h01 = float(w.sum()), float((w * x).sum())
        h11 = float((w * x * x).sum())
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a, b = a - da, b - db
        if max(abs(da), abs(db)) < 1e-10:
            break
    return a, b


def run(cfg: Config, fs: FeatureSet, strategy: str = "feature",
        feature: str = DEFAULT_FEATURE, features: str = "engineered",
        verbose: bool = True) -> RunResult:
    if strategy not in ("constant", "feature"):
        raise ValueError("strategy must be 'constant' or 'feature'")

    def predict_binary(X_train: pd.DataFrame, y_train: np.ndarray,
                       X_val: pd.DataFrame, y_val: np.ndarray,
                       X_test: pd.DataFrame, qs: list[float],
                       mid: int) -> np.ndarray:
        label = (y_train > 0).astype(float)
        if strategy == "constant":
            # The training base rate for every deal: no ordering; floors the
            # probability-quality metrics (Brier / log loss).
            return np.full((len(X_test), 1), float(label.mean()))
        if feature not in X_train.columns:
            raise KeyError(f"feature {feature!r} not in the engineered table")
        xt = X_train[feature].to_numpy(float)
        mu, sd = xt.mean(), xt.std() + 1e-12
        a, b = _univariate_logistic((xt - mu) / sd, label)
        if verbose:
            print(f"    {feature}: logit beta={b:+.4f} (per 1 sd)")
        z = a + b * (X_test[feature].to_numpy(float) - mu) / sd
        return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))[:, None]

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        if strategy == "constant":
            point_val = np.full(len(X_val), float(np.median(y_train)))
            point_test = np.full(len(X_test), float(np.median(y_train)))
        else:
            if feature not in X_train.columns:
                raise KeyError(
                    f"feature {feature!r} not in the engineered table; "
                    f"available: {sorted(X_train.columns)[:12]}…")
            xt = X_train[feature].to_numpy(float)
            mu, sd = xt.mean(), xt.std() + 1e-12
            z_tr = (xt - mu) / sd
            # Univariate OLS with an intercept — keeps predictions on the
            # target's scale so pinball/MAE stay comparable to other rungs.
            beta = float(np.cov(z_tr, y_train, ddof=0)[0, 1] / max(np.var(z_tr), 1e-12))
            alpha = float(y_train.mean() - beta * z_tr.mean())
            point_val = alpha + beta * (X_val[feature].to_numpy(float) - mu) / sd
            point_test = alpha + beta * (X_test[feature].to_numpy(float) - mu) / sd
            if verbose:
                print(f"    {feature}: beta={beta:+.4f} (per 1 sd)")

        # Band from empirical validation residuals, centred on the median.
        offsets = np.quantile(y_val - point_val, qs)
        offsets = offsets - offsets[mid]
        return point_test[:, None] + offsets[None, :]

    predictor = (predict_binary if cfg.model.head == "binary"
                 else predict_quantiles)
    return run_folds(cfg, fs, predictor, features=features, verbose=verbose)
