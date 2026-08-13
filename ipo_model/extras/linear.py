"""Regularized linear rung: Ridge / Lasso / ElasticNet on engineered features.

At N ~ 1k with ~25 features this is arguably the right complexity class, and
it is the most defensible model in the ladder. If it matches the trees and
the net, that is a finding worth reporting: the relationship is approximately
linear in these factors.

Runs through `baselines.common.run_folds`, so it sees identical purged folds,
per-fold scalers, feature table and metrics as every other rung.

Features are standardized with TRAINING-fold statistics before fitting (the
engineered table is raw-scaled, and regularization is scale-sensitive). The
penalty strength is chosen per fold on the purged validation slice, never on
test.

Two ways to get the quantile band:
  residual  (default) fit the point model, then take empirical quantiles of
            its VALIDATION residuals as constant offsets. Fast, robust, and
            honest — but the band has the same width for every deal.
  quantreg  fit a separate L1-penalized linear quantile regression per level
            (sklearn QuantileRegressor). Gives deal-specific widths; slower,
            and the tail levels are noisy at small N.

    pip install scikit-learn
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_model.baselines.common import run_folds
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult

MODELS = ("ridge", "lasso", "elasticnet")
DEFAULT_ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The linear rung needs scikit-learn:\n"
            "    pip install scikit-learn\n"
            "(or: pip install -r requirements-extras.txt)"
        ) from e


def _make(model: str, alpha: float):
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    if model == "ridge":
        return Ridge(alpha=alpha)
    if model == "lasso":
        return Lasso(alpha=alpha, max_iter=20000)
    if model == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=20000)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def run(cfg: Config, fs: FeatureSet, model: str = "ridge",
        quantile_method: str = "residual", alphas: tuple[float, ...] = DEFAULT_ALPHAS,
        features: str = "engineered", verbose: bool = True) -> RunResult:
    _require_sklearn()
    if quantile_method not in ("residual", "quantreg"):
        raise ValueError("quantile_method must be 'residual' or 'quantreg'")

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        Xtr = X_train.to_numpy(float)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xv = (X_val.to_numpy(float) - mu) / sd
        Xte = (X_test.to_numpy(float) - mu) / sd

        # Penalty strength chosen on the purged validation slice.
        best_err, best_alpha, best_est = np.inf, alphas[0], None
        for a in alphas:
            est = _make(model, a).fit(Xtr, y_train)
            err = float(np.abs(est.predict(Xv) - y_val).mean())
            if err < best_err:
                best_err, best_alpha, best_est = err, a, est
        if verbose:
            n_nz = int(np.sum(np.abs(best_est.coef_) > 1e-10))
            print(f"    alpha={best_alpha:g}  val MAE={best_err:.4f}  "
                  f"nonzero coef={n_nz}/{Xtr.shape[1]}")

        point = best_est.predict(Xte)
        if quantile_method == "residual":
            resid = y_val - best_est.predict(Xv)
            offsets = np.quantile(resid, qs)
            offsets = offsets - offsets[mid]  # keep the median head unshifted
            return point[:, None] + offsets[None, :]

        from sklearn.linear_model import QuantileRegressor

        out = np.empty((len(X_test), len(qs)))
        for qi, q in enumerate(qs):
            if qi == mid:
                out[:, qi] = point
                continue
            qr = QuantileRegressor(quantile=q, alpha=best_alpha,
                                   solver="highs").fit(Xtr, y_train)
            out[:, qi] = qr.predict(Xte)
        return out

    return run_folds(cfg, fs, predict_quantiles, features=features, verbose=verbose)


def coefficients(cfg: Config, fs: FeatureSet, model: str = "ridge",
                 alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> pd.DataFrame:
    """Standardized coefficients per fold — a readable, signed view of what
    the linear model thinks each factor does. Mean and std across folds."""
    _require_sklearn()
    from ipo_model.data.preprocess import FoldScaler, engineered_table
    from ipo_model.data.splits import purged_walk_forward

    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    y = fs.y[cfg.main_horizon]
    rows = []
    for fold in folds:
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        X = engineered_table(fs, scaler)
        Xtr = X.iloc[fold.train_idx].to_numpy(float)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xv = (X.iloc[fold.val_idx].to_numpy(float) - mu) / sd
        best = min(
            (( float(np.abs(_make(model, a).fit(Xtr, y[fold.train_idx]).predict(Xv)
                            - y[fold.val_idx]).mean()), a) for a in alphas),
            key=lambda t: t[0],
        )[1]
        est = _make(model, best).fit(Xtr, y[fold.train_idx])
        rows.append(pd.Series(est.coef_, index=X.columns))
    coefs = pd.concat(rows, axis=1)
    out = pd.DataFrame({
        "feature": coefs.index,
        "coef": coefs.mean(axis=1).to_numpy(),
        "coef_std": coefs.std(axis=1).to_numpy(),
    })
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index) \
              .reset_index(drop=True)
