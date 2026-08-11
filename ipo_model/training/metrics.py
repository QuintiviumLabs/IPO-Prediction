"""Out-of-sample evaluation metrics.

Point metrics use the median forecast; interval metrics use the outer
quantile pair. Rank IC and decile spread are the metrics that matter for a
cross-sectional signal — MAE/RMSE are reported but heavy-tailed IPO returns
make them blunt instruments.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def evaluate(y: np.ndarray, q_pred: np.ndarray,
             quantiles: tuple[float, ...]) -> dict[str, float]:
    """y: (N,) realized returns; q_pred: (N, Q) ascending quantile forecasts."""
    qs = sorted(quantiles)
    mid = qs.index(0.5)
    point = q_pred[:, mid]
    out: dict[str, float] = {}

    if len(y) >= 3 and np.std(point) > 0 and np.std(y) > 0:
        out["rank_ic"] = float(stats.spearmanr(point, y).statistic)
    else:
        out["rank_ic"] = float("nan")
    out["hit_rate"] = float((np.sign(point) == np.sign(y))[y != 0].mean())
    out["decile_spread"] = _bucket_spread(point, y)
    out["mae"] = float(np.abs(point - y).mean())
    out["rmse"] = float(np.sqrt(((point - y) ** 2).mean()))

    err = y[:, None] - q_pred
    qa = np.asarray(qs)[None, :]
    out["pinball"] = float(np.maximum(qa * err, (qa - 1.0) * err).mean())
    lo, hi = q_pred[:, 0], q_pred[:, -1]
    out["coverage"] = float(((y >= lo) & (y <= hi)).mean())
    out["nominal_coverage"] = qs[-1] - qs[0]
    return out


def _bucket_spread(point: np.ndarray, y: np.ndarray) -> float:
    """Mean realized return of the top prediction bucket minus the bottom.

    Deciles when the sample is large enough, quintiles otherwise.
    """
    n_buckets = 10 if len(y) >= 100 else 5
    if len(y) < 2 * n_buckets:
        return float("nan")
    order = np.argsort(point)
    splits = np.array_split(order, n_buckets)
    return float(y[splits[-1]].mean() - y[splits[0]].mean())


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Mean and std of each metric across folds (nan-safe)."""
    keys = fold_metrics[0].keys()
    return {
        k: (float(np.nanmean([m[k] for m in fold_metrics])),
            float(np.nanstd([m[k] for m in fold_metrics])))
        for k in keys
    }
