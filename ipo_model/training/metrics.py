"""Out-of-sample evaluation.

Design notes
------------
This is a CROSS-SECTIONAL problem: the model chooses among deals rather than
forecasting a level. So the headline metrics are ordering metrics (rank IC,
bucket spread), and the level metrics (MAE/RMSE) are reported but never
decisive — one -70% blow-up dominates a squared error while barely moving a
rank correlation.

At ~1k rows, a point estimate without an error bar is not a result. Every
number here therefore comes with machinery to ask "could this be zero?":

  evaluate()            per-fold / pooled metrics for one model
  ic_by_period()        the IC time series behind a single pooled IC
  t_test()              mean, HAC standard error, t, p for any series
  bootstrap_ci()        distribution-free CI for any metric (block-aware)
  compare_models()      paired A-vs-B test on identical rows
  benjamini_hochberg()  FDR control when many rungs are compared at once

Conventions: `y` is realized outperformance, `q_pred` is (N, Q) ascending
quantile forecasts. Point metrics use the median column; interval metrics use
the outer pair. Every function is nan-safe and degrades to nan rather than
raising on degenerate input (a constant forecast, an empty slice).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Keys that existed before the metrics expansion. Kept stable so saved
# results, the ablation table and any downstream code keep working.
LEGACY_KEYS = ("rank_ic", "hit_rate", "decile_spread", "mae", "rmse",
               "pinball", "coverage", "nominal_coverage")


# --------------------------------------------------------------------------
# Core per-slice evaluation
# --------------------------------------------------------------------------

def evaluate(y: np.ndarray, q_pred: np.ndarray,
             quantiles: tuple[float, ...]) -> dict[str, float]:
    """All metrics for one set of predictions.

    y:      (N,) realized outperformance
    q_pred: (N, Q) ascending quantile forecasts
    """
    y = np.asarray(y, dtype=float).ravel()
    q_pred = np.atleast_2d(np.asarray(q_pred, dtype=float))
    qs = sorted(quantiles)
    if q_pred.shape != (len(y), len(qs)):
        raise ValueError(f"q_pred shape {q_pred.shape} does not match "
                         f"({len(y)}, {len(qs)})")
    mid = qs.index(0.5)

    # Drop rows we cannot score rather than propagating nan through everything.
    ok = np.isfinite(y) & np.isfinite(q_pred).all(axis=1)
    n_dropped = int((~ok).sum())
    y, q_pred = y[ok], q_pred[ok]
    point = q_pred[:, mid] if len(y) else np.array([])

    out: dict[str, float] = {"n": float(len(y)), "n_dropped": float(n_dropped)}

    # ---- ordering (what actually matters) ----
    out["rank_ic"] = _spearman(point, y)
    out["rank_ic_p"] = _spearman_p(point, y)
    out["pearson_ic"] = _pearson(point, y)
    out["decile_spread"] = bucket_spread(point, y, 10)
    out["quintile_spread"] = bucket_spread(point, y, 5)
    out["top_bucket_mean"] = _bucket_mean(point, y, 5, top=True)
    out["bottom_bucket_mean"] = _bucket_mean(point, y, 5, top=False)

    # ---- directional ----
    nz = y != 0
    out["hit_rate"] = (float((np.sign(point) == np.sign(y))[nz].mean())
                       if nz.any() else float("nan"))

    # ---- level ----
    if len(y):
        err = point - y
        out["mae"] = float(np.abs(err).mean())
        out["medae"] = float(np.median(np.abs(err)))
        out["rmse"] = float(np.sqrt((err ** 2).mean()))
        # Campbell-Thompson style OOS R^2 against the natural null for a
        # benchmark-relative target: predicting zero outperformance.
        sse, sst = float((err ** 2).sum()), float((y ** 2).sum())
        out["r2_vs_zero"] = 1.0 - sse / sst if sst > 0 else float("nan")
        out["bias"] = float(err.mean())
    else:
        for k in ("mae", "medae", "rmse", "r2_vs_zero", "bias"):
            out[k] = float("nan")

    # ---- distribution / interval quality ----
    out.update(_interval_metrics(y, q_pred, qs))
    return out


def _interval_metrics(y: np.ndarray, q_pred: np.ndarray,
                      qs: list[float]) -> dict[str, float]:
    if not len(y):
        return {k: float("nan") for k in
                ("pinball", "coverage", "nominal_coverage", "coverage_error",
                 "frac_below_lo", "frac_above_hi", "mean_interval_width",
                 "interval_score")} | {"nominal_coverage": qs[-1] - qs[0]}

    resid = y[:, None] - q_pred
    qa = np.asarray(qs)[None, :]
    pinball = float(np.maximum(qa * resid, (qa - 1.0) * resid).mean())

    lo, hi = q_pred[:, 0], q_pred[:, -1]
    nominal = qs[-1] - qs[0]
    alpha = 1.0 - nominal
    below, above = y < lo, y > hi
    coverage = float((~below & ~above).mean())
    width = hi - lo
    # Winkler / interval score: width, plus a penalty for each miss. Lower is
    # better; it is the proper scoring rule for a central interval.
    penalty = np.where(below, (2 / alpha) * (lo - y), 0.0) + \
              np.where(above, (2 / alpha) * (y - hi), 0.0)
    return {
        "pinball": pinball,
        "coverage": coverage,
        "nominal_coverage": nominal,
        "coverage_error": coverage - nominal,
        "frac_below_lo": float(below.mean()),
        "frac_above_hi": float(above.mean()),
        "mean_interval_width": float(width.mean()),
        "interval_score": float((width + penalty).mean()),
    }


def _spearman_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(statistic, p) or (nan, nan) when the correlation is undefined.

    A constant forecast has no ordering, so its rank IC must be nan rather
    than 0 — reporting 0 would imply "measured, no skill" when the truth is
    "not measurable". scipy warns in these cases; the guard is intentional.
    """
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.spearmanr(a, b)
    stat, p = float(r.statistic), float(r.pvalue)
    return (stat, p) if np.isfinite(stat) else (float("nan"), float("nan"))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _spearman_pair(a, b)[0]


def _spearman_p(a: np.ndarray, b: np.ndarray) -> float:
    return _spearman_pair(a, b)[1]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bucket_spread(point: np.ndarray, y: np.ndarray, n_buckets: int = 5) -> float:
    """Mean realized outcome of the top prediction bucket minus the bottom.

    Returns nan unless every bucket would hold at least 5 observations —
    a 10-bucket split of a 200-row test block puts 20 deals in each tail,
    which is already thin; below 5 the number is noise.
    """
    if len(y) < 5 * n_buckets or np.std(point) == 0:
        return float("nan")
    splits = np.array_split(np.argsort(point), n_buckets)
    return float(y[splits[-1]].mean() - y[splits[0]].mean())


def _bucket_mean(point: np.ndarray, y: np.ndarray, n_buckets: int,
                 top: bool) -> float:
    if len(y) < 5 * n_buckets or np.std(point) == 0:
        return float("nan")
    splits = np.array_split(np.argsort(point), n_buckets)
    return float(y[splits[-1 if top else 0]].mean())


# Backwards-compatible alias for the pre-expansion private helper.
def _bucket_spread(point: np.ndarray, y: np.ndarray) -> float:
    return bucket_spread(point, y, 10 if len(y) >= 100 else 5)


# --------------------------------------------------------------------------
# Fold aggregation and significance
# --------------------------------------------------------------------------

def aggregate_folds(fold_metrics: list[dict[str, float]]
                    ) -> dict[str, tuple[float, float]]:
    """Mean and std of each metric across folds (nan-safe). Unchanged
    signature: existing callers keep working."""
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m.get(k, np.nan) for m in fold_metrics]
        with np.errstate(invalid="ignore"):
            out[k] = (_nanmean(vals), _nanstd(vals))
    return out


def summarize_folds(fold_metrics: list[dict[str, float]],
                    keys: tuple[str, ...] = ("rank_ic", "decile_spread",
                                             "quintile_spread", "pinball")
                    ) -> pd.DataFrame:
    """Per-metric mean, std, and a t-test of the across-fold mean against 0.

    With only 4-5 folds this test has very little power — treat it as a
    sanity check on consistency, and prefer `ic_by_period` + `t_test` on the
    pooled predictions, which has far more observations.
    """
    rows = []
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in fold_metrics], dtype=float)
        t = t_test(vals, hac=False)
        rows.append({"metric": k, "mean": t["mean"], "std": _nanstd(vals),
                     "n_folds": t["n"], "t_stat": t["t"], "p_value": t["p"]})
    return pd.DataFrame(rows)


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _nanstd(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")


# --------------------------------------------------------------------------
# The IC time series: the honest way to ask "is this signal real?"
# --------------------------------------------------------------------------

def ic_by_period(y: np.ndarray, q_pred: np.ndarray, dates: np.ndarray,
                 quantiles: tuple[float, ...], freq: str = "QE",
                 min_obs: int = 8) -> pd.DataFrame:
    """Rank IC computed within each calendar period.

    One pooled IC is a single number with no error bar. Slicing the same
    predictions by period gives a series you can actually test — the standard
    cross-sectional approach. Periods with fewer than `min_obs` deals are
    dropped (an IC on 3 deals is noise).

    freq: pandas offset alias — "ME" monthly, "QE" quarterly, "YE" annual.
    """
    qs = sorted(quantiles)
    point = np.asarray(q_pred, dtype=float)[:, qs.index(0.5)]
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": np.asarray(y, float),
                       "point": point})
    rows = []
    for period, g in df.groupby(pd.Grouper(key="date", freq=freq)):
        if len(g) < min_obs:
            continue
        rows.append({"period": period, "n": len(g),
                     "rank_ic": _spearman(g["point"].to_numpy(),
                                          g["y"].to_numpy())})
    if not rows:  # every period too thin — return the empty shape, not a crash
        return pd.DataFrame(columns=["period", "n", "rank_ic"])
    return pd.DataFrame(rows).dropna(subset=["rank_ic"]).reset_index(drop=True)


def ic_by_fold(y: np.ndarray, q_pred: np.ndarray, folds: np.ndarray,
               quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Rank IC computed WITHIN each fold.

    Prefer this (and its mean) over the pooled IC. Pooling mixes deals from
    different folds, and because each fold's model is trained on different
    data its predictions can sit at systematically different levels — a
    between-fold effect that inflates the pooled correlation without any
    genuine within-fold ordering skill. The constant-prediction baseline is
    the clean demonstration: pooled IC looks positive, fold-wise IC is
    undefined, and only the latter is honest.
    """
    qs = sorted(quantiles)
    point = np.asarray(q_pred, dtype=float)[:, qs.index(0.5)]
    y = np.asarray(y, dtype=float)
    folds = np.asarray(folds)
    rows = []
    for f in np.unique(folds):
        m = folds == f
        rows.append({"fold": f, "n": int(m.sum()),
                     "rank_ic": _spearman(point[m], y[m])})
    return pd.DataFrame(rows)


def t_test(values: np.ndarray, hac: bool = True,
           lags: int | None = None) -> dict[str, float]:
    """Test whether the mean of a series differs from zero.

    hac=True uses a Newey-West long-run variance, which is the right default
    for an IC series: consecutive periods share market regimes, and ignoring
    that autocorrelation overstates significance.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return {"mean": float(v[0]) if n else float("nan"), "se": float("nan"),
                "t": float("nan"), "p": float("nan"), "n": float(n)}
    mean = float(v.mean())
    se = _nw_se(v, lags) if hac else float(v.std(ddof=1) / np.sqrt(n))
    if not np.isfinite(se) or se <= 0:
        return {"mean": mean, "se": float("nan"), "t": float("nan"),
                "p": float("nan"), "n": float(n)}
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return {"mean": mean, "se": se, "t": float(t), "p": p, "n": float(n)}


def _nw_se(v: np.ndarray, lags: int | None = None) -> float:
    """Newey-West standard error of the mean."""
    n = len(v)
    if lags is None:  # Newey-West (1994) automatic bandwidth
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))
    e = v - v.mean()
    var = float(e @ e) / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(e[l:] @ e[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(np.sqrt(var / n))


# --------------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------------

def bootstrap_ci(y: np.ndarray, q_pred: np.ndarray,
                 quantiles: tuple[float, ...], metric: str = "rank_ic",
                 n_boot: int = 2000, groups: np.ndarray | None = None,
                 alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Percentile bootstrap CI for any metric produced by `evaluate`.

    Distribution-free, which suits a small fat-tailed sample. Pass `groups`
    (e.g. the calendar year of each deal) to resample whole blocks instead of
    individual rows — deals from the same period are not independent, and an
    iid bootstrap over rows will give a CI that is too narrow.
    """
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    rng = np.random.default_rng(seed)
    point_est = evaluate(y, q_pred, quantiles).get(metric, float("nan"))

    if groups is None:
        idx_pool = [np.arange(len(y))]
    else:
        groups = np.asarray(groups)
        idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]

    draws = np.empty(n_boot)
    for b in range(n_boot):
        if groups is None:
            sel = rng.integers(0, len(y), len(y))
        else:
            chosen = rng.integers(0, len(idx_pool), len(idx_pool))
            sel = np.concatenate([idx_pool[c] for c in chosen])
        try:
            draws[b] = evaluate(y[sel], q_pred[sel], quantiles).get(metric, np.nan)
        except Exception:
            draws[b] = np.nan

    good = draws[np.isfinite(draws)]
    if len(good) < 20:
        return {"estimate": point_est, "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan"), "n_boot": float(len(good))}
    lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value for "metric == 0".
    frac_le = float((good <= 0).mean())
    p = 2 * min(frac_le, 1 - frac_le)
    return {"estimate": point_est, "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(p, 1.0)), "n_boot": float(len(good))}


# --------------------------------------------------------------------------
# Paired model comparison
# --------------------------------------------------------------------------

def compare_models(y: np.ndarray, q_a: np.ndarray, q_b: np.ndarray,
                   quantiles: tuple[float, ...], name_a: str = "A",
                   name_b: str = "B", groups: np.ndarray | None = None,
                   n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Is model A genuinely better than model B, on identical rows?

    Two independent readings:
      * pinball loss — a Diebold-Mariano test on the per-deal loss
        differential, with a HAC standard error. Negative delta favours A.
      * rank IC — a paired bootstrap on the difference. Positive favours A.

    Comparing each rung against the SAME baseline on the SAME rows is far
    more powerful than eyeballing two independent confidence intervals.
    """
    y = np.asarray(y, dtype=float)
    q_a, q_b = np.asarray(q_a, dtype=float), np.asarray(q_b, dtype=float)
    qs = sorted(quantiles)
    mid = qs.index(0.5)

    la, lb = _pinball_rows(y, q_a, qs), _pinball_rows(y, q_b, qs)
    dm = t_test(la - lb, hac=True)

    ic_a = _spearman(q_a[:, mid], y)
    ic_b = _spearman(q_b[:, mid], y)

    rng = np.random.default_rng(seed)
    if groups is None:
        idx_pool = [np.arange(len(y))]
    else:
        groups = np.asarray(groups)
        idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        if groups is None:
            sel = rng.integers(0, len(y), len(y))
        else:
            chosen = rng.integers(0, len(idx_pool), len(idx_pool))
            sel = np.concatenate([idx_pool[c] for c in chosen])
        deltas[b] = (_spearman(q_a[sel, mid], y[sel])
                     - _spearman(q_b[sel, mid], y[sel]))
    good = deltas[np.isfinite(deltas)]
    if len(good) >= 20:
        lo, hi = np.percentile(good, [2.5, 97.5])
        frac_le = float((good <= 0).mean())
        ic_p = float(min(2 * min(frac_le, 1 - frac_le), 1.0))
    else:
        lo = hi = ic_p = float("nan")

    return {
        "model_a": name_a, "model_b": name_b, "n": float(len(y)),
        "pinball_a": float(la.mean()), "pinball_b": float(lb.mean()),
        "pinball_delta": float(la.mean() - lb.mean()),
        "dm_t": dm["t"], "dm_p": dm["p"],
        "rank_ic_a": ic_a, "rank_ic_b": ic_b,
        "rank_ic_delta": ic_a - ic_b,
        "rank_ic_delta_lo": float(lo), "rank_ic_delta_hi": float(hi),
        "rank_ic_delta_p": ic_p,
    }


def _pinball_rows(y: np.ndarray, q_pred: np.ndarray,
                  qs: list[float]) -> np.ndarray:
    """Per-row pinball loss averaged across quantile levels."""
    resid = y[:, None] - q_pred
    qa = np.asarray(qs)[None, :]
    return np.maximum(qa * resid, (qa - 1.0) * resid).mean(axis=1)


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------

def benjamini_hochberg(pvalues, alpha: float = 0.05) -> pd.DataFrame:
    """False-discovery-rate control across many simultaneous tests.

    Comparing ten rungs against a baseline means ten chances to find a
    spurious winner. Report BH-adjusted q-values alongside raw p-values.
    """
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.arange(1, n + 1)
    adj = np.full(n, np.nan)
    sorted_p = p[order]
    finite = np.isfinite(sorted_p)
    q = np.full(n, np.nan)
    if finite.any():
        m = int(finite.sum())
        q_sorted = sorted_p[:m] * m / ranks[:m]
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]  # enforce monotone
        q[:m] = np.clip(q_sorted, 0, 1)
    adj[order] = q
    return pd.DataFrame({"p_value": p, "q_value": adj,
                         "reject": np.where(np.isfinite(adj), adj <= alpha, False)})
