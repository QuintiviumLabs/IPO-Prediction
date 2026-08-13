"""Walk-forward conformal calibration of saved quantile predictions.

Fixes a measured coverage miss without retraining: for each fold, margins are
computed from the model's errors on STRICTLY EARLIER folds' out-of-sample
predictions, then added to that fold's band edges. Time order is preserved —
no future information calibrates the past — and the earliest fold(s), which
have no history to calibrate on, are passed through unchanged and flagged.

Two modes:

  asym  (recommended here) a separate margin per tail. The measured miss is
        asymmetric — the upper tail is exceeded far more often than nominal
        while the lower tail is slightly over-covered — so the upper edge
        must move UP a lot and the lower edge can move up a little. Each
        tail gets its own finite-sample-corrected conformal quantile, giving
        each tail its own coverage guarantee.

  sym   classic CQR (Romano-Patterson-Candes): one margin from
        max(q_lo - y, y - q_hi) applied to both edges. Guarantees total
        coverage but shares the correction across tails — with an
        asymmetric miss it widens the edge that was already too wide.

The median (point forecast, rank IC) is untouched either way. Edges are
clamped so they never cross the median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample-corrected empirical quantile (the ceil((n+1)q)/n rule)."""
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


def calibrate(df: pd.DataFrame, levels: tuple[float, ...], mode: str = "asym",
              min_cal: int = 80,
              window: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (calibrated predictions frame, per-fold report).

    df: a saved predictions frame (results_io schema) with columns
        fold, date, y_true and one q<level> column per quantile level.
    window: use only the most recent `window` prior rows as calibration.
        Under regime drift the oldest folds' errors mislead the margins
        (walk-forward breaks exchangeability); a recent window adapts faster
        at the cost of a noisier margin estimate. None = all prior rows.
    """
    if mode not in ("asym", "sym"):
        raise ValueError("mode must be 'asym' or 'sym'")
    qs = sorted(levels)
    if 0.5 not in qs or len(qs) < 3:
        raise ValueError("need a (lower, 0.5, upper) quantile structure")
    lo_c, mid_c, hi_c = f"q{qs[0]:g}", "q0.5", f"q{qs[-1]:g}"
    a_lo, a_hi = qs[0], 1.0 - qs[-1]

    out = df.copy()
    report = []
    for k in sorted(df["fold"].unique()):
        cal = df[df["fold"] < k]
        if window is not None and len(cal) > window:
            cal = cal.sort_values("date", kind="stable").tail(window)
        mask = df["fold"] == k
        if len(cal) < min_cal:
            report.append({"fold": k, "calibrated": False, "n_cal": len(cal),
                           "m_lo": 0.0, "m_hi": 0.0})
            continue
        y = cal["y_true"].to_numpy(float)
        s_lo = cal[lo_c].to_numpy(float) - y     # >0 when y broke the floor
        s_hi = y - cal[hi_c].to_numpy(float)     # >0 when y broke the ceiling
        if mode == "asym":
            m_lo = _conformal_quantile(s_lo, 1.0 - a_lo)
            m_hi = _conformal_quantile(s_hi, 1.0 - a_hi)
        else:
            m = _conformal_quantile(np.maximum(s_lo, s_hi), 1.0 - (a_lo + a_hi))
            m_lo = m_hi = m
        # Negative margins shrink an over-wide edge; positive widen. Clamp so
        # edges never cross the (untouched) median.
        out.loc[mask, lo_c] = np.minimum(df.loc[mask, lo_c] - m_lo,
                                         df.loc[mask, mid_c])
        out.loc[mask, hi_c] = np.maximum(df.loc[mask, hi_c] + m_hi,
                                         df.loc[mask, mid_c])
        report.append({"fold": k, "calibrated": True, "n_cal": len(cal),
                       "m_lo": m_lo, "m_hi": m_hi})
    return out, pd.DataFrame(report)


def tail_report(df: pd.DataFrame, levels: tuple[float, ...],
                folds: list | None = None) -> dict[str, float]:
    """Per-tail exceedance and coverage, optionally restricted to folds."""
    qs = sorted(levels)
    lo_c, hi_c = f"q{qs[0]:g}", f"q{qs[-1]:g}"
    d = df if folds is None else df[df["fold"].isin(folds)]
    y = d["y_true"].to_numpy(float)
    below = (y < d[lo_c].to_numpy(float))
    above = (y > d[hi_c].to_numpy(float))
    return {
        "n": len(d),
        "frac_below_lo": float(below.mean()),
        "frac_above_hi": float(above.mean()),
        "coverage": float((~below & ~above).mean()),
        "nominal_lo": qs[0], "nominal_hi": 1.0 - qs[-1],
        "nominal_coverage": qs[-1] - qs[0],
        "mean_width": float((d[hi_c] - d[lo_c]).mean()),
    }
