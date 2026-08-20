"""Persist out-of-sample predictions so evaluation never requires retraining.

A RunResult holds every out-of-sample prediction the model made, but it lives
in memory and dies with the process. Writing it to disk means any analysis
invented later — period IC t-tests, bootstrap intervals, paired comparisons
against a baseline — can run on a training job that already finished.

One row per (deal, rung):
    quantile head:  ipo_id, date, market, fold, y_true, q0.1, q0.5, q0.9
    binary head:    ipo_id, date, market, fold, y_true, p_out
(p_out = predicted probability of outperforming the benchmark; downstream
rank/IC analyses treat it as the point score.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult

PRED_DIR = "predictions"


def predictions_frame(cfg: Config, fs: FeatureSet, res: RunResult) -> pd.DataFrame:
    """Flatten a RunResult's out-of-sample predictions into a tidy frame."""
    qs = sorted(cfg.model.quantiles)
    market = (np.asarray(fs.market_names)[fs.market_onehot.argmax(axis=1)]
              if len(fs.market_names) else np.array(["?"] * len(fs)))
    frames = []
    for r in res.fold_results:
        idx = r.test_idx
        block = {
            "ipo_id": fs.ids[idx],
            "date": fs.dates[idx],
            "market": market[idx],
            "fold": r.fold,
            "y_true": fs.y[cfg.main_horizon][idx],
        }
        pred = np.atleast_2d(r.q_pred)
        if pred.shape[1] == 1:          # binary head: probability score
            block["p_out"] = pred[:, 0]
        else:
            for j, q in enumerate(qs):
                block[f"q{q:g}"] = pred[:, j]
        frames.append(pd.DataFrame(block))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def save_predictions(cfg: Config, fs: FeatureSet, res: RunResult, rung: str,
                     out_dir: str | Path = "results") -> Path:
    d = Path(out_dir) / PRED_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rung}.csv"
    predictions_frame(cfg, fs, res).to_csv(path, index=False)
    return path


def load_predictions(rung: str, out_dir: str | Path = "results") -> pd.DataFrame:
    path = Path(out_dir) / PRED_DIR / f"{rung}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved predictions at {path}. Re-run with prediction saving "
            "enabled (run_ablations.py saves by default)."
        )
    return pd.read_csv(path, parse_dates=["date"])


def available_rungs(out_dir: str | Path = "results") -> list[str]:
    d = Path(out_dir) / PRED_DIR
    return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []


def quantile_columns(df: pd.DataFrame) -> tuple[np.ndarray, tuple[float, ...]]:
    """Extract the (N, Q) quantile matrix and its levels from a saved frame.

    Binary-head frames have a single p_out probability column instead; it is
    returned as a (N, 1) matrix with level (0.5,) so rank/point analyses work
    unchanged. Interval analyses should check len(levels) > 1 first."""
    cols = sorted((c for c in df.columns if c.startswith("q") and c != "q"),
                  key=lambda c: float(c[1:]))
    if not cols:
        if "p_out" in df.columns:
            return df[["p_out"]].to_numpy(float), (0.5,)
        raise ValueError(f"no quantile columns found in {list(df.columns)}")
    levels = tuple(float(c[1:]) for c in cols)
    return df[cols].to_numpy(float), levels
