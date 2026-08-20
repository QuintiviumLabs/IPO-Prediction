"""Purged walk-forward cross-validation.

Targets are 21-trading-day forward returns, so a training example priced
shortly before a test block has a label window overlapping the test period —
its label contains future information relative to the test block's start.
Following the standard purging recipe (de Prado), a row is admitted to a
training (or validation) set only if

    label_end_date + embargo_days < period_start_date.

Folds are expanding-window: test blocks are contiguous, chronologically
ordered slices of the later part of the sample; each fold trains on
(purged) history before its test block, with the last `val_frac` of that
history (again purged against itself) held out for early stopping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipo_model.config import SplitConfig


@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_start: np.datetime64


def purged_walk_forward(
    dates: np.ndarray,
    label_end: np.ndarray,
    cfg: SplitConfig,
) -> list[Fold]:
    """`dates` (pricing dates) must be sorted ascending; rows keep their order."""
    n = len(dates)
    if not (np.diff(dates.astype("datetime64[ns]").astype(np.int64)) >= 0).all():
        raise ValueError("Rows must be sorted by pricing date.")
    if len(label_end) != n:
        raise ValueError("dates and label_end must be aligned.")

    embargo = np.timedelta64(cfg.embargo_days, "D")
    first_test = int(n * cfg.min_train_frac)
    bounds = np.linspace(first_test, n, cfg.n_folds + 1).round().astype(int)

    folds = []
    for k in range(cfg.n_folds):
        test_idx = np.arange(bounds[k], bounds[k + 1])
        if len(test_idx) == 0:
            continue
        test_start = dates[test_idx[0]]

        candidates = np.arange(0, bounds[k])  # expanding window
        keep = label_end[candidates] + embargo < test_start
        candidates = candidates[keep]
        if len(candidates) == 0:
            raise ValueError(f"Fold {k}: no training rows survive purging.")

        n_val = max(1, int(len(candidates) * cfg.val_frac))
        val_idx = candidates[-n_val:]
        val_start = dates[val_idx[0]]
        train_idx = candidates[:-n_val]
        train_idx = train_idx[label_end[train_idx] + embargo < val_start]
        if len(train_idx) == 0:
            raise ValueError(f"Fold {k}: no training rows survive train/val purging.")

        folds.append(Fold(train_idx=train_idx, val_idx=val_idx,
                          test_idx=test_idx, test_start=test_start))
    return folds
