"""Rank-based label transform for heavy-tailed targets.

NormalScore maps labels through the TRAINING fold's empirical CDF onto
Gaussian scores (van der Waerden / rank-based inverse normal). In score
space a +300% moonshot is merely "about +2.5 sigma": its magnitude cannot
influence training beyond its rank, so the body of the distribution — the
typical deal near the median — is never pulled toward the tail.

The transform is invertible through the same training ECDF, so predicted
quantiles map back to return space with the empirical (asymmetric, fat)
tails the training data actually had, instead of tails the network had to
stretch to reach. Fit on the training fold only: no test statistics leak.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class NormalScore:
    """Fit on training labels; transform any labels; invert predictions."""

    def fit(self, y_train: np.ndarray) -> "NormalScore":
        y = np.sort(np.asarray(y_train, dtype=float))
        n = len(y)
        if n < 10:
            raise ValueError(f"need >=10 training labels to fit, got {n}")
        self._vals = y
        self._probs = np.arange(1, n + 1) / (n + 1.0)  # Weibull plotting positions
        self._pmin, self._pmax = self._probs[0], self._probs[-1]
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Return space -> Gaussian score space (monotone)."""
        p = np.interp(np.asarray(y, dtype=float), self._vals, self._probs,
                      left=self._pmin, right=self._pmax)
        return stats.norm.ppf(np.clip(p, self._pmin, self._pmax))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Score space -> return space (monotone; clipped to the training
        range, so back-mapped quantiles never extrapolate beyond observed
        outcomes)."""
        p = stats.norm.cdf(np.asarray(z, dtype=float))
        return np.interp(np.clip(p, self._pmin, self._pmax),
                         self._probs, self._vals)
