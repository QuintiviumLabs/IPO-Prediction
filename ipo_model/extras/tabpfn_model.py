"""TabPFN rung: in-context learning for small tabular data.

TabPFN is a transformer pre-trained on millions of synthetic tabular tasks.
It does not train on your data — it conditions on it in a single forward
pass, which is exactly the few-shot regime this dataset sits in (~1k rows).
It also emits a full predictive distribution, so the quantile band comes
natively rather than from a pinball head.

Runs through `baselines.common.run_folds`: identical purged folds, feature
table and metrics as every other rung.

Two deliberate differences from the other rungs:
  * train and validation rows are CONCATENATED into the in-context set.
    There is no fitting and no early stopping, so there is nothing for a
    validation slice to protect against — and both blocks are already purged
    against the test block, so this adds context without leaking.
  * if the in-context set exceeds `max_train`, the MOST RECENT rows are kept.
    Truncating the distant past is the right bias for a time series.

Install (recommended pin — its weights are an unauthenticated download):
    pip install "tabpfn==2.2.1"

Version 8.x also works but its weights are gated on HuggingFace: you must
accept the terms at huggingface.co/Prior-Labs/tabpfn_3 and authenticate with
`hf auth login` or an HF_TOKEN. On a locked-down machine prefer 2.2.1.

Offline / firewalled machines: download the checkpoint in a browser from
    https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt
and place it at ~/.cache/tabpfn/tabpfn-v2-regressor.ckpt
(Windows: %USERPROFILE%\\.cache\\tabpfn\\tabpfn-v2-regressor.ckpt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipo_model.baselines.common import run_folds
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training.loop import RunResult

# TabPFN v2's pre-training regime; beyond this it degrades and slows sharply.
DEFAULT_MAX_TRAIN = 10000
FEATURE_WARN_LIMIT = 500


def _make_regressor(device: str, n_estimators: int, seed: int):
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The TabPFN rung needs the tabpfn package:\n"
            '    pip install "tabpfn==2.2.1"\n'
            "(or: pip install -r requirements-extras.txt)"
        ) from e
    return TabPFNRegressor(device=device, n_estimators=n_estimators,
                           random_state=seed)


def _as_quantile_matrix(raw, n_rows: int, n_q: int) -> np.ndarray:
    """TabPFN returns one array per quantile; normalise to (n_rows, n_q)."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:  # single quantile requested
        arr = arr[:, None]
    if arr.shape == (n_q, n_rows) and n_q != n_rows:
        arr = arr.T
    if arr.shape != (n_rows, n_q):
        raise ValueError(f"unexpected TabPFN quantile shape {arr.shape}, "
                         f"expected ({n_rows}, {n_q})")
    return arr


def run(cfg: Config, fs: FeatureSet, features: str = "engineered",
        max_train: int = DEFAULT_MAX_TRAIN, device: str = "auto",
        n_estimators: int = 4, seed: int = 0,
        verbose: bool = True) -> RunResult:
    if cfg.model.head == "binary":
        raise ValueError("the TabPFN rung is an interval model — run it "
                         "with model.head: quantile")

    def predict_quantiles(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          X_test: pd.DataFrame, qs: list[float],
                          mid: int) -> np.ndarray:
        # No fitting => no early stopping => validation rows are free context.
        X_ctx = pd.concat([X_train, X_val], axis=0).to_numpy(float)
        y_ctx = np.concatenate([y_train, y_val])
        if len(X_ctx) > max_train:  # keep the most recent context
            X_ctx, y_ctx = X_ctx[-max_train:], y_ctx[-max_train:]
        if X_ctx.shape[1] > FEATURE_WARN_LIMIT and verbose:
            print(f"    warning: {X_ctx.shape[1]} features exceeds TabPFN's "
                  f"~{FEATURE_WARN_LIMIT} limit; consider fewer bookrunner columns")

        model = _make_regressor(device, n_estimators, seed)
        model.fit(X_ctx, y_ctx)
        Xte = X_test.to_numpy(float)

        try:
            raw = model.predict(Xte, output_type="quantiles", quantiles=list(qs))
            out = _as_quantile_matrix(raw, len(Xte), len(qs))
        except Exception as e:
            # Older/newer builds may not expose quantile output; fall back to
            # the point forecast plus empirical validation-residual offsets.
            if verbose:
                print(f"    quantile output unavailable ({type(e).__name__}); "
                      "using point forecast + residual offsets")
            point_val = np.asarray(model.predict(X_val.to_numpy(float)), dtype=float)
            offsets = np.quantile(y_val - point_val, qs)
            offsets = offsets - offsets[mid]
            point = np.asarray(model.predict(Xte), dtype=float)
            out = point[:, None] + offsets[None, :]

        if verbose:
            print(f"    context={len(X_ctx)} rows x {X_ctx.shape[1]} features")
        return out

    return run_folds(cfg, fs, predict_quantiles, features=features, verbose=verbose)
