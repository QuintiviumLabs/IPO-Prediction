"""Ablation 0: gradient-boosted trees on engineered features.

Same purged walk-forward folds, same targets, same metrics as the deep model.
This is the bar the three-arm architecture has to clear: if the LSTM arms
can't beat hand-built summaries of the same information, the sequences are
not carrying incremental signal.

Point forecasts come from a Huber-objective model; the 10/50/90 interval
comes from three quantile-objective models so the baseline is comparable on
pinball loss and coverage too.
"""
from __future__ import annotations

import numpy as np

from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.data.preprocess import FoldScaler, engineered_table
from ipo_model.data.splits import purged_walk_forward
from ipo_model.training.loop import FoldResult, RunResult
from ipo_model.training.metrics import aggregate_folds, evaluate


def run(cfg: Config, fs: FeatureSet, seed: int = 0, verbose: bool = True) -> RunResult:
    import lightgbm as lgb

    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    main_h = cfg.main_horizon
    y = fs.y[main_h]
    qs = sorted(cfg.model.quantiles)
    mid = qs.index(0.5)
    results: list[FoldResult] = []

    for k, fold in enumerate(folds):
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        X = engineered_table(fs, scaler)
        X_train, X_val, X_test = (X.iloc[fold.train_idx], X.iloc[fold.val_idx],
                                  X.iloc[fold.test_idx])
        y_train, y_val = y[fold.train_idx], y[fold.val_idx]

        q_pred = np.empty((len(fold.test_idx), len(qs)))
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
        q_pred = np.sort(q_pred, axis=1)  # enforce monotone quantiles

        m = evaluate(y[fold.test_idx], q_pred, cfg.model.quantiles)
        results.append(FoldResult(fold=k, test_idx=fold.test_idx, q_pred=q_pred,
                                  per_seed_val_loss=[], metrics=m))
        if verbose:
            print(f"  fold {k}: n_test={len(fold.test_idx)} "
                  f"IC={m['rank_ic']:+.3f} hit={m['hit_rate']:.3f} "
                  f"spread={m['decile_spread']:+.4f} MAE={m['mae']:.4f} "
                  f"cov={m['coverage']:.2f}")

    pooled_y = np.concatenate([y[r.test_idx] for r in results])
    pooled_q = np.concatenate([r.q_pred for r in results])
    return RunResult(
        fold_results=results,
        summary=aggregate_folds([r.metrics for r in results]),
        pooled=evaluate(pooled_y, pooled_q, cfg.model.quantiles),
    )
