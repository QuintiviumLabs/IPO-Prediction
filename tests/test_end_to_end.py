"""Smoke test: the full stack runs on a small synthetic sample."""
import numpy as np

from ipo_model.baselines import lgbm, xgb
from ipo_model.training import loop


def _fast_cfg(cfg):
    return cfg.override(**{
        "split.n_folds": 2,
        "train.max_epochs": 3,
        "train.patience": 2,
        "train.seeds": (0,),
        "lgbm.num_boost_round": 30,
        "lgbm.early_stopping_rounds": 10,
        "xgb.num_boost_round": 30,
        "xgb.early_stopping_rounds": 10,
    })


def test_lgbm_baseline_runs_binary(fs, cfg):
    res = lgbm.run(_fast_cfg(cfg), fs, verbose=False)
    assert len(res.fold_results) == 2
    assert np.isfinite(res.pooled["brier"])
    for r in res.fold_results:
        assert r.q_pred.shape[1] == 1
        assert (r.q_pred >= 0).all() and (r.q_pred <= 1).all()


def test_xgb_baseline_runs_binary(fs, cfg):
    res = xgb.run(_fast_cfg(cfg), fs, verbose=False)
    assert len(res.fold_results) == 2
    assert np.isfinite(res.pooled["brier"])
    for r in res.fold_results:
        assert (r.q_pred >= 0).all() and (r.q_pred <= 1).all()


def test_deep_model_runs_binary(fs, cfg):
    res = loop.run(_fast_cfg(cfg), fs, verbose=False)
    assert len(res.fold_results) == 2
    for key in ("rank_ic", "auc", "brier", "accuracy", "base_rate"):
        assert key in res.pooled
    assert np.isfinite(res.pooled["brier"])
    for r in res.fold_results:
        assert r.q_pred.shape == (len(r.test_idx), 1)
        assert (r.q_pred >= 0).all() and (r.q_pred <= 1).all()
    n_oos = sum(len(r.test_idx) for r in res.fold_results)
    assert n_oos > 0.5 * len(fs) * (1 - 0.35)


def test_deep_model_runs_quantile(fs, cfg):
    res = loop.run(_fast_cfg(cfg).override(**{"model.head": "quantile"}),
                   fs, verbose=False)
    assert np.isfinite(res.pooled["mae"])
    for r in res.fold_results:
        assert (np.diff(r.q_pred, axis=1) >= 0).all()


def test_lgbm_baseline_runs_quantile(fs, cfg):
    res = lgbm.run(_fast_cfg(cfg).override(**{"model.head": "quantile"}),
                   fs, verbose=False)
    assert np.isfinite(res.pooled["mae"])
    for r in res.fold_results:
        assert (np.diff(r.q_pred, axis=1) >= 0).all()
