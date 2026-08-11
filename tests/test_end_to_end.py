"""Smoke test: the full stack runs on a small synthetic sample."""
import numpy as np

from ipo_model.baselines import lgbm
from ipo_model.training import loop


def _fast_cfg(cfg):
    return cfg.override(**{
        "split.n_folds": 2,
        "train.max_epochs": 3,
        "train.patience": 2,
        "train.seeds": (0,),
        "lgbm.num_boost_round": 30,
        "lgbm.early_stopping_rounds": 10,
    })


def test_lgbm_baseline_runs(fs, cfg):
    res = lgbm.run(_fast_cfg(cfg), fs, verbose=False)
    assert len(res.fold_results) == 2
    assert np.isfinite(res.pooled["mae"])
    for r in res.fold_results:
        assert (np.diff(r.q_pred, axis=1) >= 0).all()


def test_deep_model_runs(fs, cfg):
    res = loop.run(_fast_cfg(cfg), fs, verbose=False)
    assert len(res.fold_results) == 2
    assert np.isfinite(res.pooled["mae"])
    n_oos = sum(len(r.test_idx) for r in res.fold_results)
    assert n_oos > 0.5 * len(fs) * (1 - 0.35)
