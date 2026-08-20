import numpy as np

from ipo_model import diagnostics
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


def test_tree_gain_importance(fs, cfg):
    df = diagnostics.tree_gain_importance(_fast_cfg(cfg), fs)
    assert set(df.columns) == {"feature", "gain_share", "gain_share_std"}
    assert np.isfinite(df["gain_share"]).all()
    assert df["gain_share"].iloc[0] >= df["gain_share"].iloc[-1]  # sorted
    assert "f1_med_1d" in set(df["feature"])


def test_deep_permutation_importance(fs, cfg):
    df = diagnostics.deep_permutation_importance(
        _fast_cfg(cfg), fs, n_repeats=1, verbose=False)
    assert np.isfinite(df["delta_ic"]).all()
    names = set(df["feature"])
    assert "gpr" in names and "bookrunners" in names and "f1_med_1d" in names


def test_slice_metrics_binary(fs, cfg):
    fast = _fast_cfg(cfg)
    res = loop.run(fast, fs, verbose=False)
    df = diagnostics.slice_metrics(fast, fs, res)
    assert (df["n"] >= 20).all()
    assert np.isfinite(df["brier"]).all()
    assert "auc" in df.columns
    assert any(s.startswith("market=") for s in df["slice"])


def test_slice_metrics_quantile(fs, cfg):
    fast = _fast_cfg(cfg).override(**{"model.head": "quantile"})
    res = loop.run(fast, fs, verbose=False)
    df = diagnostics.slice_metrics(fast, fs, res)
    assert np.isfinite(df["mae"]).all()
    assert "coverage" in df.columns
