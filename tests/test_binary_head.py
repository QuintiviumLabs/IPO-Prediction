"""Binary outperform/underperform head: loss, metrics, loop, baselines."""
import numpy as np
import pytest
import torch

from ipo_model.baselines import naive
from ipo_model.training import loop
from ipo_model.training.losses import binary_multi_horizon_loss
from ipo_model.training.metrics import evaluate_binary


def test_evaluate_binary_hand_check():
    y = np.array([0.10, 0.05, -0.02, -0.10, 0.01, -0.01])
    p = np.array([0.90, 0.80, 0.30, 0.10, 0.60, 0.40])   # perfectly ordered
    m = evaluate_binary(y, p)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["base_rate"] == 0.5
    assert m["rank_ic"] > 0.9
    assert 0 < m["brier"] < 0.25
    assert m["brier_skill"] > 0


def test_evaluate_binary_constant_prediction():
    y = np.random.default_rng(0).normal(0, 0.1, 100)
    p = np.full(100, 0.6)
    m = evaluate_binary(y, p)
    assert np.isnan(m["rank_ic"])      # no ordering — not measurable
    assert np.isnan(m["auc"])
    assert np.isfinite(m["brier"])


def test_binary_loss_pos_weight_direction():
    logits = {21: torch.zeros(8, 1)}
    targets = {21: torch.tensor([1., 1., 1., 1., 1., 1., 0., 0.])}
    plain = binary_multi_horizon_loss(logits, targets, 21, 0.3)
    weighted = binary_multi_horizon_loss(
        logits, targets, 21, 0.3, {21: torch.tensor(3.0)})
    assert weighted > plain            # positives up-weighted at logit 0


def test_loop_binary_probabilities_and_metrics(fs, cfg):
    fast = cfg.override(**{"split.n_folds": 2, "train.max_epochs": 3,
                           "train.patience": 2, "train.seeds": (0,)})
    res = loop.run(fast, fs, verbose=False)
    for r in res.fold_results:
        assert r.q_pred.shape == (len(r.test_idx), 1)
        assert (r.q_pred >= 0).all() and (r.q_pred <= 1).all()
        assert {"auc", "brier", "base_rate"} <= set(r.metrics)
    assert 0.2 < res.pooled["base_rate"] < 0.9


def test_loop_binary_rejects_label_transform(fs, cfg):
    bad = cfg.override(**{"split.n_folds": 2, "train.seeds": (0,),
                          "train.label_transform": "normal_score"})
    with pytest.raises(ValueError, match="binary"):
        loop.run(bad, fs, verbose=False)


def test_naive_constant_binary_is_base_rate(fs, cfg):
    fast = cfg.override(**{"split.n_folds": 2, "train.seeds": (0,)})
    res = naive.run(fast, fs, strategy="constant", verbose=False)
    for r in res.fold_results:
        assert np.allclose(r.q_pred, r.q_pred[0])       # one prob for all
        assert np.isnan(r.metrics["rank_ic"])           # no ordering
        assert np.isfinite(r.metrics["brier"])


def test_naive_f1_logistic_binary(fs, cfg):
    fast = cfg.override(**{"split.n_folds": 2, "train.seeds": (0,)})
    res = naive.run(fast, fs, strategy="feature", verbose=False)
    for r in res.fold_results:
        assert (r.q_pred >= 0).all() and (r.q_pred <= 1).all()
        assert np.std(r.q_pred) > 0                      # actually varies


def test_binary_predictions_roundtrip_csv(fs, cfg, tmp_path):
    from ipo_model.results_io import (load_predictions, quantile_columns,
                                      save_predictions)
    fast = cfg.override(**{"split.n_folds": 2, "train.max_epochs": 2,
                           "train.patience": 1, "train.seeds": (0,)})
    res = loop.run(fast, fs, verbose=False)
    save_predictions(fast, fs, res, "bin_test", out_dir=tmp_path)
    df = load_predictions("bin_test", out_dir=tmp_path)
    assert "p_out" in df.columns
    q, levels = quantile_columns(df)
    assert q.shape[1] == 1 and levels == (0.5,)
    assert (q >= 0).all() and (q <= 1).all()
