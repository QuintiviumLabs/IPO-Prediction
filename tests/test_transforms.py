"""Normal-score label transform and feature clipping: outlier containment."""
import numpy as np
import pytest

from ipo_model.config import Config
from ipo_model.data.preprocess import FoldScaler
from ipo_model.training.transforms import NormalScore


def _skewed(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.lognormal(-2.0, 0.8, n) - 0.1
    y[rng.integers(0, n, 5)] += 3.0   # five absurd moonshots
    return y


def test_roundtrip_and_monotone():
    y = _skewed()
    t = NormalScore().fit(y)
    z = t.transform(y)
    back = t.inverse(z)
    assert np.allclose(back, np.clip(y, np.sort(y)[0], np.sort(y)[-1]), atol=1e-8)
    order = np.argsort(y)
    assert (np.diff(z[order]) >= 0).all()          # monotone


def test_moonshot_cannot_dominate_score_space():
    y = _skewed()
    z = NormalScore().fit(y).transform(y)
    # In return space the max is many sigma out; in score space it is bounded
    # by the inverse-normal of n/(n+1) — magnitude information is gone.
    assert (y.max() - y.mean()) / y.std() > 5
    assert z.max() < 3.5
    assert abs(np.median(z)) < 0.1                 # body sits at zero, unpulled


def test_inverse_respects_empirical_tails():
    y = _skewed()
    t = NormalScore().fit(y)
    hi = t.inverse(np.array([1.2816]))             # ~90th percentile score
    lo = t.inverse(np.array([-1.2816]))
    assert np.isclose(hi[0], np.quantile(y, 0.9), rtol=0.15)
    assert np.isclose(lo[0], np.quantile(y, 0.1), rtol=0.25)
    # never extrapolates beyond observed outcomes
    assert t.inverse(np.array([10.0]))[0] <= y.max() + 1e-12


def test_unseen_values_clamp_not_crash():
    y = _skewed()
    t = NormalScore().fit(y)
    z = t.transform(np.array([y.min() - 5, y.max() + 5]))
    assert np.isfinite(z).all()


def test_fit_requires_enough_rows():
    with pytest.raises(ValueError):
        NormalScore().fit(np.arange(5))


def test_label_transform_end_to_end(fs, cfg):
    from ipo_model.training import loop
    fast = cfg.override(**{"split.n_folds": 2, "train.max_epochs": 3,
                           "train.patience": 2, "train.seeds": (0,),
                           "train.label_transform": "normal_score"})
    res = loop.run(fast, fs, verbose=False)
    for r in res.fold_results:
        assert np.isfinite(r.q_pred).all()
        assert (np.diff(r.q_pred, axis=1) >= -1e-9).all()   # still monotone
        # back-mapped predictions live in return space, not score space
        assert np.abs(r.q_pred).max() < 3.0


def test_unknown_transform_rejected(fs, cfg):
    from ipo_model.training import loop
    bad = cfg.override(**{"split.n_folds": 2, "train.seeds": (0,),
                          "train.label_transform": "winsor"})
    with pytest.raises(ValueError):
        loop.run(bad, fs, verbose=False)


def test_feature_clip(fs, cfg):
    on = cfg.override(**{"data.feature_clip": 3.0})
    idx = np.arange(len(fs) // 2)
    clipped = FoldScaler.fit(fs, idx, on.data)
    plain = FoldScaler.fit(fs, idx, cfg.data)
    assert np.abs(clipped.momentum_matrix(fs)).max() <= 3.0
    assert np.abs(clipped.gpr_features(fs)).max() <= 3.0
    assert np.abs(plain.momentum_matrix(fs)).max() > 3.0  # synthetic has tails