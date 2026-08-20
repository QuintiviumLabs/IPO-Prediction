"""Bundle save/load round-trip and inference on unseen deals."""
import copy

import numpy as np
import pandas as pd
import pytest

from ipo_model.data.features import build_features, load_raw
from ipo_model.persist import fit_final, load_bundle


@pytest.fixture(scope="module")
def fast_cfg(cfg):
    return cfg.override(**{"train.max_epochs": 3, "train.patience": 2,
                           "train.seeds": (0, 1)})


@pytest.fixture(scope="module")
def bundle_dir(fast_cfg, fs, tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle") / "final"
    fit_final(fast_cfg, fs, out, verbose=False)
    return out


@pytest.fixture(scope="module")
def bundle(bundle_dir):
    return load_bundle(bundle_dir)


def test_roundtrip_identical_predictions(bundle, bundle_dir, fs):
    p1 = bundle.predict(fs)
    p2 = load_bundle(bundle_dir).predict(fs)
    for h in p1:
        assert np.allclose(p1[h], p2[h], atol=1e-6)


def test_predict_shape_and_monotone(bundle, fs, fast_cfg):
    preds = bundle.predict(fs)
    h = fast_cfg.main_horizon
    assert preds[h].shape == (len(fs), len(fast_cfg.model.quantiles))
    assert (np.diff(preds[h], axis=1) >= -1e-6).all()
    assert np.isfinite(preds[h]).all()


def test_subset_prediction_matches(bundle, fs):
    idx = np.arange(10)
    full = bundle.predict(fs)
    sub = bundle.predict(fs, idx=idx)
    for h in full:
        assert np.allclose(full[h][idx], sub[h], atol=1e-6)


def test_schema_guard_rejects_changed_features(bundle, small_data, fast_cfg):
    raw = copy.copy(load_raw(small_data))
    raw.peers = None                       # p1 block disappears
    fs2 = build_features(raw, fast_cfg.data)
    with pytest.raises(ValueError, match="momentum block"):
        bundle.predict(fs2)


def test_new_ipo_scored_without_prices(bundle, small_data, fast_cfg):
    raw = copy.copy(load_raw(small_data))
    new = raw.ipos.iloc[[-1]].copy()
    new["ipo_id"] = "ipo_tomorrow"
    new["first_trade_date"] = raw.ipos["first_trade_date"].max() \
        + pd.Timedelta(days=7)
    raw.ipos = pd.concat([raw.ipos, new], ignore_index=True)
    fs_inf = build_features(raw, fast_cfg.data, inference=True)
    k = list(fs_inf.ids).index("ipo_tomorrow")
    preds = bundle.predict(fs_inf, idx=np.array([k]))
    h = fast_cfg.main_horizon
    assert preds[h].shape == (1, len(fast_cfg.model.quantiles))
    assert np.isfinite(preds[h]).all()
    assert (np.diff(preds[h], axis=1) >= -1e-6).all()
