"""Macro block, peer block, syndicate extras, F4 removal, regime terciles."""
import numpy as np
import pytest

from ipo_model.data.features import (_ExpandingTerciles, build_features,
                                     load_raw)
from ipo_model.data.preprocess import FoldScaler


def _col(fs, name):
    return fs.momentum[:, fs.momentum_names.index(name)]


def test_f4_removed(fs):
    assert not [n for n in fs.momentum_names if n.startswith("f4")]


def test_macro_block_present_and_sane(fs):
    for n in ["m_geo_mom_63d", "m_geo_dd_252", "m_geo_rvol_21d",
              "m_gpr_vol63", "m_regime_hot", "m_regime_cold",
              "m_vol_index_z", "m_fx_ret_21d", "m_vix_z", "m_vix_chg_21d",
              "m_hy_oas_z", "m_em_vs_dm_63d"]:
        assert n in fs.momentum_names, n
    # the GPR level/mean is deliberately NOT an input (the GPR arm covers it)
    assert "m_gpr_mean63" not in fs.momentum_names
    assert np.isfinite(fs.momentum).all()
    # drawdown is <= 0 by construction; regime flags are exclusive binaries
    assert (_col(fs, "m_geo_dd_252") <= 1e-9).all()
    hot, cold = _col(fs, "m_regime_hot"), _col(fs, "m_regime_cold")
    assert set(np.unique(hot)) <= {0.0, 1.0}
    assert not ((hot == 1) & (cold == 1)).any()
    assert hot.sum() > 0 and cold.sum() > 0        # both regimes occur
    # rolling z-scores should look like z-scores, not raw levels
    assert np.abs(_col(fs, "m_vix_z")).max() < 8
    assert np.abs(_col(fs, "m_vol_index_z")).max() < 8


def test_peer_block_matches_raw(fs, raw):
    for n in ["p1_med_21d", "p1_med_63d", "p1_iqr_21d", "p1_n"]:
        assert n in fs.momentum_names
    peers = raw.peers
    # recompute for a handful of deals straight from peers.csv
    for ipo_id in fs.ids[:5]:
        g = peers[peers["ipo_id"] == ipo_id]
        i = list(fs.ids).index(ipo_id)
        assert _col(fs, "p1_n")[i] == len(g)
        assert np.isclose(_col(fs, "p1_med_21d")[i],
                          np.median(g["ret_21d"]), atol=1e-6)


def test_peers_asof_on_pricing_date_rejected(small_data, cfg):
    raw = load_raw(small_data)
    bad = raw.peers.copy()
    ft = raw.ipos.set_index("ipo_id")["first_trade_date"]
    bad.loc[bad.index[0], "asof"] = ft[bad["ipo_id"].iloc[0]]  # not strictly before
    raw.peers = bad
    with pytest.raises(ValueError, match="strictly before"):
        build_features(raw, cfg.data)


def test_static_extras_standardized_on_train_only(fs, cfg):
    train_idx = np.arange(len(fs) // 2)
    scaler = FoldScaler.fit(fs, train_idx, cfg.data)
    extras = scaler.static_extras(fs)
    cont = np.asarray(fs.static_extra_continuous)
    assert extras.shape == fs.static_extra.shape
    # continuous columns: ~zero mean on the train slice under its own scaler
    assert np.abs(extras[train_idx][:, cont].mean(axis=0)).max() < 0.2
    # binary columns pass through untouched
    assert (extras[:, ~cont] == fs.static_extra[:, ~cont]).all()
    names = fs.static_extra_names
    for n in ["n_banks", "log_n_banks", "prestige_rank_max", "has_bb",
              "jumbo_synd", "archetype_bb_dom"]:
        assert n in names, n


def test_static_matrix_carries_extras(fs, cfg):
    train_idx = np.arange(len(fs) // 2)
    scaler = FoldScaler.fit(fs, train_idx, cfg.data)
    binary, _bk = scaler.static_matrix(fs)
    n_extra = len(fs.static_extra_names)
    assert n_extra > 0
    # sector + n_bk + extras (market one-hot off by default)
    assert binary.shape[1] == len(fs.sector_names) + 1 + n_extra


def test_prestige_expanding_and_bounded(fs, raw, cfg):
    j = fs.static_extra_names.index("prestige_rank_max")
    v = fs.static_extra[:, j]
    assert (v >= 0).all() and (v <= 1).all()
    assert v.max() > 0                       # someone has prestige
    # prefix stability: dropping later IPOs must not change earlier values
    import copy
    cut = raw.ipos["first_trade_date"].quantile(0.6)
    raw2 = copy.copy(raw)
    raw2.ipos = raw.ipos[raw.ipos["first_trade_date"] <= cut].copy()
    fs2 = build_features(raw2, cfg.data)
    common = set(fs2.ids) & set(fs.ids)
    idx1 = {i: k for k, i in enumerate(fs.ids)}
    idx2 = {i: k for k, i in enumerate(fs2.ids)}
    j2 = fs2.static_extra_names.index("prestige_rank_max")
    for i in list(common)[:50]:
        assert np.isclose(fs.static_extra[idx1[i], j],
                          fs2.static_extra[idx2[i], j2], atol=1e-12)


def test_expanding_terciles_unit():
    t = _ExpandingTerciles()
    assert [t.classify(x) for x in [1, 2, 3, 4, 5]] == [0] * 5  # warmup
    assert t.classify(100.0) == 1     # far above anything seen
    assert t.classify(-100.0) == -1   # far below
    assert t.classify(3.0) == 0       # middle of the pack


def test_inference_mode_keeps_unlabeled_rows(small_data, cfg):
    import copy
    import pandas as pd
    raw = load_raw(small_data)
    raw2 = copy.copy(raw)
    new = raw.ipos.iloc[[-1]].copy()
    new["ipo_id"] = "ipo_brand_new"
    new["first_trade_date"] = raw.ipos["first_trade_date"].max() \
        + pd.Timedelta(days=7)
    raw2.ipos = pd.concat([raw.ipos, new], ignore_index=True)  # no price rows
    fs_train = build_features(raw2, cfg.data)
    fs_inf = build_features(raw2, cfg.data, inference=True)
    assert "ipo_brand_new" not in fs_train.ids
    assert "ipo_brand_new" in fs_inf.ids
    k = list(fs_inf.ids).index("ipo_brand_new")
    assert np.isnan(fs_inf.y[cfg.main_horizon][k])
    assert np.isfinite(fs_inf.momentum[k]).all()   # features exist regardless
