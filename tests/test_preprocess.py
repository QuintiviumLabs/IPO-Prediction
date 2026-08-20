import numpy as np

from ipo_model.data.preprocess import FoldScaler, engineered_table, raw_table


def test_rare_bookrunner_bucketing(fs, cfg):
    train_idx = np.arange(len(fs) // 2)
    scaler = FoldScaler.fit(fs, train_idx, cfg.data)
    counts = fs.bk[train_idx].sum(axis=0)
    assert (counts[scaler.keep_bk] >= cfg.data.min_bookrunner_deals).all()
    assert (counts[scaler.rare_bk] < cfg.data.min_bookrunner_deals).all()
    bk = scaler.bookrunners(fs)
    expected_cols = len(scaler.keep_bk) + (1 if len(scaler.rare_bk) else 0)
    assert bk.shape == (len(fs), expected_cols)
    if len(scaler.rare_bk):
        assert (bk[:, -1] == (fs.bk[:, scaler.rare_bk].sum(1) > 0)).all()


def test_scalers_fit_on_train_only(fs, cfg):
    a = FoldScaler.fit(fs, np.arange(100), cfg.data)
    b = FoldScaler.fit(fs, np.arange(len(fs)), cfg.data)
    assert a.gpr_mu != b.gpr_mu  # different windows -> different stats
    g = a.gpr_sequences(fs)[np.arange(100)]
    assert abs(g.mean()) < 0.2  # train slice ~standardized under its own scaler
    m = a.momentum_matrix(fs)[np.arange(100)]
    assert abs(m.mean()) < 0.2


def test_engineered_table_contains_factor_block(fs, cfg):
    scaler = FoldScaler.fit(fs, np.arange(len(fs) // 2), cfg.data)
    X = engineered_table(fs, scaler)
    assert len(X) == len(fs)
    assert X.notna().all().all()
    for col in ["f1_med_1d", "f1_vw_1w", "f2_break_1d", "f2_depth_1w",
                "f3_cnt_3m", "f3_accel",
                "m_geo_mom_63d", "m_regime_hot", "gpr_vol63",
                "m_vol_index_z", "m_vix_z", "m_hy_oas_z", "m_em_vs_dm_63d",
                "p1_med_21d", "p1_n",
                "n_banks", "has_bb", "prestige_rank_max", "archetype_bb_dom",
                "gpr_level"]:
        assert col in X.columns
    # F4 (sector issuance density) was removed.
    assert not [c for c in X.columns if c.startswith("f4_")]
    # Market-agnostic by default: no market identity columns.
    for name in fs.market_names:
        assert f"mkt_{name}" not in X.columns


def test_market_onehot_flag(fs, cfg):
    """Market identity enters features only when explicitly enabled."""
    train_idx = np.arange(len(fs) // 2)
    off = FoldScaler.fit(fs, train_idx, cfg.data)
    on_cfg = cfg.override(**{"data.include_market_onehot": True})
    on = FoldScaler.fit(fs, train_idx, on_cfg.data)
    b_off, _ = off.static_matrix(fs)
    b_on, _ = on.static_matrix(fs)
    assert b_on.shape[1] == b_off.shape[1] + len(fs.market_names)
    X_on = engineered_table(fs, on)
    for name in fs.market_names:
        assert f"mkt_{name}" in X_on.columns


def test_raw_table_adds_gpr_window(fs, cfg):
    scaler = FoldScaler.fit(fs, np.arange(len(fs) // 2), cfg.data)
    X = raw_table(fs, scaler)
    base = engineered_table(fs, scaler)
    W = fs.gpr_seq.shape[1]
    assert X.shape[1] == base.shape[1] + W
    assert (X["gpr_m01"].to_numpy() == fs.gpr_seq[:, -1]).all()
