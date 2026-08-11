import numpy as np

from ipo_model.data.preprocess import FoldScaler, engineered_table


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
        # "other" flag on exactly the rows using at least one rare bank
        assert (bk[:, -1] == (fs.bk[:, scaler.rare_bk].sum(1) > 0)).all()


def test_scalers_fit_on_train_only(fs, cfg):
    a = FoldScaler.fit(fs, np.arange(100), cfg.data)
    b = FoldScaler.fit(fs, np.arange(len(fs)), cfg.data)
    assert a.gpr_mu != b.gpr_mu  # different windows -> different stats
    g = a.gpr_sequences(fs)[np.arange(100)]
    assert abs(g.mean()) < 0.2  # train slice ~standardized under its own scaler


def test_engineered_table_shape_and_names(fs, cfg):
    scaler = FoldScaler.fit(fs, np.arange(len(fs) // 2), cfg.data)
    X = engineered_table(fs, scaler)
    assert len(X) == len(fs)
    assert X.notna().all().all()
    for col in ["panel_mean_cumret", "panel_mean_pop", "ipo_count_90d", "gpr_level"]:
        assert col in X.columns
