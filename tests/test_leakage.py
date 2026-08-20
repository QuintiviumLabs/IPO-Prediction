"""The tests that matter most: nothing observable after pricing may enter
features, and purging must keep training label windows out of test periods."""
import numpy as np

from ipo_model.data.features import build_features
from ipo_model.data.splits import purged_walk_forward


def test_rows_sorted_and_labeled(fs):
    assert (np.diff(fs.dates.astype(np.int64)) >= 0).all()
    for h, y in fs.y.items():
        assert np.isfinite(y).all(), f"horizon {h} has non-finite labels"
    assert (fs.label_end > fs.dates).all()


def test_f1_momentum_recomputed_by_hand(fs, raw, cfg):
    """Recompute F1_med_1d / F1_vw_1d for sampled targets straight from the
    CSVs, applying the observability rule (h-th close strictly before t0)."""
    ipos = raw.ipos.sort_values("first_trade_date", kind="stable").reset_index(drop=True)
    prices = raw.prices.sort_values(["ipo_id", "date"])
    dates_by_id = {k: v["date"].to_numpy() for k, v in prices.groupby("ipo_id")}
    close_by_id = {k: v["close"].to_numpy() for k, v in prices.groupby("ipo_id")}
    offer = dict(zip(ipos["ipo_id"], ipos["offer_price"]))
    med_col = fs.momentum_names.index("f1_med_1d")
    vw_col = fs.momentum_names.index("f1_vw_1d")
    win = np.timedelta64(cfg.data.momentum_window_days, "D")

    for i in range(10, len(fs), 41):
        t0, mkt = fs.dates[i], None
        row = ipos[ipos["ipo_id"] == fs.ids[i]].iloc[0]
        mkt = row["market"]
        cand = ipos[(ipos["first_trade_date"] < t0)
                    & (ipos["first_trade_date"] >= t0 - win)
                    & (ipos["market"] == mkt)]
        # stable sort: same-date ties must break identically to the builder
        cand = cand.sort_values("first_trade_date", kind="stable")
        cand = cand.iloc[::-1][: cfg.data.momentum_max_deals]
        rets, w = [], []
        for _, r in cand.iterrows():
            d = dates_by_id[r["ipo_id"]]
            if (d < t0).sum() >= 1:  # first close observable
                rets.append(np.log(close_by_id[r["ipo_id"]][0] / offer[r["ipo_id"]]))
                w.append(r["deal_size"])
        if not rets:
            continue
        assert np.isclose(fs.momentum[i, med_col], np.median(rets), atol=1e-6)
        vw = np.dot(rets, w) / np.sum(w)
        assert np.isclose(fs.momentum[i, vw_col], vw, atol=1e-6)


def test_prefix_stability_no_future_dependence(raw, cfg):
    """Momentum factors (incl. expanding z-scores) for early rows must be
    identical whether or not later IPOs exist in the dataset — the strongest
    form of the no-lookahead property."""
    full = build_features(raw, cfg.data)
    cutoff = full.dates[int(len(full) * 0.6)]
    trunc_ipos = raw.ipos[raw.ipos["first_trade_date"] <= cutoff]
    trunc_prices = raw.prices[raw.prices["ipo_id"].isin(trunc_ipos["ipo_id"])]
    from ipo_model.data.features import RawData
    trunc_peers = (raw.peers[raw.peers["ipo_id"].isin(trunc_ipos["ipo_id"])]
                   if raw.peers is not None else None)
    part = build_features(
        RawData(ipos=trunc_ipos, prices=trunc_prices, gpr=raw.gpr,
                market=raw.market, deals=raw.deals, macro=raw.macro,
                macro_global=raw.macro_global, peers=trunc_peers),
        cfg.data,
    )
    n = len(part)
    assert n > 50
    assert (full.ids[:n] == part.ids[:n]).all()
    np.testing.assert_allclose(full.momentum[:n], part.momentum[:n], atol=1e-6)
    # the expanding prestige computation must be prefix-stable too
    np.testing.assert_allclose(full.static_extra[:n], part.static_extra[:n],
                               atol=1e-12)


def test_gpr_window_strictly_before_pricing(fs, raw):
    gpr = raw.gpr.set_index("date")["gpr"]
    for i in range(0, len(fs), 53):
        t_i = fs.dates[i]
        prior = gpr[gpr.index < t_i]
        assert np.isclose(fs.gpr_seq[i, -1], prior.iloc[-1])


def test_purged_walk_forward_no_overlap(fs, cfg):
    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    assert len(folds) == cfg.split.n_folds
    embargo = np.timedelta64(cfg.split.embargo_days, "D")
    all_test = []
    for f in folds:
        assert (fs.label_end[f.train_idx] + embargo < f.test_start).all()
        assert (fs.label_end[f.val_idx] + embargo < f.test_start).all()
        val_start = fs.dates[f.val_idx[0]]
        assert (fs.label_end[f.train_idx] + embargo < val_start).all()
        assert not (set(f.train_idx) & set(f.val_idx))
        assert not (set(f.train_idx) & set(f.test_idx))
        assert not (set(f.val_idx) & set(f.test_idx))
        all_test.append(f.test_idx)
    cat = np.concatenate(all_test)
    assert len(cat) == len(set(cat))


def test_targets_are_own_market_benchmark_relative(fs, raw, cfg):
    """Recompute labels by hand: each IPO must be adjusted by ITS market's
    benchmark, at several horizons, for IPOs from different markets."""
    h_main = cfg.data.horizons[-1]
    checked_markets = set()
    for i in range(0, len(fs), 97):
        ipo_id = fs.ids[i]
        row = raw.ipos[raw.ipos["ipo_id"] == ipo_id].iloc[0]
        px = raw.prices[raw.prices["ipo_id"] == ipo_id].sort_values("date")
        bench = raw.market
        if "market" in bench.columns:
            bench = bench[bench["market"] == row["market"]]
        mkt = bench.sort_values("date").set_index("date")["close"]
        for h in cfg.data.horizons:
            gross = np.log(px["close"].iloc[h - 1] / row["offer_price"])
            m_end = mkt[mkt.index <= px["date"].iloc[h - 1]].iloc[-1]
            m_prev = mkt[mkt.index < row["first_trade_date"]].iloc[-1]
            expected = gross - np.log(m_end / m_prev)
            assert np.isclose(fs.y[h][i], expected, atol=1e-10), \
                f"{ipo_id} ({row['market']}) horizon {h}"
        checked_markets.add(row["market"])
    assert len(checked_markets) >= 2, "sample should span multiple markets"


def test_exp_of_target_is_outperformance_multiple(fs, raw, cfg):
    """exp(y) must equal (gross stock multiple) / (gross benchmark multiple)."""
    i = len(fs) // 3
    ipo_id = fs.ids[i]
    row = raw.ipos[raw.ipos["ipo_id"] == ipo_id].iloc[0]
    px = raw.prices[raw.prices["ipo_id"] == ipo_id].sort_values("date")
    bench = raw.market[raw.market["market"] == row["market"]] \
        if "market" in raw.market.columns else raw.market
    mkt = bench.sort_values("date").set_index("date")["close"]
    h = cfg.data.horizons[-1]
    stock_mult = px["close"].iloc[h - 1] / row["offer_price"]
    m_end = mkt[mkt.index <= px["date"].iloc[h - 1]].iloc[-1]
    m_prev = mkt[mkt.index < row["first_trade_date"]].iloc[-1]
    bench_mult = m_end / m_prev
    assert np.isclose(np.exp(fs.y[h][i]), stock_mult / bench_mult, rtol=1e-9)
