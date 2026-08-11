"""The tests that matter most: nothing observable after pricing may enter
features, and purging must keep training label windows out of test periods."""
import numpy as np

from ipo_model.data.splits import purged_walk_forward


def test_rows_sorted_and_labeled(fs):
    assert (np.diff(fs.dates.astype(np.int64)) >= 0).all()
    for h, y in fs.y.items():
        assert np.isfinite(y).all(), f"horizon {h} has non-finite labels"
    assert (fs.label_end > fs.dates).all()


def test_panel_uses_only_prior_ipos(fs, raw):
    """Every IPO in a target's panel must have listed strictly earlier, and its
    sequence must only cover dates strictly before the target's first trade."""
    first_trade = dict(zip(raw.ipos["ipo_id"], raw.ipos["first_trade_date"].to_numpy()))
    prices = raw.prices.sort_values(["ipo_id", "date"])
    dates_by_id = {k: v["date"].to_numpy() for k, v in prices.groupby("ipo_id")}
    id_by_date_order = sorted(first_trade, key=lambda k: first_trade[k])

    for i in range(0, len(fs), 37):  # sample of targets
        t_i = fs.dates[i]
        prior = [k for k in id_by_date_order if first_trade[k] < t_i]
        expected = prior[-fs.panel_seq.shape[1]:][::-1]  # most recent first
        for slot in range(fs.panel_valid.shape[1]):
            if not fs.panel_valid[i, slot]:
                continue
            jid = expected[slot]
            L = fs.panel_len[i, slot]
            assert L > 0
            # The L-th close of that IPO must predate the target's first trade.
            assert dates_by_id[jid][L - 1] < t_i
            # And the next close (if it exists) must NOT have been usable.
            dj = dates_by_id[jid]
            if L < min(len(dj), fs.panel_seq.shape[2]):
                assert dj[L] >= t_i


def test_gpr_window_strictly_before_pricing(fs, raw):
    gpr = raw.gpr.set_index("date")["gpr"]
    for i in range(0, len(fs), 53):
        t_i = fs.dates[i]
        last_val = fs.gpr_seq[i, -1]
        prior = gpr[gpr.index < t_i]
        assert np.isclose(last_val, prior.iloc[-1])


def test_purged_walk_forward_no_overlap(fs, cfg):
    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    assert len(folds) == cfg.split.n_folds
    embargo = np.timedelta64(cfg.split.embargo_days, "D")
    all_test = []
    for f in folds:
        # No train/val label window (plus embargo) may touch the test period.
        assert (fs.label_end[f.train_idx] + embargo < f.test_start).all()
        assert (fs.label_end[f.val_idx] + embargo < f.test_start).all()
        # Train labels must also clear the validation period.
        val_start = fs.dates[f.val_idx[0]]
        assert (fs.label_end[f.train_idx] + embargo < val_start).all()
        # No index reuse within a fold.
        assert not (set(f.train_idx) & set(f.val_idx))
        assert not (set(f.train_idx) & set(f.test_idx))
        assert not (set(f.val_idx) & set(f.test_idx))
        all_test.append(f.test_idx)
    # Test blocks tile the evaluation region without overlap.
    cat = np.concatenate(all_test)
    assert len(cat) == len(set(cat))


def test_targets_are_market_excess(fs, raw, cfg):
    """Recompute one label by hand from the raw CSVs."""
    i = len(fs) // 2
    ipo_id = fs.ids[i]
    row = raw.ipos[raw.ipos["ipo_id"] == ipo_id].iloc[0]
    px = raw.prices[raw.prices["ipo_id"] == ipo_id].sort_values("date")
    h = cfg.data.horizons[-1]
    gross = np.log(px["close"].iloc[h - 1] / row["offer_price"])
    mkt = raw.market.set_index("date")["close"]
    m_end = mkt[mkt.index <= px["date"].iloc[h - 1]].iloc[-1]
    m_prev = mkt[mkt.index < row["first_trade_date"]].iloc[-1]
    expected = gross - np.log(m_end / m_prev)
    assert np.isclose(fs.y[h][i], expected, atol=1e-10)
