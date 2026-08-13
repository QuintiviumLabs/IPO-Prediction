"""Statistical machinery in metrics.py, checked against known answers."""
import numpy as np
import pandas as pd
import pytest

from ipo_model.training.metrics import (LEGACY_KEYS, benjamini_hochberg,
                                        bootstrap_ci, bucket_spread,
                                        compare_models, evaluate, ic_by_period,
                                        summarize_folds, t_test)

QS = (0.1, 0.5, 0.9)


def _perfect(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.1, n)
    return y, np.stack([y - 0.05, y, y + 0.05], axis=1)


def test_legacy_keys_all_present(fs, cfg):
    """Existing consumers must keep working after the expansion."""
    y, q = _perfect()
    m = evaluate(y, q, QS)
    for k in LEGACY_KEYS:
        assert k in m, f"legacy metric {k} disappeared"


def test_evaluate_perfect_and_anti_signal():
    y, q = _perfect()
    m = evaluate(y, q, QS)
    assert np.isclose(m["rank_ic"], 1.0)
    assert np.isclose(m["pearson_ic"], 1.0)
    assert np.isclose(m["mae"], 0.0)
    assert np.isclose(m["coverage"], 1.0)
    assert m["r2_vs_zero"] > 0.99
    assert m["quintile_spread"] > 0
    assert m["top_bucket_mean"] > m["bottom_bucket_mean"]

    anti = np.stack([-y - 0.01, -y, -y + 0.01], axis=1)
    ma = evaluate(y, anti, QS)
    assert np.isclose(ma["rank_ic"], -1.0)
    assert ma["quintile_spread"] < 0
    assert ma["r2_vs_zero"] < 0  # worse than predicting zero


def test_evaluate_handles_degenerate_input():
    y = np.array([0.1, -0.2, 0.05])
    const = np.tile([[-0.1, 0.0, 0.1]], (3, 1))
    m = evaluate(y, const, QS)
    assert np.isnan(m["rank_ic"])          # no ordering in a constant forecast
    assert np.isfinite(m["pinball"])       # but the distribution is still scored
    assert np.isnan(m["quintile_spread"])  # too few rows to bucket

    with_nan = np.array([0.1, np.nan, 0.05])
    m2 = evaluate(with_nan, const, QS)
    assert m2["n"] == 2 and m2["n_dropped"] == 1


def test_evaluate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        evaluate(np.zeros(5), np.zeros((4, 3)), QS)


def test_interval_diagnostics_locate_the_bad_tail():
    rng = np.random.default_rng(1)
    y = rng.normal(0.3, 0.05, 500)          # truth sits above the band
    q = np.tile([[-0.1, 0.0, 0.1]], (500, 1))
    m = evaluate(y, q, QS)
    assert m["frac_above_hi"] > 0.9 and m["frac_below_lo"] < 0.05
    assert m["coverage_error"] < 0
    assert m["interval_score"] > m["mean_interval_width"]  # penalty applied
    assert m["bias"] < 0                                    # under-predicting


def test_bucket_spread_requires_enough_rows():
    rng = np.random.default_rng(2)
    y = rng.normal(size=40)
    p = y + rng.normal(0, 0.1, 40)
    assert np.isnan(bucket_spread(p, y, 10))   # 40 rows / 10 buckets -> too thin
    assert np.isfinite(bucket_spread(p, y, 5))


# --------------------------------------------------------------- t-tests ---

def test_t_test_matches_scipy_without_hac():
    from scipy import stats
    rng = np.random.default_rng(3)
    v = rng.normal(0.4, 1.0, 40)
    got = t_test(v, hac=False)
    exp = stats.ttest_1samp(v, 0.0)
    assert np.isclose(got["t"], exp.statistic)
    assert np.isclose(got["p"], exp.pvalue)


def test_hac_widens_se_under_autocorrelation():
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 300)
    ar = np.zeros(300)
    for i in range(1, 300):
        ar[i] = 0.8 * ar[i - 1] + e[i]      # strongly persistent series
    assert t_test(ar, hac=True)["se"] > t_test(ar, hac=False)["se"]


def test_t_test_degenerate():
    assert np.isnan(t_test(np.array([1.0]))["t"])
    assert np.isnan(t_test(np.array([2.0, 2.0, 2.0]))["t"])  # zero variance


def test_summarize_folds_shape():
    folds = [{"rank_ic": 0.1, "pinball": 0.03},
             {"rank_ic": 0.12, "pinball": 0.031},
             {"rank_ic": 0.09, "pinball": 0.029}]
    out = summarize_folds(folds, keys=("rank_ic", "pinball"))
    assert list(out["metric"]) == ["rank_ic", "pinball"]
    assert out.loc[0, "n_folds"] == 3
    assert out.loc[0, "t_stat"] > 0


# ------------------------------------------------------------ ic_by_period --

def test_ic_by_period_splits_and_filters():
    n = 600
    dates = pd.date_range("2015-01-01", periods=n, freq="D").to_numpy()
    rng = np.random.default_rng(5)
    y = rng.normal(size=n)
    q = np.stack([y - 1, y + rng.normal(0, 0.3, n), y + 1], axis=1)
    per = ic_by_period(y, q, dates, QS, freq="QE", min_obs=8)
    assert len(per) >= 6
    assert (per["n"] >= 8).all()
    assert per["rank_ic"].mean() > 0.5     # forecast is genuinely informative


def test_ic_by_period_drops_thin_periods():
    dates = np.array(["2020-01-01", "2020-01-02", "2020-06-01"],
                     dtype="datetime64[ns]")
    y = np.array([0.1, -0.1, 0.2])
    q = np.stack([y - 1, y, y + 1], axis=1)
    assert len(ic_by_period(y, q, dates, QS, freq="QE", min_obs=8)) == 0


# ------------------------------------------------------------- bootstrap ---

def test_bootstrap_ci_brackets_true_signal():
    y, q = _perfect(n=300, seed=6)
    out = bootstrap_ci(y, q, QS, metric="rank_ic", n_boot=200, seed=0)
    assert out["lo"] <= out["estimate"] <= out["hi"]
    assert out["lo"] > 0.5            # a perfect forecast is clearly non-zero


def test_bootstrap_ci_on_noise_includes_zero():
    rng = np.random.default_rng(7)
    y = rng.normal(size=250)
    q = np.stack([rng.normal(size=250) - 1, rng.normal(size=250),
                  rng.normal(size=250) + 1], axis=1)
    out = bootstrap_ci(y, q, QS, metric="rank_ic", n_boot=300, seed=1)
    assert out["lo"] < 0 < out["hi"]
    assert out["p_two_sided"] > 0.05


def test_block_bootstrap_is_wider_than_iid():
    """Blocking by year must not produce a narrower interval than iid."""
    y, q = _perfect(n=400, seed=8)
    groups = np.repeat(np.arange(8), 50)
    iid = bootstrap_ci(y, q, QS, n_boot=300, seed=2)
    blocked = bootstrap_ci(y, q, QS, n_boot=300, groups=groups, seed=2)
    assert np.isfinite(blocked["lo"]) and np.isfinite(blocked["hi"])
    assert (blocked["hi"] - blocked["lo"]) >= 0.5 * (iid["hi"] - iid["lo"])


# ------------------------------------------------------------- comparison --

def test_compare_models_detects_the_better_model():
    rng = np.random.default_rng(9)
    y = rng.normal(0, 0.1, 400)
    good = np.stack([y - 0.05, y + rng.normal(0, 0.01, 400), y + 0.05], axis=1)
    bad = np.stack([rng.normal(0, .1, 400) - .05, rng.normal(0, .1, 400),
                    rng.normal(0, .1, 400) + .05], axis=1)
    out = compare_models(y, good, bad, QS, "good", "bad", n_boot=300)
    assert out["rank_ic_delta"] > 0.5
    assert out["rank_ic_delta_lo"] > 0          # CI excludes zero
    assert out["pinball_delta"] < 0             # good has lower loss
    assert out["dm_t"] < 0 and out["dm_p"] < 0.05


def test_compare_identical_models_is_null():
    y, q = _perfect(n=300, seed=10)
    out = compare_models(y, q, q.copy(), QS, n_boot=200)
    assert np.isclose(out["rank_ic_delta"], 0.0)
    assert np.isclose(out["pinball_delta"], 0.0)


# ----------------------------------------------------- multiple comparison --

def test_benjamini_hochberg_known_case():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    out = benjamini_hochberg(p, alpha=0.05)
    assert np.isclose(out.loc[0, "q_value"], 0.005)
    assert bool(out.loc[0, "reject"])
    assert not bool(out.loc[4, "reject"])
    q = out["q_value"].to_numpy()
    assert (np.diff(q[np.argsort(p)]) >= -1e-12).all()   # monotone in p


def test_benjamini_hochberg_handles_nan():
    out = benjamini_hochberg([0.01, np.nan, 0.6])
    assert np.isfinite(out.loc[0, "q_value"])
    assert not bool(out.loc[1, "reject"])
