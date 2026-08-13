"""Walk-forward conformal calibration: correctness on a known miscalibration."""
import numpy as np
import pandas as pd
import pytest

from ipo_model.calibrate import calibrate, tail_report

QS = (0.2, 0.5, 0.8)


def _frame(n_folds=4, n=400, seed=0):
    """Right-skewed outcomes with a band that is too NARROW on the upside and
    slightly too WIDE on the downside — the measured real-data pattern."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_folds):
        y = rng.lognormal(mean=-1.8, sigma=0.9, size=n) - 0.12  # skewed right
        med = np.full(n, np.median(y)) + rng.normal(0, 0.01, n)
        rows.append(pd.DataFrame({
            "ipo_id": [f"d{k}_{i}" for i in range(n)],
            "fold": k, "y_true": y,
            "q0.2": med - 0.30,          # too wide below
            "q0.5": med,
            "q0.8": med + 0.06,          # clearly too narrow above
        }))
    return pd.concat(rows, ignore_index=True)


def test_asym_fixes_each_tail_separately():
    df = _frame()
    before = tail_report(df, QS, folds=[1, 2, 3])
    assert before["frac_above_hi"] > 0.3        # confirm the planted miss
    assert before["frac_below_lo"] < 0.10

    cal, rep = calibrate(df, QS, mode="asym", min_cal=80)
    after = tail_report(cal, QS, folds=[1, 2, 3])
    for tail, nominal in (("frac_above_hi", 0.2), ("frac_below_lo", 0.2)):
        assert abs(after[tail] - nominal) < abs(before[tail] - nominal), tail
    assert abs(after["frac_above_hi"] - 0.2) < 0.05
    assert abs(after["frac_below_lo"] - 0.2) < 0.05
    # margins moved in the right directions: upper widened, lower shrunk
    m = rep[rep["calibrated"]]
    assert (m["m_hi"] > 0).all()
    assert (m["m_lo"] < 0).all()


def test_sym_shares_the_correction_and_underperforms_asym():
    df = _frame(seed=1)
    cal_s, _ = calibrate(df, QS, mode="sym", min_cal=80)
    cal_a, _ = calibrate(df, QS, mode="asym", min_cal=80)
    s = tail_report(cal_s, QS, folds=[1, 2, 3])
    a = tail_report(cal_a, QS, folds=[1, 2, 3])
    err = lambda r: abs(r["frac_above_hi"] - .2) + abs(r["frac_below_lo"] - .2)
    assert err(a) < err(s)


def test_first_fold_passes_through_and_median_untouched():
    df = _frame()
    cal, rep = calibrate(df, QS, mode="asym", min_cal=80)
    assert not rep.loc[rep["fold"] == 0, "calibrated"].iloc[0]
    f0 = df["fold"] == 0
    pd.testing.assert_frame_equal(df[f0], cal[f0])
    pd.testing.assert_series_equal(df["q0.5"], cal["q0.5"])  # all folds


def test_edges_never_cross_median():
    df = _frame(seed=2)
    df["q0.2"] = df["q0.5"] - 0.9   # absurdly wide: big negative margin coming
    cal, _ = calibrate(df, QS, mode="asym", min_cal=80)
    assert (cal["q0.2"] <= cal["q0.5"] + 1e-12).all()
    assert (cal["q0.8"] >= cal["q0.5"] - 1e-12).all()


def test_min_cal_respected():
    df = _frame(n=30)  # 30 rows/fold: folds 1-2 lack history, fold 3 has 90
    _, rep = calibrate(df, QS, mode="asym", min_cal=80)
    assert list(rep["calibrated"]) == [False, False, False, True]


def test_rejects_bad_inputs():
    df = _frame()
    with pytest.raises(ValueError):
        calibrate(df, QS, mode="magic")
    with pytest.raises(ValueError):
        calibrate(df, (0.2, 0.8), mode="asym")
