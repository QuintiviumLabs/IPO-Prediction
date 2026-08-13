"""is_target=0 rows must feed the F1-F4 factors but never be modelled.

This is what lets the recent-deal pool cover the WHOLE market while the
training set stays the curated deal list.
"""
import numpy as np

from ipo_model.data.features import RawData, build_features


def _variants(raw, cfg):
    """Same modelled deals, three universes:
      full    — every IPO is a target
      subset  — half the IPOs deleted outright
      widened — same half modelled, the other half kept as is_target=0
    """
    ipos = raw.ipos.sort_values("first_trade_date", kind="stable").reset_index(drop=True)
    keep = ipos.index % 2 == 0

    sub_ipos = ipos[keep].copy()
    sub_prices = raw.prices[raw.prices["ipo_id"].isin(sub_ipos["ipo_id"])]

    wide_ipos = ipos.copy()
    wide_ipos["is_target"] = keep.astype(int)

    def build(i, p):
        return build_features(RawData(ipos=i, prices=p, gpr=raw.gpr,
                                      market=raw.market, deals=raw.deals), cfg.data)

    return (build(ipos, raw.prices), build(sub_ipos, sub_prices),
            build(wide_ipos, raw.prices))


def test_universe_rows_excluded_from_modelling(raw, cfg):
    full, subset, widened = _variants(raw, cfg)
    assert len(widened) == len(subset), "is_target=0 rows must not be modelled"
    assert len(widened) < len(full)
    assert list(widened.ids) == list(subset.ids)


def test_universe_rows_still_drive_momentum(raw, cfg):
    """F1/F2 computed with the wide universe must match the full-data values,
    and must differ from the values a truncated universe would produce."""
    full, subset, widened = _variants(raw, cfg)
    pos = {i: k for k, i in enumerate(full.ids)}
    take = np.array([pos[i] for i in widened.ids])
    f1_cols = [k for k, n in enumerate(full.momentum_names)
               if n.startswith(("f1_", "f2_"))]

    np.testing.assert_allclose(widened.momentum[:, f1_cols],
                               full.momentum[take][:, f1_cols], atol=1e-6)
    # ...and the thinner universe genuinely gives different momentum readings
    assert not np.allclose(subset.momentum[:, f1_cols],
                           widened.momentum[:, f1_cols], atol=1e-6)


def test_universe_rows_may_omit_optional_columns(raw, cfg):
    """Universe rows can leave deal_size / sector / bookrunner blank."""
    ipos = raw.ipos.sort_values("first_trade_date", kind="stable").reset_index(drop=True)
    keep = ipos.index % 2 == 0
    ipos = ipos.copy()
    ipos["is_target"] = keep.astype(int)
    bk = [c for c in ipos.columns if c.startswith("bk_")]
    ipos.loc[~keep, ["deal_size", "is_tmt", "is_healthcare"] + bk] = np.nan

    fs = build_features(RawData(ipos=ipos, prices=raw.prices, gpr=raw.gpr,
                                market=raw.market, deals=raw.deals), cfg.data)
    assert len(fs) == int(keep.sum())
    assert np.isfinite(fs.momentum).all()
    assert np.isfinite(fs.bk).all() and np.isfinite(fs.sector).all()
