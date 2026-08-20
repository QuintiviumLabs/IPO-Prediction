"""Synthetic IPO dataset generator.

Produces CSVs in the exact schema the pipeline expects from real data, with
learnable structure planted so the modeling stack can be validated end-to-end:

- three markets, each with its own persistent "IPO market sentiment" factor
  driving recent-deal aftermarket performance AND future IPO returns in that
  market (what the F1/F2 momentum block should pick up);
- issuance clustered in hot markets, so supply/density factors (F3/F4) track
  a real regime variable;
- a geopolitical-risk (GPR) index whose *changes* depress aftermarket
  returns, more strongly for TMT deals (GPR arm / FiLM gating);
- per-bookrunner skill effects on aftermarket drift (static arm), with deals
  run by 1-3 banks encoded as multi-hot binary columns;
- log-normal deal sizes for value weighting.

Schema written to <out_dir>:
  ipos.csv          ipo_id, first_trade_date, offer_price, market, deal_size,
                    is_tmt, is_healthcare, subgroup, bk_*, plus the syndicate
                    block from build_bookrunner_features.py (n_banks,
                    log_n_banks, has_*, combo_*, solo_book, jumbo_synd,
                    archetype_*). prestige_rank_max is deliberately absent so
                    the pipeline's expanding computation is exercised.
  prices.csv        ipo_id, date, close    (daily closes, event window only)
  gpr.csv           date, gpr              (daily index level)
  market.csv        date, market, close    (one benchmark index per market)
  macro.csv         date, market, vol_index, fx
  macro_global.csv  date, vix, hy_oas, em, acwi
  peers.csv         ipo_id, peer, ret_21d, ret_63d, asof  (as-of trailing
                    peer returns, asof strictly before pricing; correlated
                    with the deal's own sentiment so p1 carries signal)
(deals.csv — the optional all-deal-types file — is not generated; the
pipeline falls back to IPO-only supply factors, the same path real data
takes until an ECM deals file is provided.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_BANKS = 12
# Bank classes for the syndicate block: 0-2 bulge bracket, 3-5 global,
# 6-8 regional, 9-11 local (mirrors configs/bank_classes.csv on real data).
BANK_CLASS = ["bb"] * 3 + ["global"] * 3 + ["regional"] * 3 + ["local"] * 3
MARKETS = ("US", "EU", "JP")
MARKET_P = (0.5, 0.3, 0.2)
N_SUBGROUPS = 12
PRICE_DAYS = 26  # closes generated per IPO: enough for a 21-day target + buffer


def generate(
    out_dir: str | Path,
    n_ipos: int = 4000,
    start: str = "2010-01-04",
    end: str = "2025-12-31",
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n_days = len(days)

    # Per-market benchmark indices (long format): a shared global factor plus
    # each market's own component, so benchmarks are correlated but distinct.
    base_ret = rng.normal(3e-4, 0.01, n_days)
    mkt_ret: dict[str, np.ndarray] = {}
    market_rows = []
    for m in MARKETS:
        r = 0.6 * base_ret + rng.normal(1e-4, 0.008, n_days)
        mkt_ret[m] = r
        market_rows.append(pd.DataFrame({
            "date": days, "market": m, "close": 100 * np.exp(np.cumsum(r)),
        }))
    market = pd.concat(market_rows, ignore_index=True)

    # GPR: OU process on log-level -> positive, persistent index.
    g = np.zeros(n_days)
    for t in range(1, n_days):
        g[t] = g[t - 1] + 0.03 * (0.0 - g[t - 1]) + 0.15 * rng.normal()
    gpr = pd.DataFrame({"date": days, "gpr": 100 * np.exp(g)})
    g_series = pd.Series(g)
    g_chg = (g_series - g_series.shift(21)).fillna(0.0).to_numpy()
    g_chg = g_chg / (g_chg.std() + 1e-12)

    # Per-market latent sentiment: persistent AR(1), partially shared.
    common = np.zeros(n_days)
    for t in range(1, n_days):
        common[t] = 0.985 * common[t - 1] + 0.17 * rng.normal()
    sent = {}
    for m in MARKETS:
        own = np.zeros(n_days)
        for t in range(1, n_days):
            own[t] = 0.985 * own[t - 1] + 0.17 * rng.normal()
        s = 0.6 * common + 0.8 * own
        sent[m] = s / (s.std() + 1e-12)

    # IPO dates & markets: issuance clusters when that market's sentiment is hot.
    lo, hi = 130, n_days - PRICE_DAYS - 5
    mkts = rng.choice(len(MARKETS), size=n_ipos, p=MARKET_P)
    date_idx = np.empty(n_ipos, dtype=int)
    for k, m in enumerate(MARKETS):
        mask = mkts == k
        w = np.exp(1.3 * sent[m][lo:hi])
        date_idx[mask] = rng.choice(np.arange(lo, hi), size=mask.sum(),
                                    p=w / w.sum())
    order = np.argsort(date_idx, kind="stable")
    date_idx, mkts = date_idx[order], mkts[order]

    # Bookrunners: popularity-skewed; each deal 1-3 banks; skill shifts drift.
    bank_pop = rng.dirichlet(np.full(N_BANKS, 1.2))
    bank_skill = rng.normal(0.0, 0.015, N_BANKS)
    bank_home = rng.choice(MARKETS, N_BANKS)   # for has_domestic
    n_banks_per_deal = rng.integers(1, 4, n_ipos)

    # Per-market local vol index and FX (macro.csv) + global risk series
    # (macro_global.csv). Vol indexes load on the shared factor's turbulence.
    vol_z = np.zeros(n_days)
    for t in range(1, n_days):
        vol_z[t] = 0.97 * vol_z[t - 1] + 0.24 * rng.normal()
    macro_rows = []
    for m in MARKETS:
        local = 0.7 * vol_z + 0.5 * np.convolve(
            np.abs(mkt_ret[m]), np.ones(5) / 5, mode="same") / 0.01
        fx = np.exp(np.cumsum(rng.normal(0, 0.004, n_days)))
        macro_rows.append(pd.DataFrame({
            "date": days, "market": m,
            "vol_index": np.round(16.0 * np.exp(0.35 * local), 3),
            "fx": np.round(fx, 5),
        }))
    macro = pd.concat(macro_rows, ignore_index=True)
    macro_global = pd.DataFrame({
        "date": days,
        "vix": np.round(17.0 * np.exp(0.4 * vol_z), 3),
        "hy_oas": np.round(4.0 + 1.6 * np.maximum(vol_z, 0)
                           + 0.3 * rng.normal(0, 1, n_days).cumsum() / 40, 4),
        "em": np.round(100 * np.exp(np.cumsum(0.5 * base_ret
                                              + rng.normal(0, 0.009, n_days))), 4),
        "acwi": np.round(100 * np.exp(np.cumsum(0.8 * base_ret
                                                + rng.normal(0, 0.005, n_days))), 4),
    })

    ipo_rows, price_rows, peer_rows = [], [], []
    for i in range(n_ipos):
        t0 = int(date_idx[i])
        m_name = MARKETS[mkts[i]]
        s = sent[m_name]
        is_tmt = int(rng.random() < 0.35)
        is_hc = int(rng.random() < 0.25 * (1 - is_tmt) + 0.10 * is_tmt)
        banks = rng.choice(N_BANKS, size=n_banks_per_deal[i], replace=False, p=bank_pop)
        skill = bank_skill[banks].mean()
        deal_size = float(np.round(np.exp(rng.normal(4.5, 0.9)), 2))  # ~$90m median

        offer = float(np.round(rng.uniform(12, 40), 2))
        pop = 0.035 + 0.05 * s[t0] + rng.normal(0, 0.08)
        t_idx = t0 + 1 + np.arange(PRICE_DAYS - 1)
        alpha = (
            0.0045 * s[t_idx]
            + skill / 21.0
            - 0.0022 * g_chg[t_idx] * (1.0 + 0.7 * is_tmt)
            + rng.normal(0, 0.02, PRICE_DAYS - 1)
        )
        r_m = mkt_ret[m_name]
        log_close = (
            np.log(offer)
            + pop
            + r_m[t0]
            + np.cumsum(np.concatenate([[0.0], alpha + r_m[t_idx]]))
        )
        closes = np.exp(log_close)

        ipo_id = f"ipo_{i:05d}"
        classes = {BANK_CLASS[b] for b in banks}
        has_bb = int("bb" in classes)
        has_domestic = int(any(bank_home[b] == m_name for b in banks))
        nb = int(n_banks_per_deal[i])
        row = {
            "ipo_id": ipo_id,
            "first_trade_date": days[t0],
            "offer_price": offer,
            "market": m_name,
            "deal_size": deal_size,
            "is_tmt": is_tmt,
            "is_healthcare": is_hc,
            "subgroup": f"SG{rng.integers(0, N_SUBGROUPS):02d}",
            # syndicate block (what build_bookrunner_features.py produces)
            "n_banks": nb,
            "log_n_banks": float(np.log1p(nb)),
            "has_bb": has_bb,
            "has_global": int("global" in classes),
            "has_regional": int("regional" in classes),
            "has_local": int("local" in classes),
            "has_domestic": has_domestic,
            "combo_bb_x_dom": int(has_bb and has_domestic),
            "combo_bb_x_reg": int(has_bb and "regional" in classes),
            "solo_book": int(nb == 1),
            "jumbo_synd": int(nb >= 3),
            "archetype_bb_dom": int(has_bb and has_domestic),
            "archetype_bb_intl": int(has_bb and not has_domestic),
            "archetype_other_dom": int(not has_bb and has_domestic),
            "archetype_other_intl": int(not has_bb and not has_domestic),
        }
        for b in range(N_BANKS):
            row[f"bk_{b:02d}"] = int(b in banks)
        ipo_rows.append(row)

        # Peers: as-of trailing returns of same-subgroup names, correlated
        # with this market's sentiment so the p1 block carries real signal.
        n_peers = int(rng.integers(3, 9))
        asof = days[t0 - 1]                       # strictly before pricing
        peer_rows.append(pd.DataFrame({
            "ipo_id": ipo_id,
            "peer": [f"PEER_{i:05d}_{j}" for j in range(n_peers)],
            "ret_21d": np.round(0.04 * s[t0 - 1]
                                + rng.normal(0, 0.06, n_peers), 5),
            "ret_63d": np.round(0.06 * s[t0 - 1]
                                + rng.normal(0, 0.10, n_peers), 5),
            "asof": asof,
        }))
        price_rows.append(pd.DataFrame({
            "ipo_id": ipo_id,
            "date": days[t0: t0 + PRICE_DAYS],
            "close": np.round(closes, 4),
        }))

    ipos = pd.DataFrame(ipo_rows)
    prices = pd.concat(price_rows, ignore_index=True)
    peers = pd.concat(peer_rows, ignore_index=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ipos.to_csv(out / "ipos.csv", index=False)
    prices.to_csv(out / "prices.csv", index=False)
    gpr.to_csv(out / "gpr.csv", index=False)
    market.to_csv(out / "market.csv", index=False)
    macro.to_csv(out / "macro.csv", index=False)
    macro_global.to_csv(out / "macro_global.csv", index=False)
    peers.to_csv(out / "peers.csv", index=False)
    return {"ipos": ipos, "prices": prices, "gpr": gpr, "market": market,
            "macro": macro, "macro_global": macro_global, "peers": peers}
