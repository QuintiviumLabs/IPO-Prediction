"""Leakage-safe feature construction.

Every feature for a target IPO uses only data strictly before its
first_trade_date (pricing happens the evening before trading starts, so
nothing dated on or after the first trade date is observable at pricing).

Targets
-------
y_h = log((P_h / P_anchor) / (I_h / I_anchor)) — log outperformance vs the
IPO's OWN market's benchmark index (market.csv in long format date, market,
close; a single global (date, close) series is also accepted). exp(y_h) is
the outperformance multiple. The last configured horizon is the main target.

`data.anchor` sets P_anchor:
  "offer"        offer price (includes the first-day pop).
  "first_close"  the first daily close (aftermarket drift only; horizon 1 is
                 degenerate — use horizons like (3, 5, 21)).
F1/F2 momentum factors stay offer-anchored either way.

Momentum universe
-----------------
ipos.csv may carry an optional `is_target` column. Rows with is_target=0 are
"momentum universe" deals: they feed the market-state factors but never
become training/test examples.

Market-state factor blocks (grouped by name prefix; toggled via
model.momentum_groups)
-----------------------------------------------------------------
f1 — Recent deal performance: over the ≤K most recent same-market IPOs in 90
  days, median and value-weighted returns from offer at 1d/1w, plus count
  and IQR. Only returns observable before the target's pricing enter.
f2 — Break-issue rate: share of that deal set below offer at 1d/1w, and mean
  downside among the breakers.
f3 — Rolling supply: same-market deal count, log proceeds, IPO count (90d),
  issuance acceleration vs the trailing year. Expanding market-specific
  z-scores. Uses deals.csv (all deal types: date, market, proceeds) when
  present, else the IPO record itself.
m  — Macro block:
  from market.csv (per-market benchmark): m_geo_mom_63d, m_geo_dd_252,
    m_geo_rvol_21d;
  (GPR-derived features live in the GPR ARM, not here, so model.use_gpr
  removes ALL geopolitical-risk information in one switch);
  from macro.csv (date, market, vol_index, fx — optional file/columns):
    m_vol_index_z (252d rolling z), m_fx_ret_21d;
  from macro_global.csv (date, vix, hy_oas, em, acwi — optional):
    m_vix_z, m_vix_chg_21d, m_hy_oas_z, m_em_vs_dm_63d;
  regime: m_regime_hot / m_regime_cold — mean first-day pop of the prior
    `regime_pop_window` same-market IPOs, cut by EXPANDING terciles of that
    market's own history (neutral until ≥5 prior observations).
p1 — Industry-subgroup peers (peers.csv, produced by scripts/pull_peers.py):
  per-IPO aggregates of the as-of trailing returns of up to K peer companies
  in the same Bloomberg industry subgroup: p1_med_21d, p1_med_63d,
  p1_iqr_21d, p1_n. peers.csv rows must carry features computed strictly
  before the IPO's pricing date (validated via the asof column).

Static extras (bookrunner syndicate block)
------------------------------------------
Columns produced by scripts/build_bookrunner_features.py when present in
ipos.csv: n_banks, log_n_banks, has_bb, has_global, has_regional, has_local,
has_domestic, combo_bb_x_dom, combo_bb_x_reg, solo_book, jumbo_synd, and
archetype_* one-hots. prestige_rank_max (max bookrunner percentile of
trailing-3y same-market IPO proceeds share) is computed HERE, expanding and
strictly pre-pricing, unless the column is supplied.

GPR arm
-------
The last `gpr_window` daily values strictly before the first trade date,
plus engineered summaries (level, d5, dW, meanW, stdW).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_model.config import DataConfig

# Engineered GPR summaries. ORDER MATTERS: gpr_vol63 (63d SD of daily
# changes, annualized — the slow risk-regime signal) must stay LAST — the
# gru-mode GPR arm consumes the sequence plus this final column.
GPR_FEAT_NAMES = ["gpr_level", "gpr_d5", "gpr_dW", "gpr_meanW", "gpr_stdW",
                  "gpr_vol63"]

# Bookrunner-syndicate columns consumed from ipos.csv when present.
STATIC_EXTRA_BINARY = ["has_bb", "has_global", "has_regional", "has_local",
                       "has_domestic", "combo_bb_x_dom", "combo_bb_x_reg",
                       "solo_book", "jumbo_synd"]
STATIC_EXTRA_CONTINUOUS = ["n_banks", "log_n_banks", "prestige_rank_max"]
ARCHETYPE_PREFIX = "archetype_"


@dataclass
class RawData:
    ipos: pd.DataFrame
    prices: pd.DataFrame
    gpr: pd.DataFrame
    market: pd.DataFrame
    deals: pd.DataFrame | None = None         # optional: all deal types for F3
    macro: pd.DataFrame | None = None         # optional: date, market, vol_index, fx
    macro_global: pd.DataFrame | None = None  # optional: date, vix, hy_oas, em, acwi
    peers: pd.DataFrame | None = None         # optional: ipo_id, ret_21d, ret_63d, asof


def load_raw(data_dir: str | Path) -> RawData:
    d = Path(data_dir)
    ipos = pd.read_csv(d / "ipos.csv", parse_dates=["first_trade_date"])
    prices = pd.read_csv(d / "prices.csv", parse_dates=["date"])
    gpr = pd.read_csv(d / "gpr.csv", parse_dates=["date"]).sort_values("date")
    market = pd.read_csv(d / "market.csv", parse_dates=["date"]).sort_values("date")

    def opt(name: str, dates=("date",)) -> pd.DataFrame | None:
        path = d / name
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=[c for c in dates if c])
        return df.sort_values(dates[0]) if dates[0] in df.columns else df

    deals = opt("deals.csv")
    macro = opt("macro.csv")
    macro_global = opt("macro_global.csv")
    peers = None
    if (d / "peers.csv").exists():
        peers = pd.read_csv(d / "peers.csv")
        if "asof" in peers.columns:
            peers["asof"] = pd.to_datetime(peers["asof"])
    return RawData(ipos=ipos, prices=prices, gpr=gpr, market=market,
                   deals=deals, macro=macro, macro_global=macro_global,
                   peers=peers)


@dataclass
class FeatureSet:
    """All model inputs as aligned numpy arrays, rows sorted by first_trade_date."""
    ids: np.ndarray               # (N,) str
    dates: np.ndarray             # (N,) datetime64[ns] — first trade date
    label_end: np.ndarray         # (N,) datetime64[ns] — date of main-horizon close
    y: dict[int, np.ndarray]      # horizon -> (N,) float
    sector: np.ndarray            # (N, S) float — binary sector flags
    sector_names: list[str]
    market_onehot: np.ndarray     # (N, M) float
    market_names: list[str]
    bk: np.ndarray                # (N, B) float — bookrunner multi-hot
    bk_names: list[str]
    n_bk: np.ndarray              # (N,) float — number of bookrunners
    static_extra: np.ndarray      # (N, E) float — syndicate block (may be E=0)
    static_extra_names: list[str]
    static_extra_continuous: list[bool]   # which extras need standardization
    momentum: np.ndarray          # (N, F) float32 — f1/f2/f3/m/p1 blocks
    momentum_names: list[str]
    gpr_seq: np.ndarray           # (N, W) float32 — raw GPR levels
    gpr_feats: np.ndarray         # (N, 6) float32 — see GPR_FEAT_NAMES

    def __len__(self) -> int:
        return len(self.ids)

    def momentum_columns(self, groups: tuple[str, ...]) -> np.ndarray:
        idx = [i for i, n in enumerate(self.momentum_names)
               if n.split("_")[0] in groups]
        return self.momentum[:, idx]


class _Expanding:
    """Expanding-window z-score: standardizes each value against strictly
    earlier values of the same series (Welford). First two values get z=0."""

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def z(self, x: float) -> float:
        if self.n >= 2:
            sd = (self.m2 / (self.n - 1)) ** 0.5
            out = (x - self.mean) / sd if sd > 1e-12 else 0.0
        else:
            out = 0.0
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        return out


class _ExpandingTerciles:
    """Hot/neutral/cold vs the EXPANDING distribution of a market's own
    history. Classification uses strictly prior values; needs >=5 of them."""

    def __init__(self) -> None:
        self._sorted: list[float] = []

    def classify(self, x: float) -> int:
        """-1 cold, 0 neutral, +1 hot; then records x for future calls."""
        out = 0
        n = len(self._sorted)
        if n >= 5:
            lo = self._sorted[int(np.floor(n / 3))]
            hi = self._sorted[int(np.floor(2 * n / 3))]
            out = 1 if x > hi else (-1 if x < lo else 0)
        bisect.insort(self._sorted, x)
        return out


def _asof_idx(dates: np.ndarray, t0) -> int:
    """Index of the last observation strictly before t0 (-1 if none)."""
    return int(np.searchsorted(dates, t0, side="left")) - 1


def _series_ret(logvals: np.ndarray, idx: int, lag: int) -> float:
    if idx < lag or idx < 0:
        return 0.0
    return float(logvals[idx] - logvals[idx - lag])


def _rolling_z(vals: np.ndarray, idx: int, win: int = 252, min_obs: int = 40) -> float:
    if idx < min_obs:
        return 0.0
    w = vals[max(0, idx - win + 1): idx + 1]
    sd = w.std()
    return float((vals[idx] - w.mean()) / sd) if sd > 1e-12 else 0.0


def build_features(raw: RawData, cfg: DataConfig,
                   inference: bool = False) -> FeatureSet:
    """Build the aligned FeatureSet.

    inference=False (training): only target rows with a full main-horizon
    label window are kept. inference=True: every is_target row is kept even
    without labels (missing horizons become NaN) — this is how a brand-new
    IPO with no trading history yet gets its feature row for scripts/predict.py.
    """
    ipos = raw.ipos.sort_values("first_trade_date", kind="stable").reset_index(drop=True)
    for col in ("market", "deal_size"):
        if col not in ipos.columns:
            raise ValueError(f"ipos.csv is missing required column '{col}'")
    # Optional momentum-universe rows: is_target=0 rows feed the market-state
    # factors but never become training/test examples themselves.
    is_target = (ipos["is_target"].fillna(1).to_numpy(float) != 0
                 if "is_target" in ipos.columns
                 else np.ones(len(ipos), dtype=bool))
    horizons = sorted(cfg.horizons)
    main_h = horizons[-1]

    # Benchmark log-close series, one per market (long format: date, market,
    # close) or a single global series (date, close) applied to every market.
    bench: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "market" in raw.market.columns:
        for m, g in raw.market.sort_values("date").groupby("market"):
            bench[str(m)] = (g["date"].to_numpy(),
                             np.log(g["close"].to_numpy(float)))
        missing_bench = set(ipos["market"].astype(str)) - set(bench)
        if missing_bench:
            raise ValueError("market.csv has no benchmark series for markets: "
                             f"{sorted(missing_bench)}")
    else:
        g = raw.market.sort_values("date")
        bench["__global__"] = (g["date"].to_numpy(),
                               np.log(g["close"].to_numpy(float)))

    def get_bench(market: str) -> tuple[np.ndarray, np.ndarray]:
        return bench[market] if market in bench else bench["__global__"]

    market_of = dict(zip(ipos["ipo_id"], ipos["market"].astype(str)))

    # ---- Per-IPO price series -> cumulative returns from offer ----
    prices = raw.prices.sort_values(["ipo_id", "date"], kind="stable")
    offer = dict(zip(ipos["ipo_id"], ipos["offer_price"].astype(float)))
    if cfg.anchor not in ("offer", "first_close"):
        raise ValueError(f"data.anchor must be 'offer' or 'first_close', "
                         f"got {cfg.anchor!r}")
    if cfg.anchor == "first_close" and 1 in horizons:
        raise ValueError(
            "horizon 1 is degenerate under anchor='first_close' "
            "(close_1 / close_1 = 0). Use horizons like (3, 5, 21); the "
            "first-day pop is what this anchor deliberately removes.")

    ipo_dates: dict[str, np.ndarray] = {}
    cum_raw: dict[str, np.ndarray] = {}      # offer-anchored raw (F1/F2 spec)
    cum_raw_excess: dict[str, np.ndarray] = {}  # offer-anchored, benchmark-rel.
    cum_target: dict[str, np.ndarray] = {}   # anchor-dependent (the labels)
    for ipo_id, g in prices.groupby("ipo_id", sort=False):
        if ipo_id not in market_of:
            continue  # price rows for unknown IPOs are ignored
        d = g["date"].to_numpy()
        log_c = np.log(g["close"].to_numpy(float))
        b_dates, b_log = get_bench(market_of[ipo_id])
        pos = np.searchsorted(b_dates, d, side="right") - 1
        if (pos < 0).any():
            raise ValueError(f"Benchmark for market '{market_of[ipo_id]}' does "
                             f"not cover the earliest price dates of {ipo_id}.")
        m = b_log[pos]
        prev_pos = np.searchsorted(b_dates, d[0], side="left") - 1
        m_prev = b_log[prev_pos] if prev_pos >= 0 else m[0]

        # F1/F2 always measure from the offer price, whatever the target does.
        raw_cum = log_c - np.log(offer[ipo_id])
        ipo_dates[ipo_id] = d
        cum_raw[ipo_id] = raw_cum
        cum_raw_excess[ipo_id] = (raw_cum - (m - m_prev) if cfg.market_adjust
                                  else raw_cum)
        if cfg.anchor == "offer":
            cum_target[ipo_id] = cum_raw_excess[ipo_id]
        else:
            tgt = log_c - log_c[0]
            cum_target[ipo_id] = (tgt - (m - m[0]) if cfg.market_adjust else tgt)

    # ---- Static blocks ----
    bk_names = sorted(c for c in ipos.columns if c.startswith(cfg.bookrunner_prefix))
    sector_names = [c for c in cfg.sector_cols if c in ipos.columns]
    missing = set(cfg.sector_cols) - set(sector_names)
    if missing:
        raise ValueError(f"ipos.csv is missing sector columns: {sorted(missing)}")

    all_ids = ipos["ipo_id"].to_numpy()
    all_dates = ipos["first_trade_date"].to_numpy()
    all_market = ipos["market"].astype(str).to_numpy()
    # Momentum-universe rows may lack deal size / sector / bookrunner detail;
    # missing values become 0 rather than propagating NaN.
    all_size = ipos["deal_size"].fillna(0.0).to_numpy(float)
    sector_all = ipos[sector_names].fillna(0.0).to_numpy(float)
    bk_all = (ipos[bk_names].fillna(0.0).to_numpy(float) if bk_names
              else np.zeros((len(ipos), 0)))
    market_names = sorted(set(all_market))
    market_onehot_all = np.stack(
        [(all_market == m).astype(float) for m in market_names], axis=1)

    # ---- Static extras: syndicate block from ipos.csv columns ----
    extra_names: list[str] = []
    extra_cont: list[bool] = []
    extra_cols: list[np.ndarray] = []
    for name in STATIC_EXTRA_CONTINUOUS:
        if name == "prestige_rank_max":
            continue  # handled below (computed if absent)
        if name in ipos.columns:
            extra_names.append(name)
            extra_cont.append(True)
            extra_cols.append(ipos[name].fillna(0.0).to_numpy(float))
    for name in STATIC_EXTRA_BINARY:
        if name in ipos.columns:
            extra_names.append(name)
            extra_cont.append(False)
            extra_cols.append(ipos[name].fillna(0.0).to_numpy(float))
    for name in sorted(c for c in ipos.columns if c.startswith(ARCHETYPE_PREFIX)):
        extra_names.append(name)
        extra_cont.append(False)
        extra_cols.append(ipos[name].fillna(0.0).to_numpy(float))

    # prestige_rank_max: supplied column wins; else computed expanding from
    # the bookrunner multi-hot — a bank's trailing-3y share of same-market
    # IPO proceeds, percentile-ranked among banks active in that window; the
    # deal takes the max over its bookrunners. Strictly pre-pricing.
    if "prestige_rank_max" in ipos.columns:
        extra_names.append("prestige_rank_max")
        extra_cont.append(True)
        extra_cols.append(ipos["prestige_rank_max"].fillna(0.0).to_numpy(float))
    elif len(bk_names):
        win_prestige = np.timedelta64(int(cfg.prestige_years * 365.25), "D")
        prestige = np.zeros(len(ipos))
        for i in range(len(ipos)):
            deal_banks = np.where(bk_all[i] > 0)[0]
            if not len(deal_banks):
                continue
            lo = np.searchsorted(all_dates, all_dates[i] - win_prestige, "left")
            hi = np.searchsorted(all_dates, all_dates[i], "left")
            sel = np.arange(lo, hi)
            sel = sel[all_market[sel] == all_market[i]]
            if not len(sel):
                continue
            shares = bk_all[sel].T @ all_size[sel]          # (B,) proceeds per bank
            active = shares > 0
            if not active.any():
                continue
            ranks = np.zeros_like(shares)
            order = shares[active].argsort().argsort()       # 0..n_active-1
            ranks[active] = (order + 1) / active.sum()       # percentile (0,1]
            prestige[i] = float(ranks[deal_banks].max())
        extra_names.append("prestige_rank_max")
        extra_cont.append(True)
        extra_cols.append(prestige)
    static_extra_all = (np.stack(extra_cols, axis=1) if extra_cols
                        else np.zeros((len(ipos), 0)))

    # ---- Deal universe for F3 (all deal types if deals.csv provided) ----
    if raw.deals is not None:
        for col in ("date", "market", "proceeds"):
            if col not in raw.deals.columns:
                raise ValueError(f"deals.csv is missing required column '{col}'")
        deal_dates = raw.deals["date"].to_numpy()
        deal_market = raw.deals["market"].astype(str).to_numpy()
        deal_proceeds = raw.deals["proceeds"].to_numpy(float)
    else:  # fall back: the IPO record IS the deal universe
        deal_dates, deal_market, deal_proceeds = all_dates, all_market, all_size
    order = np.argsort(deal_dates, kind="stable")
    deal_dates, deal_market = deal_dates[order], deal_market[order]
    deal_proceeds = deal_proceeds[order]

    def deals_between(lo: np.datetime64, hi: np.datetime64) -> slice:
        return slice(np.searchsorted(deal_dates, lo, side="left"),
                     np.searchsorted(deal_dates, hi, side="left"))

    # ---- Macro sources ----
    gpr_dates = raw.gpr["date"].to_numpy()
    gpr_vals = raw.gpr["gpr"].to_numpy(float)

    macro_mkt: dict[str, dict[str, np.ndarray]] = {}
    macro_cols: list[str] = []
    if raw.macro is not None:
        if not {"date", "market"} <= set(raw.macro.columns):
            raise ValueError("macro.csv needs columns (date, market, ...)")
        macro_cols = [c for c in ("vol_index", "fx") if c in raw.macro.columns]
        for m, g in raw.macro.sort_values("date").groupby("market"):
            entry = {"date": g["date"].to_numpy()}
            for c in macro_cols:
                entry[c] = g[c].to_numpy(float)
            macro_mkt[str(m)] = entry

    glob: dict[str, np.ndarray] = {}
    glob_cols: list[str] = []
    if raw.macro_global is not None:
        if "date" not in raw.macro_global.columns:
            raise ValueError("macro_global.csv needs a date column")
        g = raw.macro_global.sort_values("date")
        glob["date"] = g["date"].to_numpy()
        glob_cols = [c for c in ("vix", "hy_oas", "em", "acwi") if c in g.columns]
        for c in glob_cols:
            glob[c] = g[c].to_numpy(float)

    # ---- Peer aggregates (p1) from peers.csv ----
    peer_stats: dict[str, dict[str, float]] = {}
    have_peers = raw.peers is not None and len(raw.peers)
    if have_peers:
        pr = raw.peers
        if "ipo_id" not in pr.columns or "ret_21d" not in pr.columns:
            raise ValueError("peers.csv needs at least (ipo_id, ret_21d); "
                             "run scripts/pull_peers.py to produce it")
        # The asof column is MANDATORY: without it there is no way to prove
        # the peer returns were observable at pricing, and silent look-ahead
        # here would inflate IC. pull_peers.py always writes it.
        if "asof" not in pr.columns:
            raise ValueError(
                "peers.csv has no asof column — cannot verify the peer "
                "returns are strictly pre-pricing. Re-pull with "
                "scripts/pull_peers.py, or add asof (the last price date "
                "used) to every row.")
        ft = ipos.set_index("ipo_id")["first_trade_date"]
        joined = pr.join(ft, on="ipo_id", how="inner")
        bad = joined[joined["asof"] >= joined["first_trade_date"]]
        if len(bad):
            raise ValueError(
                f"peers.csv: {len(bad)} rows have asof >= first_trade_date "
                f"(e.g. {bad['ipo_id'].iloc[0]}) — peer features must be "
                "computed strictly before pricing")
        for ipo_id, g in pr.groupby("ipo_id"):
            r21 = g["ret_21d"].dropna().to_numpy(float)
            r63 = (g["ret_63d"].dropna().to_numpy(float)
                   if "ret_63d" in g.columns else np.array([]))
            st = {"p1_n": float(len(r21))}
            st["p1_med_21d"] = float(np.median(r21)) if len(r21) else 0.0
            st["p1_iqr_21d"] = (float(np.percentile(r21, 75) - np.percentile(r21, 25))
                                if len(r21) >= 2 else 0.0)
            st["p1_med_63d"] = float(np.median(r63)) if len(r63) else 0.0
            peer_stats[str(ipo_id)] = st

    # ---- Factor names ----
    mh = sorted(cfg.momentum_horizons)
    tag = {1: "1d", 3: "3d", 5: "1w", 21: "1m"}
    f1_names, f2_names = [], []
    for h in mh:
        t = tag.get(h, f"{h}d")
        f1_names += [f"f1_med_{t}", f"f1_vw_{t}"]
        if cfg.momentum_extended:
            f1_names += [f"f1_n_{t}", f"f1_iqr_{t}"]
        f2_names += [f"f2_break_{t}", f"f2_depth_{t}"]
    f3_names = ["f3_cnt_3m", "f3_prc_3m", "f3_ipo_cnt_3m", "f3_accel"]
    m_names = ["m_geo_mom_63d", "m_geo_dd_252", "m_geo_rvol_21d",
               "m_regime_hot", "m_regime_cold"]
    if "vol_index" in macro_cols:
        m_names.append("m_vol_index_z")
    if "fx" in macro_cols:
        m_names.append("m_fx_ret_21d")
    if "vix" in glob_cols:
        m_names += ["m_vix_z", "m_vix_chg_21d"]
    if "hy_oas" in glob_cols:
        m_names.append("m_hy_oas_z")
    if {"em", "acwi"} <= set(glob_cols):
        m_names.append("m_em_vs_dm_63d")
    p1_names = (["p1_med_21d", "p1_med_63d", "p1_iqr_21d", "p1_n"]
                if have_peers else [])
    momentum_names = f1_names + f2_names + f3_names + m_names + p1_names

    zstate: dict[tuple[str, str], _Expanding] = {}

    def expz(market: str, name: str, x: float) -> float:
        key = (market, name)
        if key not in zstate:
            zstate[key] = _Expanding()
        return zstate[key].z(x)

    regime_state: dict[str, _ExpandingTerciles] = {}
    W = cfg.gpr_window
    win90 = np.timedelta64(cfg.momentum_window_days, "D")
    sup90 = np.timedelta64(cfg.supply_window_days, "D")
    sup365 = np.timedelta64(cfg.supply_trailing_days, "D")

    # Factors are computed for EVERY ipo row in chronological order (the
    # expanding states must see all past events, including rows later dropped
    # for missing labels), then subset to labeled rows.
    momentum_all = np.zeros((len(ipos), len(momentum_names)), dtype=np.float32)
    for i in range(len(ipos)):
        t0 = all_dates[i]
        m_i = all_market[i]
        vals: dict[str, float] = {}

        # --- f1/f2: up to K most recent same-market IPOs within the window ---
        lo = np.searchsorted(all_dates, t0 - win90, side="left")
        hi = np.searchsorted(all_dates, t0, side="left")
        cand = [j for j in range(hi - 1, lo - 1, -1) if all_market[j] == m_i]
        cand = cand[: cfg.momentum_max_deals]
        mom_cum = cum_raw_excess if cfg.momentum_market_adjust else cum_raw
        for h in mh:
            t = tag.get(h, f"{h}d")
            rets, weights = [], []
            for j in cand:
                dj = ipo_dates.get(all_ids[j])
                if dj is None:
                    continue
                n_obs = int(np.searchsorted(dj, t0, side="left"))
                if n_obs >= h:  # h-th close is strictly before t0 -> observable
                    rets.append(mom_cum[all_ids[j]][h - 1])
                    weights.append(all_size[j])
            r = np.asarray(rets)
            w = np.asarray(weights)
            if len(r):
                vals[f"f1_med_{t}"] = float(np.median(r))
                vals[f"f1_vw_{t}"] = float((r * w).sum() / w.sum()) if w.sum() > 0 \
                    else float(r.mean())
                vals[f"f2_break_{t}"] = float((r < 0).mean())
                down = r[r < 0]
                vals[f"f2_depth_{t}"] = float(down.mean()) if len(down) else 0.0
                if cfg.momentum_extended:
                    vals[f"f1_n_{t}"] = float(len(r))
                    vals[f"f1_iqr_{t}"] = float(np.percentile(r, 75) - np.percentile(r, 25))
            else:
                for k in (f"f1_med_{t}", f"f1_vw_{t}", f"f2_break_{t}", f"f2_depth_{t}"):
                    vals[k] = 0.0
                if cfg.momentum_extended:
                    vals[f"f1_n_{t}"] = 0.0
                    vals[f"f1_iqr_{t}"] = 0.0

        # --- f3: rolling supply in this market (expanding z-scored) ---
        s3 = deals_between(t0 - sup90, t0)
        in_mkt = deal_market[s3] == m_i
        cnt_3m = float(in_mkt.sum())
        prc_3m = float(np.log1p(deal_proceeds[s3][in_mkt].sum()))
        s12 = deals_between(t0 - sup365, t0)
        cnt_12m = float((deal_market[s12] == m_i).sum())
        pace = cnt_12m * (cfg.supply_window_days / cfg.supply_trailing_days)
        accel = cnt_3m / max(pace, 1.0)
        ipo_cnt_3m = float(sum(1 for j in range(lo, hi) if all_market[j] == m_i))
        vals["f3_cnt_3m"] = expz(m_i, "f3_cnt_3m", cnt_3m)
        vals["f3_prc_3m"] = expz(m_i, "f3_prc_3m", prc_3m)
        vals["f3_ipo_cnt_3m"] = expz(m_i, "f3_ipo_cnt_3m", ipo_cnt_3m)
        vals["f3_accel"] = expz(m_i, "f3_accel", accel)

        # --- m: macro block ---
        b_dates, b_log = get_bench(m_i)
        bi = _asof_idx(b_dates, t0)
        vals["m_geo_mom_63d"] = _series_ret(b_log, bi, 63)
        if bi >= 0:
            lo252 = max(0, bi - 251)
            vals["m_geo_dd_252"] = float(b_log[bi] - b_log[lo252: bi + 1].max())
            dr = np.diff(b_log[max(0, bi - 21): bi + 1])
            vals["m_geo_rvol_21d"] = (float(dr.std() * np.sqrt(252))
                                      if len(dr) >= 10 else 0.0)
        else:
            vals["m_geo_dd_252"] = 0.0
            vals["m_geo_rvol_21d"] = 0.0

        # regime: mean pop of the prior `regime_pop_window` same-market IPOs
        pops = []
        for j in range(hi - 1, -1, -1):
            if all_market[j] != m_i:
                continue
            dj = ipo_dates.get(all_ids[j])
            if dj is None or np.searchsorted(dj, t0, side="left") < 1:
                continue
            pops.append(cum_raw[all_ids[j]][0])
            if len(pops) >= cfg.regime_pop_window:
                break
        if len(pops) >= 3:
            state = regime_state.setdefault(m_i, _ExpandingTerciles())
            regime = state.classify(float(np.mean(pops)))
        else:
            regime = 0
        vals["m_regime_hot"] = float(regime == 1)
        vals["m_regime_cold"] = float(regime == -1)

        if m_i in macro_mkt:
            mm = macro_mkt[m_i]
            mi = _asof_idx(mm["date"], t0)
            if "vol_index" in macro_cols:
                vals["m_vol_index_z"] = _rolling_z(mm["vol_index"], mi)
            if "fx" in macro_cols:
                vals["m_fx_ret_21d"] = (_series_ret(np.log(mm["fx"]), mi, 21)
                                        if mi >= 0 else 0.0)
        else:
            if "vol_index" in macro_cols:
                vals["m_vol_index_z"] = 0.0
            if "fx" in macro_cols:
                vals["m_fx_ret_21d"] = 0.0

        if glob_cols:
            gi2 = _asof_idx(glob["date"], t0)
            if "vix" in glob_cols:
                vals["m_vix_z"] = _rolling_z(glob["vix"], gi2)
                vals["m_vix_chg_21d"] = (float(glob["vix"][gi2] / glob["vix"][gi2 - 21] - 1.0)
                                         if gi2 >= 21 else 0.0)
            if "hy_oas" in glob_cols:
                vals["m_hy_oas_z"] = _rolling_z(glob["hy_oas"], gi2)
            if {"em", "acwi"} <= set(glob_cols):
                vals["m_em_vs_dm_63d"] = (_series_ret(np.log(glob["em"]), gi2, 63)
                                          - _series_ret(np.log(glob["acwi"]), gi2, 63))

        # --- p1: industry-subgroup peers (precomputed as-of by the puller) ---
        if have_peers:
            st = peer_stats.get(str(all_ids[i]),
                                {"p1_med_21d": 0.0, "p1_med_63d": 0.0,
                                 "p1_iqr_21d": 0.0, "p1_n": 0.0})
            vals.update(st)

        momentum_all[i] = [vals[n] for n in momentum_names]

    # ---- Assemble labeled rows ----
    rows, label_end = [], []
    ys: dict[int, list[float]] = {h: [] for h in horizons}
    gpr_seqs, gpr_feats = [], []
    for i in range(len(ipos)):
        if not is_target[i]:
            continue  # momentum-universe only: feeds factors, never modelled
        d = ipo_dates.get(all_ids[i])
        if d is None or len(d) < main_h:
            if not inference:
                continue  # end-of-sample rows without a full label window
            # inference: keep the row; fill whatever labels exist, NaN the rest
            cum = cum_target.get(all_ids[i], np.array([]))
            for h in horizons:
                ys[h].append(float(cum[h - 1]) if len(cum) >= h else float("nan"))
            label_end.append(d[main_h - 1] if d is not None and len(d) >= main_h
                             else all_dates[i])
            rows.append(i)
        else:
            cum = cum_target[all_ids[i]]
            for h in horizons:
                ys[h].append(cum[h - 1])
            label_end.append(d[main_h - 1])
            rows.append(i)

        pos = np.searchsorted(gpr_dates, all_dates[i], side="left")
        if pos == 0:
            raise ValueError(f"GPR series starts after first IPO date {all_dates[i]}.")
        window = gpr_vals[max(0, pos - W): pos]
        if len(window) < W:  # early-sample edge: pad with earliest value
            window = np.concatenate([np.full(W - len(window), window[0]), window])
        gpr_seqs.append(window.astype(np.float32))
        d5 = window[-1] - window[-6] if W >= 6 else 0.0
        # Slow risk-regime signal: 63d SD of daily changes, annualized.
        # Lives in the GPR arm (not the macro block) so use_gpr removes all
        # GPR information and FiLM can condition on risk volatility.
        g63 = gpr_vals[max(0, pos - 63): pos]
        vol63 = (float(np.diff(g63).std() * np.sqrt(252))
                 if len(g63) >= 63 else 0.0)
        gpr_feats.append(np.array([
            window[-1], d5, window[-1] - window[0], window.mean(), window.std(),
            vol63,
        ], dtype=np.float32))

    rows = np.asarray(rows)
    return FeatureSet(
        ids=all_ids[rows],
        dates=all_dates[rows],
        label_end=np.asarray(label_end),
        y={h: np.asarray(ys[h], dtype=np.float64) for h in horizons},
        sector=sector_all[rows],
        sector_names=sector_names,
        market_onehot=market_onehot_all[rows],
        market_names=market_names,
        bk=bk_all[rows],
        bk_names=bk_names,
        n_bk=bk_all[rows].sum(axis=1),
        static_extra=static_extra_all[rows],
        static_extra_names=extra_names,
        static_extra_continuous=extra_cont,
        momentum=momentum_all[rows],
        momentum_names=momentum_names,
        gpr_seq=np.stack(gpr_seqs),
        gpr_feats=np.stack(gpr_feats),
    )
