#!/usr/bin/env python3
"""Validate a data directory against the pipeline's schema BEFORE running
anything. Prints every blocking error and warning with the fix, so assembling
the CSVs (from Bloomberg exports etc.) is a check-fix loop, not a stack trace.

    python scripts/check_data.py --data data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_IPO_COLS = ["ipo_id", "first_trade_date", "offer_price", "market",
                     "deal_size", "is_tmt", "is_healthcare"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    args = p.parse_args()
    d = Path(args.data)
    errors: list[str] = []
    warns: list[str] = []
    infos: list[str] = []

    def need(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            errors.append(f"missing file: {path}")
            return None
        try:
            return pd.read_csv(path)
        except Exception as e:
            errors.append(f"{path.name}: cannot read ({e})")
            return None

    ipos, prices = need(d / "ipos.csv"), need(d / "prices.csv")
    gpr, market = need(d / "gpr.csv"), need(d / "market.csv")
    deals = pd.read_csv(d / "deals.csv") if (d / "deals.csv").exists() else None

    # ---- ipos.csv ----
    if ipos is not None:
        for c in REQUIRED_IPO_COLS:
            if c not in ipos.columns:
                errors.append(f"ipos.csv: missing required column '{c}'")
        bk = [c for c in ipos.columns if c.startswith("bk_")]
        if not bk:
            warns.append("ipos.csv: no bk_* bookrunner columns found — the "
                         "static arm will run without bookrunner identity")
        if "ipo_id" in ipos.columns and ipos["ipo_id"].duplicated().any():
            errors.append("ipos.csv: duplicate ipo_id values")
        if "first_trade_date" in ipos.columns:
            dates = pd.to_datetime(ipos["first_trade_date"], errors="coerce")
            if dates.isna().any():
                errors.append(f"ipos.csv: {int(dates.isna().sum())} unparseable "
                              "first_trade_date values (use YYYY-MM-DD)")
        for c, positive in (("offer_price", True), ("deal_size", True)):
            if c in ipos.columns:
                v = pd.to_numeric(ipos[c], errors="coerce")
                if v.isna().any():
                    errors.append(f"ipos.csv: {int(v.isna().sum())} non-numeric {c}")
                elif positive and (v <= 0).any():
                    warns.append(f"ipos.csv: {int((v <= 0).sum())} rows with {c} <= 0")
        for c in ("is_tmt", "is_healthcare"):
            if c in ipos.columns and not ipos[c].isin([0, 1]).all():
                errors.append(f"ipos.csv: {c} must be 0/1")
        if "market" in ipos.columns:
            counts = ipos["market"].value_counts()
            small = counts[counts < 30]
            if len(small):
                warns.append("ipos.csv: markets with < 30 IPOs (momentum factors "
                             f"will be thin there): {dict(small)}")
        if "is_target" in ipos.columns:
            n_t = int((ipos["is_target"].fillna(1) != 0).sum())
            n_u = len(ipos) - n_t
            infos.append(f"{n_t} modelled IPOs + {n_u} momentum-universe rows "
                         "(is_target=0: feed F1-F4, never trained on)")
            if n_t == 0:
                errors.append("ipos.csv: every row has is_target=0 — nothing to model")
        else:
            warns.append("ipos.csv: no is_target column — every row is both a "
                         "modelling target and part of the F1/F2 recent-deal "
                         "pool. Add is_target=0 rows to widen the momentum "
                         "universe without training on them.")

    # ---- prices.csv ----
    if prices is not None and ipos is not None:
        for c in ("ipo_id", "date", "close"):
            if c not in prices.columns:
                errors.append(f"prices.csv: missing column '{c}'")
        if not errors:
            v = pd.to_numeric(prices["close"], errors="coerce")
            if v.isna().any() or (v <= 0).any():
                errors.append("prices.csv: close must be numeric and > 0")
            n_closes = prices.groupby("ipo_id").size()
            have = set(n_closes.index)
            missing = set(ipos["ipo_id"]) - have
            if missing:
                warns.append(f"prices.csv: {len(missing)} IPOs have no price rows "
                             "(they will be dropped)")
            short = int((n_closes.reindex(ipos["ipo_id"]).fillna(0) < 22).sum())
            if short:
                warns.append(f"{short} IPOs have < 22 closes — they lose the "
                             "1-month label and are dropped from training "
                             "(still used inside momentum factors)")

    # ---- market.csv (per-market benchmarks) ----
    if market is not None:
        if "date" not in market.columns or "close" not in market.columns:
            errors.append("market.csv: needs columns (date, [market,] close)")
        else:
            dts = pd.to_datetime(market["date"], errors="coerce")
            if dts.isna().any():
                errors.append("market.csv: unparseable dates")
            elif "market" in market.columns and ipos is not None \
                    and {"market", "first_trade_date"} <= set(ipos.columns):
                missing_b = set(ipos["market"].astype(str)) \
                    - set(market["market"].astype(str))
                if missing_b:
                    errors.append("market.csv: no benchmark series for markets "
                                  f"{sorted(missing_b)} (one index per market, "
                                  "long format: date, market, close)")
                for m, g in market.groupby("market"):
                    gd = pd.to_datetime(g["date"]).sort_values()
                    gap = gd.diff().dt.days.dropna()
                    if len(gap) and gap.median() > 4:
                        warns.append(f"market.csv[{m}]: median gap "
                                     f"{gap.median():.0f} days — expects DAILY")
                    in_m = ipos[ipos["market"].astype(str) == str(m)]
                    if len(in_m):
                        first = pd.to_datetime(in_m["first_trade_date"],
                                               errors="coerce").min()
                        if pd.notna(first) and gd.min() >= first:
                            errors.append(f"market.csv[{m}]: starts "
                                          f"{gd.min().date()}, on/after that "
                                          f"market's first IPO ({first.date()})")
            else:
                warns.append("market.csv: no 'market' column — ONE global "
                             "benchmark will be used for every market "
                             "(per-market indices recommended: date, market, close)")
                if ipos is not None and "first_trade_date" in ipos.columns:
                    first_ipo = pd.to_datetime(ipos["first_trade_date"],
                                               errors="coerce").min()
                    if pd.notna(first_ipo) and dts.min() >= first_ipo:
                        errors.append(f"market.csv: series starts "
                                      f"{dts.min().date()}, on/after the first "
                                      f"IPO ({first_ipo.date()})")

    # ---- gpr.csv ----
    if gpr is not None:
        if "date" not in gpr.columns or "gpr" not in gpr.columns:
            errors.append("gpr.csv: needs columns (date, gpr)")
        else:
            dts = pd.to_datetime(gpr["date"], errors="coerce")
            if dts.isna().any():
                errors.append("gpr.csv: unparseable dates")
            else:
                gap = dts.sort_values().diff().dt.days.dropna()
                if len(gap) and gap.median() > 4:
                    warns.append(f"gpr.csv: median gap {gap.median():.0f} days "
                                 "— the pipeline expects a DAILY series")
                if ipos is not None and "first_trade_date" in ipos.columns:
                    first_ipo = pd.to_datetime(ipos["first_trade_date"],
                                               errors="coerce").min()
                    if pd.notna(first_ipo) and dts.min() >= first_ipo:
                        errors.append(f"gpr.csv: series starts {dts.min().date()}, "
                                      f"on/after the first IPO ({first_ipo.date()})")

    # ---- deals.csv (optional) ----
    if deals is not None:
        for c in ("date", "market", "proceeds"):
            if c not in deals.columns:
                errors.append(f"deals.csv: missing column '{c}'")
    else:
        warns.append("no deals.csv — F3 supply factors fall back to IPO-only")

    # ---- syndicate extras in ipos.csv (optional) ----
    if ipos is not None:
        synd = [c for c in ("n_banks", "has_bb", "prestige_rank_max")
                if c in ipos.columns]
        arch = [c for c in ipos.columns if c.startswith("archetype_")]
        if synd or arch:
            infos.append(f"syndicate block present: {synd + arch[:3]}"
                         + (" ..." if len(arch) > 3 else ""))
        else:
            infos.append("no syndicate columns (n_banks/has_*/archetype_*) — "
                         "run scripts/build_bookrunner_features.py to add them")
        if "subgroup" in ipos.columns:
            n_sg = int(ipos["subgroup"].notna().sum())
            infos.append(f"subgroup present on {n_sg}/{len(ipos)} rows")
        else:
            infos.append("no subgroup column — run scripts/pull_subgroups.py "
                         "(needed for the peer block)")

    # ---- macro.csv / macro_global.csv (optional) ----
    if (d / "macro.csv").exists():
        mc = pd.read_csv(d / "macro.csv")
        if not {"date", "market"} <= set(mc.columns):
            errors.append("macro.csv: needs columns (date, market, ...)")
        else:
            got = [c for c in ("vol_index", "fx") if c in mc.columns]
            infos.append(f"macro.csv present with {got} for markets "
                         f"{sorted(mc['market'].astype(str).unique())}")
    else:
        infos.append("no macro.csv — local vol/FX macro features skipped "
                     "(scripts/pull_macro.py)")
    if (d / "macro_global.csv").exists():
        mg = pd.read_csv(d / "macro_global.csv")
        if "date" not in mg.columns:
            errors.append("macro_global.csv: needs a date column")
        else:
            got = [c for c in ("vix", "hy_oas", "em", "acwi") if c in mg.columns]
            infos.append(f"macro_global.csv present with {got}")
    else:
        infos.append("no macro_global.csv — global risk features skipped "
                     "(scripts/pull_macro.py)")

    # ---- peers.csv (optional) ----
    if (d / "peers.csv").exists() and ipos is not None:
        pr = pd.read_csv(d / "peers.csv")
        for c in ("ipo_id", "ret_21d"):
            if c not in pr.columns:
                errors.append(f"peers.csv: missing column '{c}'")
        if "asof" in pr.columns and "ipo_id" in pr.columns \
                and {"ipo_id", "first_trade_date"} <= set(ipos.columns):
            asof = pd.to_datetime(pr["asof"], errors="coerce")
            ft = pr["ipo_id"].map(dict(zip(
                ipos["ipo_id"],
                pd.to_datetime(ipos["first_trade_date"], errors="coerce"))))
            late = int((asof >= ft).sum())
            if late:
                errors.append(f"peers.csv: {late} rows have asof >= the IPO's "
                              "first_trade_date — peer features must be "
                              "strictly pre-pricing (re-run pull_peers.py)")
        elif "asof" not in pr.columns:
            errors.append("peers.csv: no asof column — the pipeline cannot "
                          "verify the peer returns are pre-pricing and will "
                          "refuse the file (re-run scripts/pull_peers.py)")
        if "ipo_id" in pr.columns:
            cov = pr["ipo_id"].nunique()
            infos.append(f"peers.csv covers {cov}/{len(ipos)} IPOs "
                         f"({len(pr)} peer rows)")
    else:
        infos.append("no peers.csv — subgroup peer (p1) features skipped "
                     "(scripts/pull_peers.py)")

    print(f"\n{'='*60}\nData check: {d}/")
    for i in infos:
        print(f"  info     {i}")
    for e in errors:
        print(f"  ERROR    {e}")
    for w in warns:
        print(f"  warning  {w}")
    if not errors:
        n = len(ipos) if ipos is not None else 0
        print(f"  OK       {n} IPOs — ready to run" if not warns
              else f"  OK       {n} IPOs — runnable, see warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
