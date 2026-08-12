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

    # ---- market.csv / gpr.csv ----
    for name, df, col in (("market.csv", market, "close"), ("gpr.csv", gpr, "gpr")):
        if df is None:
            continue
        if "date" not in df.columns or col not in df.columns:
            errors.append(f"{name}: needs columns (date, {col})")
            continue
        dts = pd.to_datetime(df["date"], errors="coerce")
        if dts.isna().any():
            errors.append(f"{name}: unparseable dates")
            continue
        gap = dts.sort_values().diff().dt.days.dropna()
        if len(gap) and gap.median() > 4:
            warns.append(f"{name}: median gap {gap.median():.0f} days — the "
                         "pipeline expects a DAILY series")
        if ipos is not None and "first_trade_date" in ipos.columns:
            first_ipo = pd.to_datetime(ipos["first_trade_date"], errors="coerce").min()
            if pd.notna(first_ipo) and dts.min() >= first_ipo:
                errors.append(f"{name}: series starts {dts.min().date()}, on/after "
                              f"the first IPO ({first_ipo.date()}) — need history "
                              "before it")

    # ---- deals.csv (optional) ----
    if deals is not None:
        for c in ("date", "market", "sector", "proceeds"):
            if c not in deals.columns:
                errors.append(f"deals.csv: missing column '{c}'")
        if "sector" in deals.columns:
            bad = set(deals["sector"].astype(str).unique()) - {"tmt", "healthcare", "other"}
            if bad:
                warns.append(f"deals.csv: unrecognized sector labels {sorted(bad)} "
                             "(expected tmt/healthcare/other) — they will never "
                             "match a target's sector")
    else:
        warns.append("no deals.csv — F3/F4 supply factors fall back to IPO-only")

    print(f"\n{'='*60}\nData check: {d}/")
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
