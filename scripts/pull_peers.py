#!/usr/bin/env python3
"""Build peers.csv: as-of trailing returns of industry-subgroup peers.

For each target IPO, finds up to --k listed companies in the SAME Bloomberg
industry subgroup from your peer universe file, pulls their prices for the
window ending strictly BEFORE the IPO's first trade date, and writes their
trailing 21d/63d returns. The pipeline aggregates these into the p1 factor
block (p1_med_21d, p1_med_63d, p1_iqr_21d, p1_n).

YOU MUST SUPPLY data/peer_universe.csv first — the Bloomberg API cannot
screen "all members of subgroup X"; export it once from the Terminal (EQS
screen or BI <GO>) with columns:
    ticker, subgroup [, market] [, market_cap]
market restricts peers to the deal's own market when present; market_cap
picks the largest --k peers (else the first --k in file order).

    python scripts/pull_peers.py --data data --k 8

Requires ipos.csv to have a `subgroup` column (scripts/pull_subgroups.py).
Resumable: IPOs already in peers.csv are skipped. The asof column (last
price date used) is validated by the pipeline to be strictly pre-pricing.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _flatten_multi(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    """xbbg bdh with several tickers -> {ticker: close series indexed by date}."""
    out: dict[str, pd.Series] = {}
    if raw is None or len(raw) == 0:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "PX_LAST") in raw.columns:
                out[t] = raw[(t, "PX_LAST")].dropna()
    elif len(tickers) == 1:
        out[tickers[0]] = raw.iloc[:, 0].dropna()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--universe", default=None,
                   help="peer universe file (default <data>/peer_universe.csv)")
    p.add_argument("--k", type=int, default=8, help="max peers per IPO")
    p.add_argument("--window-days", type=int, default=140,
                   help="calendar days of peer prices to pull (>= 63 trading days)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process this many IPOs (smoke test first)")
    args = p.parse_args()

    from xbbg import blp  # imported here so --help works without Bloomberg

    d = Path(args.data)
    ipos = pd.read_csv(d / "ipos.csv", parse_dates=["first_trade_date"])
    if "subgroup" not in ipos.columns:
        raise SystemExit("ipos.csv has no subgroup column — run "
                         "scripts/pull_subgroups.py first")
    upath = Path(args.universe) if args.universe else d / "peer_universe.csv"
    if not upath.exists():
        raise SystemExit(
            f"{upath} not found. Export it from the Terminal (EQS/BI screen) "
            "with columns: ticker, subgroup [, market] [, market_cap] — the "
            "API cannot enumerate subgroup members.")
    uni = pd.read_csv(upath)
    for col in ("ticker", "subgroup"):
        if col not in uni.columns:
            raise SystemExit(f"{upath} is missing required column '{col}'")

    out_path = d / "peers.csv"
    done: set[str] = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["ipo_id"])["ipo_id"].unique())
        print(f"resuming: {len(done)} IPOs already in {out_path.name}")

    todo = ipos[~ipos["ipo_id"].isin(done)]
    todo = todo[todo["subgroup"].notna()]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"pulling peers for {len(todo)} IPOs (k={args.k}) ...")

    failures, wrote_header = [], out_path.exists()
    for n, (_, row) in enumerate(todo.iterrows(), start=1):
        sel = uni[uni["subgroup"] == row["subgroup"]]
        if "market" in uni.columns and "market" in row:
            same = sel[sel["market"].astype(str) == str(row["market"])]
            if len(same):
                sel = same
        # never let the IPO itself be its own peer
        self_ticks = {str(row.get("bbg_ticker", "")), str(row["ipo_id"])}
        sel = sel[~sel["ticker"].astype(str).isin(self_ticks)]
        if "market_cap" in sel.columns:
            sel = sel.sort_values("market_cap", ascending=False)
        peers = sel["ticker"].astype(str).head(args.k).tolist()
        if not peers:
            failures.append({"ipo_id": row["ipo_id"], "error":
                             f"no peers in universe for subgroup {row['subgroup']}"})
            continue

        end = row["first_trade_date"] - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=args.window_days)
        try:
            raw = blp.bdh(tickers=peers, flds="PX_LAST",
                          start_date=start.strftime("%Y-%m-%d"),
                          end_date=end.strftime("%Y-%m-%d"))
            series = _flatten_multi(raw, peers)
        except Exception as e:
            failures.append({"ipo_id": row["ipo_id"], "error": str(e)})
            continue

        rows = []
        for t, px in series.items():
            px = px[px.index < row["first_trade_date"]]   # belt and braces
            v = px.to_numpy(float)
            if len(v) < 22:
                continue
            r21 = float(np.log(v[-1] / v[-22]))
            r63 = float(np.log(v[-1] / v[-64])) if len(v) >= 64 else np.nan
            rows.append({"ipo_id": row["ipo_id"], "peer": t,
                         "ret_21d": round(r21, 6),
                         "ret_63d": round(r63, 6) if np.isfinite(r63) else "",
                         "asof": px.index[-1].strftime("%Y-%m-%d")})
        if not rows:
            failures.append({"ipo_id": row["ipo_id"],
                             "error": "no peer had >=22 closes before pricing"})
            continue
        pd.DataFrame(rows).to_csv(out_path, mode="a",
                                  header=not wrote_header, index=False)
        wrote_header = True
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} ({len(failures)} failed)")

    if failures:
        fp = d / "peer_failures.csv"
        pd.DataFrame(failures).to_csv(fp, index=False)
        print(f"{len(failures)} IPOs failed — see {fp}")
    print(f"\nDone -> {out_path}. Next: python scripts/check_data.py --data {args.data}")


if __name__ == "__main__":
    main()
