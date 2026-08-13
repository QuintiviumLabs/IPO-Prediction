#!/usr/bin/env python3
"""Pull prices.csv and market.csv from a running Bloomberg Terminal.

Requires the Terminal running and logged in on this machine, plus:
    pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
    pip install xbbg

Reads ipos.csv (needs `bbg_ticker`, or `ipo_id` used as the ticker) and pulls
PX_LAST for each deal's first `--days` calendar days, then pulls one
benchmark index per market.

    python scripts/pull_bloomberg.py --data data ^
        --benchmarks "HK=HSI Index,US=SPX Index" --days 45

Safe to re-run: already-fetched tickers are skipped, so an interrupted pull
resumes where it stopped. Failures are written to bloomberg_failures.csv
rather than aborting the run.

NOTE: written against the xbbg API but NOT executable without a Terminal —
run it on a handful of deals first (--limit 5) and eyeball the output before
launching the full pull.
"""
import argparse
from pathlib import Path

import pandas as pd


def _parse_benchmarks(spec: str) -> dict[str, str]:
    out = {}
    for pair in filter(None, (p.strip() for p in spec.split(","))):
        if "=" not in pair:
            raise SystemExit(f"--benchmarks entry '{pair}' must look like MARKET=TICKER")
        market, ticker = pair.split("=", 1)
        out[market.strip()] = ticker.strip()
    return out


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """xbbg returns MultiIndex (ticker, field) columns; reduce to date/close."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "close"])
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df[(ticker, "PX_LAST")]
        except KeyError:
            s = df.iloc[:, 0]
    else:
        s = df.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(s.index), "close": s.to_numpy()})
    return out.dropna()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data", help="folder holding ipos.csv")
    p.add_argument("--benchmarks", required=True,
                   help='e.g. "HK=HSI Index,US=SPX Index" — one index per market')
    p.add_argument("--days", type=int, default=45,
                   help="calendar days of prices per IPO (45 ~ 30 trading days)")
    p.add_argument("--limit", type=int, default=None,
                   help="only pull this many IPOs (use for a smoke test first)")
    args = p.parse_args()

    from xbbg import blp  # imported here so --help works without Bloomberg

    d = Path(args.data)
    ipos = pd.read_csv(d / "ipos.csv", parse_dates=["first_trade_date"])
    if "bbg_ticker" not in ipos.columns:
        print("no bbg_ticker column — using ipo_id as the Bloomberg ticker")
        ipos["bbg_ticker"] = ipos["ipo_id"]
    benchmarks = _parse_benchmarks(args.benchmarks)
    missing = set(ipos["market"].astype(str)) - set(benchmarks)
    if missing:
        raise SystemExit(f"no --benchmarks entry for markets: {sorted(missing)}")

    # ---- prices.csv (resumable) ----
    prices_path = d / "prices.csv"
    done: set[str] = set()
    if prices_path.exists():
        done = set(pd.read_csv(prices_path, usecols=["ipo_id"])["ipo_id"].unique())
        print(f"resuming: {len(done)} IPOs already in {prices_path.name}")

    todo = ipos[~ipos["ipo_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"pulling prices for {len(todo)} IPOs ...")

    failures, wrote_header = [], prices_path.exists()
    for n, (_, row) in enumerate(todo.iterrows(), start=1):
        start = row["first_trade_date"]
        end = start + pd.Timedelta(days=args.days)
        try:
            raw = blp.bdh(tickers=row["bbg_ticker"], flds="PX_LAST",
                          start_date=start.strftime("%Y-%m-%d"),
                          end_date=end.strftime("%Y-%m-%d"))
            px = _flatten(raw, row["bbg_ticker"])
            if px.empty:
                raise ValueError("no data returned")
            px.insert(0, "ipo_id", row["ipo_id"])
            px.to_csv(prices_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        except Exception as e:  # keep going; one bad ticker must not kill the run
            failures.append({"ipo_id": row["ipo_id"],
                             "bbg_ticker": row["bbg_ticker"], "error": str(e)})
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} ({len(failures)} failed)")

    if failures:
        fp = d / "bloomberg_failures.csv"
        pd.DataFrame(failures).to_csv(fp, index=False)
        print(f"{len(failures)} tickers failed — see {fp}")

    # ---- market.csv (one benchmark series per market) ----
    lo = ipos["first_trade_date"].min() - pd.Timedelta(days=400)
    hi = ipos["first_trade_date"].max() + pd.Timedelta(days=args.days + 10)
    print(f"pulling {len(benchmarks)} benchmark indices "
          f"({lo.date()} … {hi.date()}) ...")
    frames = []
    for market, ticker in benchmarks.items():
        raw = blp.bdh(tickers=ticker, flds="PX_LAST",
                      start_date=lo.strftime("%Y-%m-%d"),
                      end_date=hi.strftime("%Y-%m-%d"))
        bm = _flatten(raw, ticker)
        if bm.empty:
            raise SystemExit(f"benchmark {ticker} returned no data")
        bm.insert(1, "market", market)
        frames.append(bm)
        print(f"  {market}: {ticker} — {len(bm)} rows")
    pd.concat(frames, ignore_index=True).to_csv(d / "market.csv", index=False)

    print(f"\nDone. Next: python scripts\\check_data.py --data {args.data}")


if __name__ == "__main__":
    main()
