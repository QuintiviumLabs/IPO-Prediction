#!/usr/bin/env python3
"""Pull macro.csv (per-market vol index + FX) and macro_global.csv (global
risk series) from a running Bloomberg Terminal.

The macro (m_*) factor block degrades gracefully — features whose source
column is missing are simply not built — so pull whatever your markets have.

    python scripts/pull_macro.py --data data ^
        --vol "HK=VHSI Index,US=VIX Index,CN=VXFXI Index" ^
        --fx  "HK=USDHKD Curncy,CN=USDCNH Curncy,JP=USDJPY Curncy"

Defaults for the globals (override if your desk prefers others):
    vix    VIX Index      global equity vol
    hy_oas LF98OAS Index  US HY option-adjusted spread (credit stress)
    em     MXEF Index     MSCI Emerging Markets
    acwi   MXWD Index     MSCI ACWI (em vs dm relative momentum)

Local vol notes: HK=VHSI, Japan=VNKY, Europe=V2X, Korea=VKOSPI; markets with
no vol index can be omitted (the z-scored VIX still covers them globally).
FX is quoted as the LOCAL currency per USD so a rising value = local
weakening; the factor uses the 21d log change, direction just needs to be
consistent across your sample.
"""
import argparse
from pathlib import Path

import pandas as pd

GLOBAL_DEFAULTS = {"vix": "VIX Index", "hy_oas": "LF98OAS Index",
                   "em": "MXEF Index", "acwi": "MXWD Index"}


def _parse_map(spec: str | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in (spec or "").split(","))):
        if "=" not in pair:
            raise SystemExit(f"--{what} entry '{pair}' must look like MARKET=TICKER")
        market, ticker = pair.split("=", 1)
        out[market.strip()] = ticker.strip()
    return out


def _flatten(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        if (ticker, "PX_LAST") in raw.columns:
            return raw[(ticker, "PX_LAST")].dropna()
        return raw.iloc[:, 0].dropna()
    return raw.iloc[:, 0].dropna()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--vol", default=None,
                   help='per-market vol index, e.g. "HK=VHSI Index,US=VIX Index"')
    p.add_argument("--fx", default=None,
                   help='per-market FX vs USD, e.g. "HK=USDHKD Curncy"')
    for name, tick in GLOBAL_DEFAULTS.items():
        p.add_argument(f"--{name.replace('_', '-')}", default=tick,
                       help=f"global series (default: {tick}; pass '' to skip)")
    args = p.parse_args()

    from xbbg import blp  # imported here so --help works without Bloomberg

    d = Path(args.data)
    ipos = pd.read_csv(d / "ipos.csv", parse_dates=["first_trade_date"])
    lo = ipos["first_trade_date"].min() - pd.Timedelta(days=500)  # z-score warmup
    hi = ipos["first_trade_date"].max() + pd.Timedelta(days=10)
    lo_s, hi_s = lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")

    vol_map = _parse_map(args.vol, "vol")
    fx_map = _parse_map(args.fx, "fx")
    markets = sorted(set(vol_map) | set(fx_map))

    # ---- macro.csv: long format (date, market, vol_index, fx) ----
    if markets:
        frames = []
        for m in markets:
            cols = {}
            for col, tick in (("vol_index", vol_map.get(m)),
                              ("fx", fx_map.get(m))):
                if not tick:
                    continue
                s = _flatten(blp.bdh(tickers=tick, flds="PX_LAST",
                                     start_date=lo_s, end_date=hi_s), tick)
                if s.empty:
                    print(f"  WARNING {m} {col}: {tick} returned no data")
                    continue
                cols[col] = s
                print(f"  {m} {col}: {tick} — {len(s)} rows")
            if cols:
                df = pd.DataFrame(cols)
                df.insert(0, "market", m)
                df.insert(0, "date", df.index)
                frames.append(df.reset_index(drop=True))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(d / "macro.csv", index=False)
            print(f"macro.csv written for markets: {markets}")
    else:
        print("no --vol/--fx maps given — skipping macro.csv")

    # ---- macro_global.csv: wide format (date, vix, hy_oas, em, acwi) ----
    glob = {}
    for name in GLOBAL_DEFAULTS:
        tick = getattr(args, name)
        if not tick:
            continue
        s = _flatten(blp.bdh(tickers=tick, flds="PX_LAST",
                             start_date=lo_s, end_date=hi_s), tick)
        if s.empty:
            print(f"  WARNING global {name}: {tick} returned no data")
            continue
        glob[name] = s
        print(f"  global {name}: {tick} — {len(s)} rows")
    if glob:
        df = pd.DataFrame(glob)
        df.insert(0, "date", df.index)
        df.reset_index(drop=True).to_csv(d / "macro_global.csv", index=False)
        print("macro_global.csv written")

    print(f"\nDone. Next: python scripts/check_data.py --data {args.data}")


if __name__ == "__main__":
    main()
