#!/usr/bin/env python3
"""Download the Caldara-Iacoviello daily GPR index and write data/gpr.csv.

    python scripts/fetch_gpr.py --out data/gpr.csv

Pulls the daily file from matteoiacoviello.com (override with --url if the
link has moved — find the current "Daily GPR" link on
https://www.matteoiacoviello.com/gpr.htm). If the download fails (firewall,
moved link), download the file in a browser instead and convert it with:

    python scripts/prepare_gpr.py --in <downloaded file> --out data/gpr.csv
"""
import argparse
import io
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
DATE_COLS = ["DAY", "day", "date", "Date", "DATE"]
GPR_COLS = ["GPRD", "gprd", "GPR", "gpr"]


def parse_gpr(df: pd.DataFrame) -> pd.DataFrame:
    date_col = next((c for c in DATE_COLS if c in df.columns), None)
    gpr_col = next((c for c in GPR_COLS if c in df.columns), None)
    if date_col is None or gpr_col is None:
        raise SystemExit(f"Could not find date/GPR columns; got: {list(df.columns)}")
    dates = df[date_col]
    if pd.api.types.is_numeric_dtype(dates):  # yyyymmdd integers
        dates = pd.to_datetime(dates.astype("Int64").astype(str), format="%Y%m%d")
    else:
        dates = pd.to_datetime(dates)
    out = pd.DataFrame({"date": dates,
                        "gpr": pd.to_numeric(df[gpr_col], errors="coerce")})
    return out.dropna().sort_values("date")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="data/gpr.csv")
    args = p.parse_args()

    print(f"Downloading {args.url} ...")
    req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60,
                                    context=ssl.create_default_context()) as r:
            blob = r.read()
    except Exception as e:
        print(f"\nDownload failed: {e}\n"
              "Fallback: open https://www.matteoiacoviello.com/gpr.htm in a "
              "browser, download the DAILY GPR data file, then run:\n"
              "  python scripts/prepare_gpr.py --in <downloaded file> "
              f"--out {args.out}", file=sys.stderr)
        raise SystemExit(1)

    # The daily file is legacy .xls (needs xlrd); tolerate .xlsx and .csv too.
    df = None
    for reader in (
        lambda b: pd.read_excel(io.BytesIO(b)),
        lambda b: pd.read_csv(io.BytesIO(b)),
    ):
        try:
            df = reader(blob)
            break
        except Exception:
            continue
    if df is None:
        raise SystemExit("Downloaded file could not be parsed as Excel or CSV. "
                         "Save it manually and use scripts/prepare_gpr.py.")

    out = parse_gpr(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows ({out['date'].iloc[0].date()} … "
          f"{out['date'].iloc[-1].date()}) to {args.out}")


if __name__ == "__main__":
    main()
