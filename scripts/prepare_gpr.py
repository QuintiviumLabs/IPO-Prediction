#!/usr/bin/env python3
"""Convert the Caldara-Iacoviello daily GPR file into the pipeline's gpr.csv.

Download the daily series from https://www.matteoiacoviello.com/gpr.htm
(data_gpr_daily_recent — export/convert to .csv or .xlsx if it is .xls),
then run:

    python scripts/prepare_gpr.py --in data_gpr_daily_recent.csv --out data/gpr.csv

Accepts a date column named DAY/day/date/Date (yyyymmdd integers or parseable
dates) and a level column named GPRD/gpr/GPR. Rows with missing values are
dropped; output columns are (date, gpr).
"""
import argparse
from pathlib import Path

import pandas as pd

DATE_COLS = ["DAY", "day", "date", "Date", "DATE"]
GPR_COLS = ["GPRD", "gprd", "GPR", "gpr"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", required=True, help="downloaded GPR daily file")
    p.add_argument("--out", default="data/gpr.csv")
    args = p.parse_args()

    src = Path(args.src)
    if src.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(src)  # .xls may need: pip install xlrd
    else:
        df = pd.read_csv(src)

    date_col = next((c for c in DATE_COLS if c in df.columns), None)
    gpr_col = next((c for c in GPR_COLS if c in df.columns), None)
    if date_col is None or gpr_col is None:
        raise SystemExit(f"Could not find date/GPR columns; got: {list(df.columns)}")

    dates = df[date_col]
    if pd.api.types.is_numeric_dtype(dates):  # yyyymmdd integers
        dates = pd.to_datetime(dates.astype("Int64").astype(str), format="%Y%m%d")
    else:
        dates = pd.to_datetime(dates)

    out = pd.DataFrame({"date": dates, "gpr": pd.to_numeric(df[gpr_col], errors="coerce")})
    out = out.dropna().sort_values("date")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows ({out['date'].iloc[0].date()} … "
          f"{out['date'].iloc[-1].date()}) to {args.out}")


if __name__ == "__main__":
    main()
