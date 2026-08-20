#!/usr/bin/env python3
"""Pull each company's Bloomberg industry subgroup into your deal file.

Adds/fills a `subgroup` column in ipos.csv (or an .xlsx export — pass the
file directly) via BDP INDUSTRY_SUBGROUP. The subgroup is what pull_peers.py
uses to find comparable companies.

    python scripts/pull_subgroups.py --file data/ipos.csv
    python scripts/pull_subgroups.py --file "C:\\deals\\ipo master.xlsx" --sheet Deals

Requires a running, logged-in Terminal (pip install xbbg blpapi; .xlsx also
needs openpyxl). Safe to re-run: rows that already have a subgroup are
skipped, so an interrupted pull resumes. Uses `bbg_ticker` if present, else
`ipo_id`, as the security identifier.

Also writes a BICS fallback: if INDUSTRY_SUBGROUP returns nothing for a
name, GICS_SUB_INDUSTRY_NAME is tried before giving up.
"""
import argparse
from pathlib import Path

import pandas as pd

FIELDS = ["INDUSTRY_SUBGROUP", "GICS_SUB_INDUSTRY_NAME"]
BATCH = 50


def _read(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path, sheet: str | None) -> None:
    if path.suffix.lower() in (".xlsx", ".xls"):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as xl:
            df.to_excel(xl, sheet_name=sheet or "Sheet1", index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, help="ipos.csv or an .xlsx export")
    p.add_argument("--sheet", default=None, help="sheet name for .xlsx files")
    p.add_argument("--limit", type=int, default=None,
                   help="only pull this many names (smoke test first)")
    args = p.parse_args()

    from xbbg import blp  # imported here so --help works without Bloomberg

    path = Path(args.file)
    df = _read(path, args.sheet)
    tick_col = "bbg_ticker" if "bbg_ticker" in df.columns else "ipo_id"
    if tick_col == "ipo_id":
        print("no bbg_ticker column — using ipo_id as the Bloomberg ticker")
    if "subgroup" not in df.columns:
        df["subgroup"] = pd.NA

    todo = df.index[df["subgroup"].isna() | (df["subgroup"].astype(str) == "")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"pulling subgroup for {len(todo)} of {len(df)} rows ...")

    n_filled = 0
    for start in range(0, len(todo), BATCH):
        idx = todo[start: start + BATCH]
        tickers = df.loc[idx, tick_col].astype(str).tolist()
        try:
            ref = blp.bdp(tickers=tickers, flds=FIELDS)
        except Exception as e:
            print(f"  batch failed ({e}); continuing")
            continue
        # xbbg lower-cases field names in the result
        cols = {c.lower(): c for c in ref.columns}
        for i, t in zip(idx, tickers):
            if t not in ref.index:
                continue
            for f in FIELDS:
                v = ref.loc[t].get(cols.get(f.lower(), f), None)
                if isinstance(v, str) and v.strip():
                    df.loc[i, "subgroup"] = v.strip()
                    n_filled += 1
                    break
        print(f"  {min(start + BATCH, len(todo))}/{len(todo)} ({n_filled} filled)")
        _write(df, path, args.sheet)   # checkpoint after every batch

    still = int((df["subgroup"].isna() | (df["subgroup"].astype(str) == "")).sum())
    print(f"\nDone: {n_filled} filled this run, {still} still empty -> {path}")
    if still:
        print("Empty rows are usually delisted/renamed tickers — fill by hand "
              "or map them to their new ticker in a bbg_ticker column.")


if __name__ == "__main__":
    main()
