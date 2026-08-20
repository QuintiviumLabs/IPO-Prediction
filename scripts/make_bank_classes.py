#!/usr/bin/env python3
"""Generate configs/bank_classes.csv from YOUR ipos.csv bank columns.

Reads every bk_* column header, matches each bank against a built-in
taxonomy of ~90 well-known ECM banks, and writes a bank_classes.csv with
class (bb / global / regional / local) and home_region pre-filled for every
bank it recognizes. Unrecognized banks are written with EMPTY class/region
cells at the top of the file so you can fill them in by hand.

    python scripts/make_bank_classes.py --file data/ipos.csv
    python scripts/make_bank_classes.py --file data/ipos.csv --force   # overwrite

THIS IS A DRAFT GENERATOR, NOT AN ORACLE — review the output. Classes are a
judgment call (one desk's "global" is another's "regional") and home_region
must use the SAME codes as your ipos.csv market column for has_domestic to
fire: the script prints your distinct market values and warns when none of
the assigned regions overlap them.

Matching is token-based (headers normalized to lowercase_underscores, alias
must appear as a contiguous token run), so bk_Goldman Sachs Asia matches
"goldman sachs" but bk_CITIC never matches "citi".
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# alias -> (class, home_region). Longest alias wins; first match by order.
# Classes: bb = top-tier US-anchored global five; global = other
# multinational full-service; regional = strong in one region; local =
# single-country. Reclassify to taste — this is a starting point.
KNOWN: list[tuple[str, str, str]] = [
    # ---- bulge bracket ----
    ("goldman sachs", "bb", "US"), ("goldman", "bb", "US"),
    ("morgan stanley", "bb", "US"),
    ("jpmorgan", "bb", "US"), ("jp morgan", "bb", "US"),
    ("j p morgan", "bb", "US"), ("jpm", "bb", "US"),
    ("bank of america", "bb", "US"), ("bofa", "bb", "US"),
    ("merrill lynch", "bb", "US"), ("merrill", "bb", "US"),
    ("baml", "bb", "US"),
    ("citigroup", "bb", "US"), ("citi", "bb", "US"),
    # ---- global ----
    ("ubs", "global", "CH"), ("credit suisse", "global", "CH"),
    ("deutsche bank", "global", "DE"), ("deutsche", "global", "DE"),
    ("barclays", "global", "GB"), ("hsbc", "global", "HK"),
    ("jefferies", "global", "US"), ("wells fargo", "global", "US"),
    ("rbc", "global", "CA"), ("royal bank of canada", "global", "CA"),
    ("macquarie", "global", "AU"), ("bnp paribas", "global", "FR"),
    ("bnp", "global", "FR"), ("nomura", "global", "JP"),
    # ---- regionals: Greater China ----
    ("cicc", "regional", "CN"), ("citic securities", "regional", "CN"),
    ("citic", "regional", "CN"), ("clsa", "regional", "HK"),
    ("haitong", "regional", "CN"), ("huatai", "regional", "CN"),
    ("guotai junan", "regional", "CN"), ("china securities", "regional", "CN"),
    ("galaxy", "regional", "CN"), ("gf securities", "regional", "CN"),
    ("guosen", "regional", "CN"), ("everbright", "regional", "CN"),
    ("cmb international", "regional", "HK"), ("cmbi", "regional", "HK"),
    ("boc international", "regional", "HK"), ("boci", "regional", "HK"),
    ("abc international", "regional", "HK"), ("abci", "regional", "HK"),
    ("ccb international", "regional", "HK"), ("ccbi", "regional", "HK"),
    ("icbc international", "regional", "HK"), ("icbci", "regional", "HK"),
    ("bocom international", "regional", "HK"), ("minsheng", "regional", "CN"),
    ("cinda", "regional", "CN"), ("futu", "local", "HK"),
    ("tiger brokers", "local", "CN"),
    # ---- regionals: Japan ----
    ("daiwa", "regional", "JP"), ("mizuho", "regional", "JP"),
    ("smbc nikko", "regional", "JP"), ("nikko", "regional", "JP"),
    ("smbc", "regional", "JP"), ("mitsubishi ufj", "regional", "JP"),
    ("mufg", "regional", "JP"), ("okasan", "local", "JP"),
    ("tokai tokyo", "local", "JP"),
    # ---- regionals: Korea ----
    ("mirae asset", "regional", "KR"), ("kb securities", "regional", "KR"),
    ("nh investment", "regional", "KR"), ("korea investment", "regional", "KR"),
    ("samsung securities", "regional", "KR"), ("shinhan", "local", "KR"),
    # ---- regionals: Europe ----
    ("societe generale", "regional", "FR"), ("socgen", "regional", "FR"),
    ("credit agricole", "regional", "FR"), ("natixis", "regional", "FR"),
    ("santander", "regional", "ES"), ("bbva", "regional", "ES"),
    ("unicredit", "regional", "IT"), ("intesa", "regional", "IT"),
    ("mediobanca", "regional", "IT"), ("commerzbank", "regional", "DE"),
    ("berenberg", "regional", "DE"), ("ing", "regional", "NL"),
    ("abn amro", "regional", "NL"), ("nordea", "regional", "SE"),
    ("seb", "regional", "SE"), ("carnegie", "regional", "SE"),
    ("dnb", "regional", "NO"), ("investec", "regional", "GB"),
    ("numis", "local", "GB"), ("peel hunt", "local", "GB"),
    ("panmure", "local", "GB"), ("cenkos", "local", "GB"),
    # ---- regionals/locals: North America mid-market ----
    ("cowen", "regional", "US"), ("piper sandler", "regional", "US"),
    ("piper jaffray", "regional", "US"), ("raymond james", "regional", "US"),
    ("william blair", "regional", "US"), ("stifel", "regional", "US"),
    ("baird", "regional", "US"), ("needham", "local", "US"),
    ("oppenheimer", "local", "US"), ("roth capital", "local", "US"),
    ("canaccord", "regional", "CA"), ("bmo", "regional", "CA"),
    ("cibc", "regional", "CA"), ("td securities", "regional", "CA"),
    ("scotiabank", "regional", "CA"), ("evercore", "regional", "US"),
    # ---- regionals/locals: SEA, India, Taiwan, Australia ----
    ("standard chartered", "regional", "HK"), ("dbs", "regional", "SG"),
    ("ocbc", "local", "SG"), ("uob", "local", "SG"),
    ("maybank", "regional", "MY"), ("cimb", "regional", "MY"),
    ("rhb", "local", "MY"), ("bualuang", "local", "TH"),
    ("kasikorn", "local", "TH"), ("mandiri", "local", "ID"),
    ("kotak", "regional", "IN"), ("icici", "regional", "IN"),
    ("axis capital", "regional", "IN"), ("jm financial", "regional", "IN"),
    ("motilal oswal", "local", "IN"), ("sbi capital", "regional", "IN"),
    ("yuanta", "regional", "TW"), ("fubon", "local", "TW"),
    ("ctbc", "local", "TW"), ("bell potter", "local", "AU"),
    ("ord minnett", "local", "AU"), ("wilsons", "local", "AU"),
]


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _match(key: str) -> tuple[str, str] | None:
    """Token-run matching: alias tokens must appear contiguously in the
    bank's tokens. Longest alias first, so 'citic securities' beats 'citic'
    and 'citic' beats nothing that would wrongly hit 'citi'."""
    toks = key.split("_")
    for alias, klass, home in sorted(KNOWN, key=lambda t: -len(t[0])):
        at = _norm(alias).split("_")
        n = len(at)
        if any(toks[i:i + n] == at for i in range(len(toks) - n + 1)):
            return klass, home
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, help="ipos.csv or .xlsx")
    p.add_argument("--sheet", default=None, help="sheet name for .xlsx")
    p.add_argument("--out", default="configs/bank_classes.csv")
    p.add_argument("--prefix", default="bk_")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists — pass --force to overwrite "
                         "(your hand edits would be lost).")

    path = Path(args.file)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=args.sheet or 0, nrows=5)
    else:
        df = pd.read_csv(path, nrows=5)
    bk = [c for c in df.columns if c.startswith(args.prefix)]
    if not bk:
        raise SystemExit(f"no {args.prefix}* columns in {path} — if your "
                         "headers are raw bank names, add the bk_ prefix "
                         "first (see WINDOWS_QUICKSTART.md).")

    rows, unknown = [], []
    for c in bk:
        key = _norm(c[len(args.prefix):])
        hit = _match(key)
        if hit:
            rows.append({"bank": key, "class": hit[0], "home_region": hit[1]})
        else:
            unknown.append({"bank": key, "class": "", "home_region": ""})

    # Unknowns FIRST so the ones needing your attention are at the top.
    frame = pd.DataFrame(unknown + rows).drop_duplicates(subset="bank")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print(f"{len(bk)} bank columns -> {out}")
    print(f"  {len(rows)} classified automatically (REVIEW THEM — judgment calls)")
    print(f"  {len(unknown)} need hand-filling (top of the file):")
    for u in unknown[:25]:
        print(f"    {u['bank']}")
    if len(unknown) > 25:
        print(f"    ... and {len(unknown) - 25} more")

    if "market" in df.columns:
        full = (pd.read_excel(path, sheet_name=args.sheet or 0, usecols=["market"])
                if path.suffix.lower() in (".xlsx", ".xls")
                else pd.read_csv(path, usecols=["market"]))
        markets = sorted(full["market"].astype(str).unique())
        regions = {r["home_region"] for r in rows}
        print(f"\nYour market codes: {markets}")
        if not regions & set(markets):
            print("WARNING: no assigned home_region matches any of your "
                  "market codes — has_domestic would be all-zero. Edit the "
                  "home_region column to use YOUR market labels.")


if __name__ == "__main__":
    main()
