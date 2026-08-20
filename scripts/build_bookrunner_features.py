#!/usr/bin/env python3
"""Derive the bookrunner-syndicate feature block from bk_* columns.

No Bloomberg needed — this is pure bookkeeping over the bk_* multi-hot
columns you already have, plus a hand-maintained bank classification file
(configs/bank_classes.csv: bank, class, home_region). Writes into ipos.csv
(or an .xlsx — pass the file directly):

  n_banks, log_n_banks                syndicate size
  has_bb / has_global / has_regional  bank-class presence flags
  has_local / has_domestic            (domestic = any bank's home_region ==
                                      the deal's market)
  combo_bb_x_dom, combo_bb_x_reg      interaction flags
  solo_book, jumbo_synd               size == 1 / size >= --jumbo
  archetype_*                         one-hot syndicate archetype (below)

prestige_rank_max is NOT written here — the pipeline computes it expanding
(trailing --prestige-years share of same-market IPO proceeds) so it can
never see post-pricing data.

Default archetype taxonomy (8, mutually exclusive, applied top-down —
REVIEW: this is a sensible default, not a market convention):
  bb_solo        one bank, bulge bracket
  bb_dom         BB present + a domestic bank present
  bb_intl        BB present, no domestic bank
  global_led     no BB, a global bank present
  regional_dom   regional-led with a domestic bank
  regional_intl  regional-led, no domestic bank
  local_only     only local banks
  unclassified   no bank matched bank_classes.csv

    python scripts/build_bookrunner_features.py --file data/ipos.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ARCHETYPES = ["bb_solo", "bb_dom", "bb_intl", "global_led",
              "regional_dom", "regional_intl", "local_only", "unclassified"]


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


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


def archetype(nb: int, classes: set[str], domestic: bool) -> str:
    if not classes:
        return "unclassified"
    if "bb" in classes:
        if nb == 1:
            return "bb_solo"
        return "bb_dom" if domestic else "bb_intl"
    if "global" in classes:
        return "global_led"
    if "regional" in classes:
        return "regional_dom" if domestic else "regional_intl"
    return "local_only"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, help="ipos.csv or an .xlsx export")
    p.add_argument("--sheet", default=None, help="sheet name for .xlsx files")
    p.add_argument("--classes", default="configs/bank_classes.csv")
    p.add_argument("--jumbo", type=int, default=6,
                   help="jumbo_synd = 1 when n_banks >= this (spec: >= 6)")
    p.add_argument("--prefix", default="bk_")
    args = p.parse_args()

    path = Path(args.file)
    df = _read(path, args.sheet)
    bk_cols = [c for c in df.columns if c.startswith(args.prefix)]
    if not bk_cols:
        raise SystemExit(f"no {args.prefix}* columns found in {path}")

    cl = pd.read_csv(args.classes, comment="#", dtype={"bank": str})
    # Rows with a blank class (e.g. make_bank_classes.py placeholders you
    # haven't filled yet) count as unmatched, not as some default class.
    cl = cl[cl["class"].fillna("").astype(str).str.strip() != ""]
    cl["bank"] = cl["bank"].map(_norm)
    klass = dict(zip(cl["bank"], cl["class"].str.lower()))
    home = dict(zip(cl["bank"], cl["home_region"].astype(str)))

    matched, unmatched = [], []
    for c in bk_cols:
        key = _norm(c[len(args.prefix):])
        (matched if key in klass else unmatched).append((c, key))
    print(f"{len(bk_cols)} bk_ columns: {len(matched)} matched to "
          f"bank_classes.csv, {len(unmatched)} unmatched")
    if unmatched:
        print("  unmatched (add rows to bank_classes.csv): "
              + ", ".join(k for _, k in unmatched[:20])
              + (" ..." if len(unmatched) > 20 else ""))

    bk = df[bk_cols].fillna(0).to_numpy(float) > 0
    mkt = df["market"].astype(str) if "market" in df.columns else pd.Series([""] * len(df))
    nb = bk.sum(axis=1).astype(int)

    rows = []
    for i in range(len(df)):
        banks = [bk_cols[j] for j in np.where(bk[i])[0]]
        keys = [_norm(b[len(args.prefix):]) for b in banks]
        classes = {klass[k] for k in keys if k in klass}
        domestic = any(home.get(k) == mkt.iloc[i] for k in keys)
        r = {
            "n_banks": int(nb[i]),
            "log_n_banks": float(np.log1p(nb[i])),
            "has_bb": int("bb" in classes),
            "has_global": int("global" in classes),
            "has_regional": int("regional" in classes),
            "has_local": int("local" in classes),
            "has_domestic": int(domestic),
            "combo_bb_x_dom": int("bb" in classes and domestic),
            "combo_bb_x_reg": int("bb" in classes and "regional" in classes),
            "solo_book": int(nb[i] == 1),
            "jumbo_synd": int(nb[i] >= args.jumbo),
        }
        a = archetype(int(nb[i]), classes, domestic)
        for name in ARCHETYPES:
            r[f"archetype_{name}"] = int(a == name)
        rows.append(r)

    feats = pd.DataFrame(rows)
    for c in feats.columns:      # overwrite stale values on re-run
        df[c] = feats[c].to_numpy()
    _write(df, path, args.sheet)

    arch_counts = {a: int(feats[f"archetype_{a}"].sum()) for a in ARCHETYPES}
    print(f"archetype counts: {arch_counts}")
    if arch_counts["unclassified"] > 0.2 * len(df):
        print("WARNING: >20% of deals have no classified bank — extend "
              "bank_classes.csv before trusting these features.")
    print(f"written -> {path}")


if __name__ == "__main__":
    main()
