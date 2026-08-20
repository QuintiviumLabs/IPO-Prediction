#!/usr/bin/env python3
"""Fill the blank class/home_region cells of bank_classes.csv with an LLM.

Sends your unclassified bank names to the OpenAI API in batches and asks
for, per bank: class (bb / global / regional / local), home_region, and a
confidence grade. Only rows where class or home_region is blank are sent;
your hand-filled rows are never touched. The file is checkpointed after
every batch, so an interrupted run resumes where it stopped.

    set OPENAI_API_KEY=sk-...            (PowerShell: $env:OPENAI_API_KEY="sk-...")
    pip install openai
    python scripts/fill_bank_classes_llm.py --csv configs/bank_classes.csv ^
        --markets "US,HK,CN,JP" --model gpt-5.6-luna

--markets should list the market codes YOUR ipos.csv uses — the model is
told to use those codes for home_region whenever the bank's home market is
among them, which is what makes has_domestic fire correctly.

TRUST ACCORDINGLY: an LLM classifying banks is a well-read intern, not a
league table. Everything it fills is marked in a `source` column
(llm_high / llm_low); review the llm_low rows by hand, and spot-check the
rest. Answers failing validation (unknown class, garbage region) are left
blank rather than guessed.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

CLASSES = {"bb", "global", "regional", "local"}
BATCH = 20

PROMPT = """You classify investment banks that appear as IPO bookrunners.
The names below are lowercase_underscore-normalized bank names (possibly
with suffixes like _asia, _international, _securities, _ag).

For EACH name return:
- "class": exactly one of
    "bb"       top-tier US-anchored bulge bracket (Goldman Sachs, Morgan
               Stanley, J.P. Morgan, Bank of America, Citigroup)
    "global"   other multinational full-service investment banks
    "regional" strong franchise concentrated in one region
    "local"    single-country or boutique player
- "home_region": the bank PARENT's home market as a short code{market_hint}
- "confidence": "high" if you are certain of the institution's identity,
  "low" if the name is ambiguous or you are inferring.

If a name is not identifiable as a real bank, use class "unknown".
Answer with JSON only: {{"banks": [{{"name": ..., "class": ...,
"home_region": ..., "confidence": ...}}, ...]}} — one entry per input name,
names copied verbatim.

Names:
{names}"""


def parse_response(text: str, expected: list[str]) -> dict[str, dict]:
    """Validate one batch response. Returns name -> {class, home_region,
    confidence} for entries that pass validation; invalid entries are
    dropped (left blank in the CSV), never guessed."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    out = {}
    want = set(expected)
    for e in data.get("banks", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        klass = str(e.get("class", "")).strip().lower()
        region = str(e.get("home_region", "")).strip().upper()
        conf = str(e.get("confidence", "low")).strip().lower()
        if (name in want and klass in CLASSES
                and re.fullmatch(r"[A-Z]{2,6}", region)):
            out[name] = {"class": klass, "home_region": region,
                         "confidence": "high" if conf == "high" else "low"}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="configs/bank_classes.csv")
    p.add_argument("--model", default="gpt-5.6-luna",
                   help="OpenAI model id — if the API rejects it, the script "
                        "lists your org's available models")
    p.add_argument("--markets", default=None,
                   help='your ipos.csv market codes, e.g. "US,HK,CN,JP" — '
                        "the model will prefer these for home_region")
    p.add_argument("--base-url", default=None,
                   help="override the API endpoint (corporate proxy/gateway)")
    p.add_argument("--limit", type=int, default=None,
                   help="only classify this many banks (smoke test first)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the first batch's prompt and exit — no API call")
    args = p.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    for col in ("bank", "class", "home_region"):
        if col not in df.columns:
            raise SystemExit(f"{path} is missing column '{col}'")
    if "source" not in df.columns:
        df["source"] = ""

    todo = df.index[(df["class"].str.strip() == "")
                    | (df["home_region"].str.strip() == "")].tolist()
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to fill — every row already has class and home_region")
        return
    print(f"{len(todo)} of {len(df)} banks need classification "
          f"(model: {args.model})")

    hint = (f" — when the home market is one of [{args.markets}], use "
            f"EXACTLY that code; otherwise a sensible ISO country code"
            if args.markets else ' (ISO-style, e.g. "US", "HK", "JP")')

    if args.dry_run:
        names = df.loc[todo[:BATCH], "bank"].tolist()
        print(PROMPT.format(market_hint=hint, names="\n".join(names)))
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    filled = low = 0
    for start in range(0, len(todo), BATCH):
        idx = todo[start: start + BATCH]
        names = df.loc[idx, "bank"].tolist()
        prompt = PROMPT.format(market_hint=hint, names="\n".join(names))
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "model" in msg.lower() and ("not" in msg.lower()
                                           or "404" in msg):
                try:
                    avail = sorted(m.id for m in client.models.list())
                    raise SystemExit(
                        f"model {args.model!r} rejected: {msg}\n"
                        f"available to your org: {avail}")
                except SystemExit:
                    raise
                except Exception:
                    pass
            raise SystemExit(f"API call failed: {msg}")

        got = parse_response(text, names)
        for i in idx:
            hit = got.get(df.at[i, "bank"])
            if hit is None:
                continue
            if hit["class"] == "unknown":
                continue
            df.at[i, "class"] = hit["class"]
            df.at[i, "home_region"] = hit["home_region"]
            df.at[i, "source"] = f"llm_{hit['confidence']}"
            filled += 1
            low += hit["confidence"] == "low"
        df.to_csv(path, index=False)          # checkpoint every batch
        print(f"  {min(start + BATCH, len(todo))}/{len(todo)} "
              f"({filled} filled, {low} low-confidence)")

    still = int(((df["class"].str.strip() == "")
                 | (df["home_region"].str.strip() == "")).sum())
    print(f"\nDone: {filled} filled ({low} marked llm_low — REVIEW THESE), "
          f"{still} still blank -> {path}")
    print("Spot-check llm_high rows too, then re-run "
          "build_bookrunner_features.py.")
    if still:
        sys.exit(1)


if __name__ == "__main__":
    main()
