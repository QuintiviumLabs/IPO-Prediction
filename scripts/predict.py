#!/usr/bin/env python3
"""Score IPOs with a saved bundle (see scripts/fit_final.py) — no retraining.

To score a brand-new deal that hasn't traded yet:
  1. Append its row to ipos.csv: ipo_id, first_trade_date (planned), market,
     offer_price, deal_size, sector flags, bk_* bookrunner columns, and any
     syndicate-extra columns your training data had. No price rows needed.
  2. Refresh peers.csv (scripts/pull_peers.py) and the macro csvs so its
     market-state features exist as of pricing.
  3. python scripts/predict.py --data data --bundle results/final_model --new-only

Output: one row per deal with q10/q50/q90 (or your configured quantiles) of
the main-horizon log outperformance, plus the exp() multiples. Without
--new-only every row is scored (useful as a sanity check against
training-period deals — but those are IN-SAMPLE numbers, not OOS).
"""
import argparse

import numpy as np
import pandas as pd

from ipo_model.data.features import build_features, load_raw
from ipo_model.persist import load_bundle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--bundle", default="results/final_model")
    p.add_argument("--new-only", action="store_true",
                   help="score only rows without a full main-horizon label "
                        "(i.e. deals the model could not have trained on)")
    p.add_argument("--out", default=None, help="optional CSV to write")
    args = p.parse_args()

    bundle = load_bundle(args.bundle)
    cfg = bundle.cfg
    fs = build_features(load_raw(args.data), cfg.data, inference=True)

    main_h = cfg.main_horizon
    idx = np.arange(len(fs))
    if args.new_only:
        idx = idx[~np.isfinite(fs.y[main_h])]
        if not len(idx):
            print("No unlabeled rows found — every deal in ipos.csv already "
                  "has a full label window. Re-run without --new-only to "
                  "score them anyway (in-sample).")
            return

    preds = bundle.predict(fs, idx=idx)
    q = preds[main_h]
    levels = cfg.model.quantiles

    rows = {
        "ipo_id": fs.ids[idx],
        "first_trade_date": pd.to_datetime(fs.dates[idx]).strftime("%Y-%m-%d"),
        "market": [fs.market_names[j] for j in fs.market_onehot[idx].argmax(1)],
    }
    for qi, lv in enumerate(levels):
        rows[f"q{int(round(lv * 100))}"] = q[:, qi]
    for qi, lv in enumerate(levels):
        rows[f"mult_q{int(round(lv * 100))}"] = np.exp(q[:, qi])
    df = pd.DataFrame(rows)

    med = len(levels) // 2
    print(f"Bundle: {args.bundle} (trained on {bundle.meta['n_train_rows']} deals, "
          f"{bundle.meta['train_date_range'][0]} .. {bundle.meta['train_date_range'][1]})")
    print(f"Main horizon {main_h}d; median forecast is the outperformance "
          f"multiple vs the deal's own benchmark.\n")
    with pd.option_context("display.width", 140, "display.max_rows", 50):
        print(df.round(4).to_string(index=False))
    lo, hi = int(round(levels[0] * 100)), int(round(levels[-1] * 100))
    print(f"\nRead q50 as the central call; [q{lo}, q{hi}] is the raw model "
          f"band — apply scripts/calibrate_intervals.py factors for "
          f"honest coverage.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
