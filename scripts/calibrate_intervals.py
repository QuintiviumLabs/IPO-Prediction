#!/usr/bin/env python3
"""Conformally calibrate a rung's saved prediction intervals — no retraining.

    python scripts\\calibrate_intervals.py --results results --rung full_v1
    python scripts\\calibrate_intervals.py --results results --rung full_v1 --mode sym

Reads results/predictions/<rung>.csv, computes per-tail conformal margins for
each fold from strictly earlier folds' out-of-sample errors, and writes the
calibrated band to results/predictions/<rung>_cqr.csv (medians and therefore
rank IC are untouched). Prints per-tail coverage before/after, on the
calibrated folds only — the earliest fold has no history and passes through.
"""
import argparse
from pathlib import Path

from ipo_model.calibrate import calibrate, tail_report
from ipo_model.results_io import load_predictions, quantile_columns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--rung", required=True)
    p.add_argument("--mode", choices=["asym", "sym"], default="asym")
    p.add_argument("--min-cal", type=int, default=80,
                   help="minimum earlier-fold rows required to calibrate a fold")
    p.add_argument("--window", type=int, default=None,
                   help="calibrate on only the most recent N prior rows — "
                        "adapts to regime drift; None uses all prior rows")
    args = p.parse_args()

    df = load_predictions(args.rung, args.results)
    _, levels = quantile_columns(df)
    if len(levels) < 2:
        raise SystemExit(
            f"{args.rung} holds binary-head probabilities (p_out), not "
            "quantile intervals — nothing to conformally calibrate. Train a "
            "quantile run (model.head: quantile) for interval work.")
    cal_df, rep = calibrate(df, levels, mode=args.mode, min_cal=args.min_cal,
                            window=args.window)

    done = rep[rep["calibrated"]]["fold"].tolist()
    skipped = rep[~rep["calibrated"]]["fold"].tolist()
    if not done:
        raise SystemExit("No fold had enough earlier history to calibrate — "
                         "lower --min-cal or check the fold column.")
    print(f"Calibrated folds {done} (mode={args.mode}); "
          f"passed through unchanged: {skipped or 'none'}")
    print("\nPer-fold margins (negative shrinks an over-wide edge):")
    print(rep.round(4).to_string(index=False))

    before = tail_report(df, levels, folds=done)
    after = tail_report(cal_df, levels, folds=done)
    print(f"\nOn the calibrated folds (n={before['n']}):")
    print(f"{'':>16}{'below-lo':>10}{'above-hi':>10}{'coverage':>10}{'width':>8}")
    print(f"{'nominal':>16}{before['nominal_lo']:>10.3f}{before['nominal_hi']:>10.3f}"
          f"{before['nominal_coverage']:>10.3f}{'—':>8}")
    print(f"{'before':>16}{before['frac_below_lo']:>10.3f}{before['frac_above_hi']:>10.3f}"
          f"{before['coverage']:>10.3f}{before['mean_width']:>8.3f}")
    print(f"{'after':>16}{after['frac_below_lo']:>10.3f}{after['frac_above_hi']:>10.3f}"
          f"{after['coverage']:>10.3f}{after['mean_width']:>8.3f}")

    out = Path(args.results) / "predictions" / f"{args.rung}_cqr.csv"
    cal_df.to_csv(out, index=False)
    rep.to_csv(Path(args.results) / f"calibration_{args.rung}.csv", index=False)
    print(f"\nWrote {out} (appears as rung '{args.rung}_cqr' in analyze_results)")


if __name__ == "__main__":
    main()
