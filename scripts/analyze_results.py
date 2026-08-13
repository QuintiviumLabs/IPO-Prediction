#!/usr/bin/env python3
"""Statistical analysis of saved predictions — no retraining required.

`run_ablations.py` writes every rung's per-deal out-of-sample predictions to
results/predictions/. This reads them back and answers the questions a
reviewer will ask:

  1. Is each rung's rank IC different from zero?   (quarterly IC series,
     Newey-West t-test, plus a block-bootstrap CI on the pooled IC)
  2. Does each rung beat the baseline on the SAME deals?  (paired
     Diebold-Mariano test on pinball loss + paired bootstrap on rank IC)
  3. Do the winners survive testing many rungs at once?  (Benjamini-Hochberg)

    python scripts\\analyze_results.py --results results
    python scripts\\analyze_results.py --results results --baseline 0_lgbm

Writes significance.csv, comparisons.csv and analysis.md alongside the
predictions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_model.results_io import available_rungs, load_predictions, quantile_columns
from ipo_model.training.metrics import (benjamini_hochberg, bootstrap_ci,
                                        compare_models, evaluate, ic_by_fold,
                                        ic_by_period, t_test)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--baseline", default=None,
                   help="rung every other rung is compared against "
                        "(default: the first naive or lgbm rung present)")
    p.add_argument("--freq", default="QE", help="IC period: ME | QE | YE")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rungs = available_rungs(args.results)
    if not rungs:
        raise SystemExit(f"No saved predictions in {args.results}/predictions/. "
                         "Run scripts/run_ablations.py first.")
    print(f"Found {len(rungs)} rungs: {', '.join(rungs)}")

    preds = {r: load_predictions(r, args.results) for r in rungs}
    baseline = args.baseline
    if baseline is None:
        for cand in ("00_naive_f1", "00_naive_const", "0_lgbm"):
            if cand in preds:
                baseline = cand
                break
        baseline = baseline or rungs[0]
    if baseline not in preds:
        raise SystemExit(f"baseline {baseline!r} not among {rungs}")
    print(f"Baseline for paired comparisons: {baseline}")

    # ---- 1. Is each rung's IC different from zero? ----
    sig_rows = []
    for rung, df in preds.items():
        q, levels = quantile_columns(df)
        y = df["y_true"].to_numpy(float)
        m = evaluate(y, q, levels)
        per = ic_by_period(y, q, df["date"].to_numpy(), levels, freq=args.freq)
        tt = t_test(per["rank_ic"].to_numpy(), hac=True) if len(per) >= 2 else \
            {"mean": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": len(per)}
        boot = bootstrap_ci(y, q, levels, metric="rank_ic", n_boot=args.n_boot,
                            groups=df["date"].dt.year.to_numpy(), seed=args.seed)
        by_fold = ic_by_fold(y, q, df["fold"].to_numpy(), levels)
        sig_rows.append({
            "rung": rung, "n_deals": int(m["n"]),
            "ic_fold_mean": float(np.nanmean(by_fold["rank_ic"])),
            "ic_pooled": m["rank_ic"], "quintile_spread": m["quintile_spread"],
            "r2_vs_zero": m["r2_vs_zero"], "coverage": m["coverage"],
            "periods": int(tt["n"]), "mean_period_ic": tt["mean"],
            "ic_t": tt["t"], "ic_p": tt["p"],
            "ic_boot_lo": boot["lo"], "ic_boot_hi": boot["hi"],
        })
    sig = pd.DataFrame(sig_rows).sort_values("ic_fold_mean", ascending=False)
    bh = benjamini_hochberg(sig["ic_p"].to_numpy())
    sig["ic_q_value"] = bh["q_value"].to_numpy()

    print("\n=== Is the rank IC different from zero? ===")
    print("(ic_fold_mean is the honest headline; ic_pooled mixes folds and can "
          "be inflated by between-fold level shifts)")
    print(sig[["rung", "n_deals", "ic_fold_mean", "ic_pooled", "ic_boot_lo",
               "ic_boot_hi", "periods", "ic_t", "ic_p", "ic_q_value"]]
          .round(4).to_string(index=False))

    # ---- 2. Paired comparisons against the baseline ----
    base = preds[baseline]
    qb, levels = quantile_columns(base)
    cmp_rows = []
    for rung, df in preds.items():
        if rung == baseline:
            continue
        merged = base[["ipo_id", "y_true"]].merge(
            df, on="ipo_id", suffixes=("_base", ""), how="inner")
        if len(merged) < 30:
            print(f"  skipping {rung}: only {len(merged)} shared deals")
            continue
        qa, _ = quantile_columns(merged)
        base_aligned = base.set_index("ipo_id").loc[merged["ipo_id"]]
        qb_aligned, _ = quantile_columns(base_aligned.reset_index())
        y = merged["y_true"].to_numpy(float)
        res = compare_models(y, qa, qb_aligned, levels, name_a=rung,
                             name_b=baseline,
                             groups=merged["date"].dt.year.to_numpy(),
                             n_boot=args.n_boot, seed=args.seed)
        cmp_rows.append(res)

    comparisons = pd.DataFrame(cmp_rows)
    if len(comparisons):
        comparisons = comparisons.sort_values("rank_ic_delta", ascending=False)
        bh2 = benjamini_hochberg(comparisons["rank_ic_delta_p"].to_numpy())
        comparisons["delta_q_value"] = bh2["q_value"].to_numpy()
        print(f"\n=== Does each rung beat {baseline} on the same deals? ===")
        print("(positive rank_ic_delta and negative pinball_delta favour the rung)")
        print(comparisons[["model_a", "rank_ic_delta", "rank_ic_delta_lo",
                           "rank_ic_delta_hi", "rank_ic_delta_p", "delta_q_value",
                           "pinball_delta", "dm_t", "dm_p"]]
              .round(4).to_string(index=False))

    out = Path(args.results)
    sig.to_csv(out / "significance.csv", index=False)
    if len(comparisons):
        comparisons.to_csv(out / "comparisons.csv", index=False)
    _write_markdown(out / "analysis.md", sig, comparisons, baseline, args.freq)
    print(f"\nWrote {out}/significance.csv, {out}/comparisons.csv, {out}/analysis.md")


def _write_markdown(path: Path, sig: pd.DataFrame, cmp_: pd.DataFrame,
                    baseline: str, freq: str) -> None:
    L = ["# Statistical analysis", "",
         f"Baseline for paired tests: `{baseline}`. IC periods: `{freq}`.",
         "", "## Is the rank IC different from zero?", "",
         "**Read `IC (fold mean)`, not `IC (pooled)`.** Pooling mixes deals "
         "across folds, and since each fold's model is trained on different "
         "data, systematic level differences between folds can inflate the "
         "pooled correlation without any real within-fold ordering skill.",
         "",
         "`t` / `p` come from a Newey-West t-test of the period-by-period IC "
         "series; the bootstrap CI is year-blocked (deals in the same year "
         "are not independent). `q` is the Benjamini-Hochberg adjustment for "
         "testing every rung at once.", "",
         "| rung | n | IC (fold mean) | IC (pooled) | boot 95% CI | periods | t | p | q |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sig.iterrows():
        L.append(f"| {r['rung']} | {r['n_deals']:.0f} | "
                 f"{r['ic_fold_mean']:+.4f} | {r['ic_pooled']:+.4f} | "
                 f"[{r['ic_boot_lo']:+.3f}, {r['ic_boot_hi']:+.3f}] | "
                 f"{r['periods']:.0f} | {r['ic_t']:+.2f} | {r['ic_p']:.4f} | "
                 f"{r['ic_q_value']:.4f} |")
    if len(cmp_):
        L += ["", f"## Paired comparison against `{baseline}`", "",
              "Same deals, both models. Positive `rank_ic_delta` and negative "
              "`pinball_delta` favour the rung. `dm_*` is a Diebold-Mariano "
              "test on the per-deal pinball loss differential.", "",
              "| rung | ΔIC | Δ 95% CI | p | q | Δpinball | DM t | DM p |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in cmp_.iterrows():
            L.append(f"| {r['model_a']} | {r['rank_ic_delta']:+.4f} | "
                     f"[{r['rank_ic_delta_lo']:+.3f}, {r['rank_ic_delta_hi']:+.3f}] | "
                     f"{r['rank_ic_delta_p']:.4f} | {r['delta_q_value']:.4f} | "
                     f"{r['pinball_delta']:+.5f} | {r['dm_t']:+.2f} | {r['dm_p']:.4f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
