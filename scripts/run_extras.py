#!/usr/bin/env python3
"""Run the optional add-on rungs (ipo_model/extras) on the same purged folds.

    python scripts\\run_extras.py --data data --config configs\\small.yaml --what linear
    python scripts\\run_extras.py --data data --config configs\\small.yaml --what tabpfn
    python scripts\\run_extras.py --data data --config configs\\small.yaml --what linear tabpfn

Needs the optional dependencies:  pip install -r requirements-extras.txt
Results are directly comparable to results\\ablations.md — same folds, same
features, same metrics.
"""
import argparse
from pathlib import Path

import pandas as pd

from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw


def _report(name: str, res) -> dict:
    print("\nAcross folds (mean ± std) | pooled OOS:")
    for k, (mu, sd) in res.summary.items():
        print(f"  {k:>18s}: {mu:+.4f} ± {sd:.4f} | {res.pooled[k]:+.4f}")
    row = {"rung": name}
    for m in ("rank_ic", "hit_rate", "decile_spread", "mae", "pinball", "coverage"):
        row[m] = res.summary[m][0]
        row[f"{m}_std"] = res.summary[m][1]
        row[f"{m}_pooled"] = res.pooled[m]
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--what", nargs="+", default=["linear"],
                   choices=["linear", "tabpfn"])
    p.add_argument("--model", default="ridge",
                   choices=["ridge", "lasso", "elasticnet"],
                   help="linear only: which regularized estimator")
    p.add_argument("--quantile-method", default="residual",
                   choices=["residual", "quantreg"], help="linear only")
    p.add_argument("--device", default="auto", help="tabpfn only: cpu | cuda | auto")
    p.add_argument("--n-estimators", type=int, default=4, help="tabpfn only")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    fs = build_features(load_raw(args.data), cfg.data)
    print(f"{len(fs)} IPOs; extras: {args.what}")
    rows = []

    if "linear" in args.what:
        from ipo_model.extras import linear
        name = f"x_{args.model}"
        print(f"\n=== {name} ({args.quantile_method} quantiles) ===")
        res = linear.run(cfg, fs, model=args.model,
                         quantile_method=args.quantile_method)
        rows.append(_report(name, res))

        coefs = linear.coefficients(cfg, fs, model=args.model)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        coefs.to_csv(out / f"{name}_coefficients.csv", index=False)
        print(f"\nTop standardized coefficients (mean across folds):")
        print(coefs.head(12).round(4).to_string(index=False))

    if "tabpfn" in args.what:
        from ipo_model.extras import tabpfn_model
        print("\n=== x_tabpfn ===")
        res = tabpfn_model.run(cfg, fs, device=args.device,
                               n_estimators=args.n_estimators)
        rows.append(_report("x_tabpfn", res))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).set_index("rung")
    df.to_csv(out / "extras.csv")
    print(f"\nWrote {out}/extras.csv")


if __name__ == "__main__":
    main()
