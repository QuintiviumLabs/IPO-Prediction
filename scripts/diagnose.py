#!/usr/bin/env python3
"""Feature and failure diagnostics over purged walk-forward folds.

  --what tree    LightGBM gain importance per engineered feature (fast)
  --what perm    permutation importance on the deep model (trains 1 seed/fold)
  --what slice   OOS rank IC / MAE / coverage by market, sector, GPR regime,
                 momentum drought (runs the deep model first)

Writes CSVs to results/diagnostics/.
"""
import argparse
from pathlib import Path

from ipo_model import diagnostics
from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw
from ipo_model.training import loop


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--what", nargs="+", default=["tree", "perm", "slice"],
                   choices=["tree", "perm", "slice"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="results/diagnostics")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    fs = build_features(load_raw(args.data), cfg.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(fs)} IPOs; diagnostics: {args.what}")

    if "tree" in args.what:
        print("\n=== LightGBM gain importance (share of total gain) ===")
        df = diagnostics.tree_gain_importance(cfg, fs)
        df.to_csv(out / "tree_gain_importance.csv", index=False)
        print(df.head(20).round(4).to_string(index=False))

    if "perm" in args.what:
        print("\n=== Deep-model permutation importance (rank-IC drop) ===")
        df = diagnostics.deep_permutation_importance(cfg, fs, n_repeats=args.repeats)
        df.to_csv(out / "permutation_importance.csv", index=False)
        print(df.round(4).to_string(index=False))

    if "slice" in args.what:
        print("\n=== OOS metrics by slice (deep model, 1 seed) ===")
        fast = cfg.override(**{"train.seeds": (cfg.train.seeds[0],)})
        res = loop.run(fast, fs, verbose=False)
        df = diagnostics.slice_metrics(cfg, fs, res)
        df.to_csv(out / "slice_metrics.csv", index=False)
        print(df.round(4).to_string(index=False))

    print(f"\nWrote CSVs to {out}/")


if __name__ == "__main__":
    main()
