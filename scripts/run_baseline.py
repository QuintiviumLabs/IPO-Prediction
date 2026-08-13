#!/usr/bin/env python3
"""Ablation 0: trees on engineered features over purged walk-forward folds."""
import argparse

from ipo_model.baselines import lgbm, xgb
from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw
from ipo_model.results_io import save_predictions

ENGINES = {"lgbm": lgbm.run, "xgb": xgb.run}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--engine", choices=[*ENGINES, "both"], default="lgbm")
    p.add_argument("--features", choices=["engineered", "raw", "both"],
                   default="engineered",
                   help="engineered summaries, or the full flattened inputs")
    p.add_argument("--out", default="results",
                   help="per-deal predictions are saved under <out>/predictions/")
    p.add_argument("--no-save", action="store_true",
                   help="skip saving predictions (they cannot be recovered later)")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    fs = build_features(load_raw(args.data), cfg.data)
    print(f"{len(fs)} IPOs with full {cfg.main_horizon}d labels; "
          f"target = {cfg.main_horizon}d excess log return from offer price")
    engines = list(ENGINES) if args.engine == "both" else [args.engine]
    feature_sets = (["engineered", "raw"] if args.features == "both"
                    else [args.features])
    for name in engines:
        for features in feature_sets:
            print(f"\n=== {name} ({features}) ===")
            res = ENGINES[name](cfg, fs, features=features)
            print("\nAcross folds (mean ± std) | pooled OOS:")
            for k, (mu, sd) in res.summary.items():
                print(f"  {k:>18s}: {mu:+.4f} ± {sd:.4f} | {res.pooled[k]:+.4f}")
            if not args.no_save:
                rung = f"0_{name}" + ("_raw" if features == "raw" else "")
                print(f"  predictions -> "
                      f"{save_predictions(cfg, fs, res, rung, out_dir=args.out)}")


if __name__ == "__main__":
    main()
