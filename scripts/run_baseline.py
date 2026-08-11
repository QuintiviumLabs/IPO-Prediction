#!/usr/bin/env python3
"""Ablation 0: LightGBM on engineered features over purged walk-forward folds."""
import argparse

from ipo_model.baselines import lgbm
from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    fs = build_features(load_raw(args.data), cfg.data)
    print(f"{len(fs)} IPOs with full {cfg.main_horizon}d labels; "
          f"target = {cfg.main_horizon}d excess log return from offer price")
    res = lgbm.run(cfg, fs)
    print("\nAcross folds (mean ± std) | pooled OOS:")
    for k, (mu, sd) in res.summary.items():
        print(f"  {k:>18s}: {mu:+.4f} ± {sd:.4f} | {res.pooled[k]:+.4f}")


if __name__ == "__main__":
    main()
