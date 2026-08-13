#!/usr/bin/env python3
"""Run the ablation ladder (rung 0 = LightGBM, rungs 1-6 = deep variants).

Writes results/ablations.csv and results/ablations.md.
"""
import argparse

from ipo_model.ablation import EXTRA_RUNGS, RUNGS, run_ladder
from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--rungs", nargs="+", default=None,
                   help=f"subset of: 0_lgbm 0_xgb 0_lgbm_raw 0_xgb_raw "
                        f"{' '.join(RUNGS)} | opt-in extras (need "
                        f"requirements-extras.txt): {' '.join(EXTRA_RUNGS)}")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.seeds:
        cfg = cfg.override(**{"train.seeds": tuple(args.seeds)})
    fs = build_features(load_raw(args.data), cfg.data)
    print(f"{len(fs)} IPOs; seeds={cfg.train.seeds}")
    df = run_ladder(cfg, fs, rungs=args.rungs, out_dir=args.out)
    with_std = df[[c for c in df.columns if not c.endswith("_pooled")]]
    print("\n" + with_std.round(4).to_string())
    print(f"\nWrote {args.out}/ablations.csv and {args.out}/ablations.md")


if __name__ == "__main__":
    main()
