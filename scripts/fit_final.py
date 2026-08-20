#!/usr/bin/env python3
"""Fit the deployable model on ALL labeled data and save a reusable bundle.

The walk-forward loop (run_full.py) tells you whether to trust the model;
this script trains the one you actually use going forward. It trains one
model per seed on every labeled row (latest val_frac reserved for early
stopping) and writes weights + fold scaler + label transform + schema to a
bundle directory. Score new IPOs later with scripts/predict.py — no
retraining needed.

    python scripts/fit_final.py --data data --out results/final_model --film
"""
import argparse

from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw
from ipo_model.persist import fit_final


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--film", action="store_true", help="FiLM gating by the GPR arm")
    p.add_argument("--out", default="results/final_model",
                   help="bundle directory to write")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    overrides = {}
    if args.seeds:
        overrides["train.seeds"] = tuple(args.seeds)
    if args.film:
        overrides["model.gating"] = "film"
    if overrides:
        cfg = cfg.override(**overrides)

    fs = build_features(load_raw(args.data), cfg.data)
    print(f"Fitting final model on {len(fs)} IPOs "
          f"({cfg.train.seeds} seeds, target {cfg.main_horizon}d)...")
    fit_final(cfg, fs, args.out)
    print(f"\nScore new deals with:\n"
          f"  python scripts/predict.py --data {args.data} --bundle {args.out} --new-only")


if __name__ == "__main__":
    main()
