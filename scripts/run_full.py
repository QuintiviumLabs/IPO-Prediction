#!/usr/bin/env python3
"""Train and evaluate the full three-arm model (V1 by default).

Use --film for the V2 gating extension, --seeds to control the ensemble.
"""
import argparse

from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw
from ipo_model.training import loop


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--film", action="store_true", help="FiLM gating by the GPR arm")
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
    print(f"{len(fs)} IPOs; target = {cfg.main_horizon}d excess log return from offer; "
          f"seeds={cfg.train.seeds}")
    res = loop.run(cfg, fs)
    print("\nAcross folds (mean ± std) | pooled OOS:")
    for k, (mu, sd) in res.summary.items():
        print(f"  {k:>18s}: {mu:+.4f} ± {sd:.4f} | {res.pooled[k]:+.4f}")


if __name__ == "__main__":
    main()
