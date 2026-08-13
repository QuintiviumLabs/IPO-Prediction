#!/usr/bin/env python3
"""Is the out-of-sample score real, or is the pipeline broken?

Two checks that answer different questions:

  permute   Randomly reassign realized outcomes among deals, keeping every
            feature, date and fold identical, then re-run. The link between
            inputs and outcomes is destroyed, so a correct pipeline MUST
            score ~0. Anything meaningfully above 0 means the model is
            exploiting something other than a genuine relationship — a
            leak in feature construction, splitting, or evaluation.
            This is the single most informative check in the repo.

  traingap  Rank IC on the training rows vs the held-out rows. A large gap
            means the model overfits its training data — which is normal and
            does NOT invalidate the test score, but tells you whether
            capacity or regularization needs attention.

    python scripts\\sanity_check.py --data data --config configs\\small.yaml
    python scripts\\sanity_check.py --data data --what permute --repeats 10 --model deep

Note what this canNOT detect: leakage that is already baked into the input
files. If deals that delisted early are missing from ipos.csv (survivorship),
or the GPR series was retrospectively revised, the permutation test passes
cleanly and the score is still inflated. Those require checking the data
source, not the code.
"""
import argparse

import numpy as np
import pandas as pd

from ipo_model.baselines import lgbm
from ipo_model.config import Config
from ipo_model.data.features import build_features, load_raw
from ipo_model.data.preprocess import FoldScaler, engineered_table
from ipo_model.data.splits import purged_walk_forward
from ipo_model.training import loop
from ipo_model.training.metrics import evaluate


def _run_model(cfg, fs, model: str):
    return (loop.run(cfg, fs, verbose=False) if model == "deep"
            else lgbm.run(cfg, fs, verbose=False))


def _pooled_ic(cfg, fs, res) -> float:
    idx = np.concatenate([r.test_idx for r in res.fold_results])
    q = np.concatenate([r.q_pred for r in res.fold_results])
    return evaluate(fs.y[cfg.main_horizon][idx], q, cfg.model.quantiles)["rank_ic"]


def _fold_mean_ic(cfg, fs, res) -> float:
    ics = []
    for r in res.fold_results:
        m = evaluate(fs.y[cfg.main_horizon][r.test_idx], r.q_pred,
                     cfg.model.quantiles)
        ics.append(m["rank_ic"])
    return float(np.nanmean(ics))


def permutation_test(cfg, fs, model: str, repeats: int, seed: int) -> pd.DataFrame:
    real = _run_model(cfg, fs, model)
    real_ic = _fold_mean_ic(cfg, fs, real)
    print(f"  actual fold-mean IC: {real_ic:+.4f}")

    horizons = sorted(fs.y)
    original = {h: fs.y[h].copy() for h in horizons}
    rows = []
    try:
        for i in range(repeats):
            rng = np.random.default_rng(seed + i)
            # ONE permutation applied to every horizon, so a deal keeps its
            # own set of outcomes — otherwise the auxiliary heads would still
            # see the true label and the test would be meaningless.
            perm = rng.permutation(len(fs))
            for h in horizons:
                fs.y[h] = original[h][perm]
            res = _run_model(cfg, fs, model)
            ic = _fold_mean_ic(cfg, fs, res)
            rows.append({"repeat": i, "shuffled_ic": ic})
            print(f"  shuffled run {i + 1}/{repeats}: IC {ic:+.4f}")
    finally:
        for h in horizons:  # always restore, even on Ctrl-C
            fs.y[h] = original[h]

    df = pd.DataFrame(rows)
    mu, sd = df["shuffled_ic"].mean(), df["shuffled_ic"].std()
    print(f"\n  shuffled IC: mean {mu:+.4f}, sd {sd:.4f}, "
          f"max |IC| {df['shuffled_ic'].abs().max():.4f}")
    if repeats >= 2 and sd > 0:
        z = (real_ic - mu) / sd
        print(f"  actual IC sits {z:.1f} sd above the shuffled distribution")
    if abs(mu) > 0.05:
        print("\n  *** WARNING: shuffled outcomes still score above 0.05. "
              "The pipeline is finding signal that cannot exist. Investigate "
              "feature construction and splitting before trusting any result.")
    else:
        print("\n  PASS: shuffled outcomes score ~0, so the split, feature "
              "construction and evaluation are not manufacturing signal.")
        print("  (This does NOT rule out leakage already present in the input "
              "files — survivorship, revised GPR vintages, restated indices.)")
    df.attrs["real_ic"] = real_ic
    return df


def train_gap(cfg, fs, seed: int) -> pd.DataFrame:
    """Rank IC on training rows vs held-out rows, per fold (LightGBM)."""
    import lightgbm as lgb

    folds = purged_walk_forward(fs.dates, fs.label_end, cfg.split)
    y = fs.y[cfg.main_horizon]
    rows = []
    for k, fold in enumerate(folds):
        scaler = FoldScaler.fit(fs, fold.train_idx, cfg.data)
        X = engineered_table(fs, scaler)
        booster = lgb.train(
            dict(cfg.lgbm.params, seed=seed),
            lgb.Dataset(X.iloc[fold.train_idx], label=y[fold.train_idx]),
            num_boost_round=cfg.lgbm.num_boost_round,
            valid_sets=[lgb.Dataset(X.iloc[fold.val_idx], label=y[fold.val_idx])],
            callbacks=[lgb.early_stopping(cfg.lgbm.early_stopping_rounds,
                                          verbose=False)],
        )
        def ic(idx):
            p = booster.predict(X.iloc[idx], num_iteration=booster.best_iteration)
            q = np.stack([p, p, p], axis=1)  # ordering only; band is irrelevant
            return evaluate(y[idx], q, (0.1, 0.5, 0.9))["rank_ic"]
        rows.append({"fold": k, "n_train": len(fold.train_idx),
                     "train_ic": ic(fold.train_idx),
                     "test_ic": ic(fold.test_idx)})
    df = pd.DataFrame(rows)
    df["gap"] = df["train_ic"] - df["test_ic"]
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--config", default=None)
    p.add_argument("--what", nargs="+", default=["permute", "traingap"],
                   choices=["permute", "traingap"])
    p.add_argument("--model", default="lgbm", choices=["lgbm", "deep"],
                   help="permute: lgbm is much faster and tests the same "
                        "feature/split machinery")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    fs = build_features(load_raw(args.data), cfg.data)
    print(f"{len(fs)} deals\n")

    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if "permute" in args.what:
        print(f"=== Permutation test ({args.model}, {args.repeats} shuffles) ===")
        df = permutation_test(cfg, fs, args.model, args.repeats, args.seed)
        df.to_csv(out / "sanity_permutation.csv", index=False)

    if "traingap" in args.what:
        print("\n=== Train vs test IC (LightGBM) ===")
        df = train_gap(cfg, fs, args.seed)
        print(df.round(4).to_string(index=False))
        print(f"\n  mean gap {df['gap'].mean():+.4f} — a large gap means the "
              "model memorizes its training data.\n  That is normal and does "
              "NOT invalidate the test score; it only suggests more "
              "regularization\n  might buy a little generalization.")
        df.to_csv(out / "sanity_train_gap.csv", index=False)

    print(f"\nWrote sanity CSVs to {out}/")


if __name__ == "__main__":
    main()
