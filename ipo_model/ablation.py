"""The ablation ladder.

Each rung is a named set of config overrides on the same data, folds, seeds
and metrics, so differences between rungs are attributable to the component
being toggled:

  00 naive_const    predict the training median — no ordering at all, so it
                    floors the distribution metrics
  00 naive_f1       one-factor regression on recent same-market IPO
                    performance — the floor that actually matters
  0  lgbm / xgb     trees on the engineered features (two engines as a
                    robustness check on the baseline)
  0  *_tuned        the same trees with hyperparameters random-searched on
                    the validation slice. THIS is the honest bar: beating a
                    library-defaults baseline proves nothing
  0r *_raw          trees on engineered features + the raw GPR window
  1  static         static arm only
  2  static+f1      + F1 only
  3  static+moment. + the full F1-F4 factor block
  4  +gpr_level     + the GPR level
  5  full_v1        + the GPR temporal encoder
  6  full_v2_film   + FiLM gating
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ipo_model.baselines import lgbm, naive, xgb
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.results_io import save_predictions
from ipo_model.training import loop
from ipo_model.training.loop import RunResult
from ipo_model.training.metrics import ic_by_period, t_test

RUNGS: dict[str, dict[str, Any]] = {
    "1_static": {"model.use_momentum": False, "model.use_gpr": False},
    "2_static+f1": {"model.use_momentum": True, "model.momentum_groups": ("f1",),
                    "model.use_gpr": False},
    "3_static+momentum": {"model.use_momentum": True, "model.use_gpr": False},
    "4_momentum+gpr_level": {"model.use_momentum": True, "model.use_gpr": True,
                             "model.gpr_mode": "level"},
    "5_full_v1": {"model.use_momentum": True, "model.use_gpr": True,
                  "model.gpr_mode": "lstm", "model.gating": "none"},
    "6_full_v2_film": {"model.use_momentum": True, "model.use_gpr": True,
                       "model.gpr_mode": "lstm", "model.gating": "film"},
}

# Optional add-on rungs (ipo_model/extras). NOT in the default ladder — they
# need extra dependencies (requirements-extras.txt). Request explicitly:
#     python scripts/run_ablations.py --rungs 0_lgbm x_ridge x_tabpfn 5_full_v1
EXTRA_RUNGS: dict[str, tuple[str, dict[str, Any]]] = {
    "x_ridge": ("linear", {"model": "ridge"}),
    "x_lasso": ("linear", {"model": "lasso"}),
    "x_elasticnet": ("linear", {"model": "elasticnet"}),
    "x_tabpfn": ("tabpfn", {}),
}

REPORT_METRICS = ["rank_ic", "hit_rate", "decile_spread", "mae", "pinball", "coverage"]


def run_ladder(cfg: Config, fs: FeatureSet, rungs: list[str] | None = None,
               out_dir: str | Path = "results", verbose: bool = True) -> pd.DataFrame:
    tree_rungs = {
        "0_lgbm": (lgbm, "engineered"),
        "0_xgb": (xgb, "engineered"),
        "0_lgbm_raw": (lgbm, "raw"),
        "0_xgb_raw": (xgb, "raw"),
    }
    # Tuned variants: a baseline on library defaults is not a fair
    # comparison. These search hyperparameters on the validation slice.
    tuned_rungs = {"0_lgbm_tuned": (lgbm, 40), "0_xgb_tuned": (xgb, 40)}
    naive_rungs = {
        "00_naive_const": {"strategy": "constant"},
        "00_naive_f1": {"strategy": "feature", "feature": naive.DEFAULT_FEATURE},
    }
    names = (rungs if rungs is not None
             else [*naive_rungs, *tree_rungs, *RUNGS.keys()])
    rows = []
    for name in names:
        if verbose:
            print(f"\n=== {name} ===")
        if name in naive_rungs:
            res = naive.run(cfg, fs, verbose=verbose, **naive_rungs[name])
        elif name in tree_rungs:
            engine, features = tree_rungs[name]
            res = engine.run(cfg, fs, features=features, verbose=verbose)
        elif name in tuned_rungs:
            engine, n_cfg = tuned_rungs[name]
            res = engine.run(cfg, fs, tune=n_cfg, verbose=verbose)
        elif name in EXTRA_RUNGS:
            # Optional add-ons: imported here so a missing dependency can
            # never break the default ladder.
            kind, kwargs = EXTRA_RUNGS[name]
            if kind == "linear":
                from ipo_model.extras import linear as _mod
            else:
                from ipo_model.extras import tabpfn_model as _mod
            res = _mod.run(cfg, fs, verbose=verbose, **kwargs)
        else:
            res = loop.run(cfg.override(**RUNGS[name]), fs, verbose=verbose)
        row: dict[str, Any] = {"rung": name}
        for m in REPORT_METRICS:
            mean, std = res.summary[m]
            row[m] = mean
            row[f"{m}_std"] = std
            row[f"{m}_pooled"] = res.pooled[m]
        # Significance of the pooled IC, from the period-by-period IC series.
        row.update(_ic_significance(cfg, fs, res))
        rows.append(row)

        # Always persist per-deal predictions: every later analysis (paired
        # tests, bootstrap CIs, new metrics) then runs without retraining.
        save_predictions(cfg, fs, res, name, out_dir=out_dir)

    df = pd.DataFrame(rows).set_index("rung")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablations.csv")
    _write_markdown(df, out / "ablations.md")
    if verbose:
        print(f"\nPer-deal predictions saved to {out}/predictions/ — analyse with:"
              f"\n  python scripts/analyze_results.py --results {out}")
    return df


def _ic_significance(cfg: Config, fs: FeatureSet, res: RunResult) -> dict[str, float]:
    """t-test of the quarterly IC series behind this rung's pooled IC."""
    idx = np.concatenate([r.test_idx for r in res.fold_results])
    q = np.concatenate([r.q_pred for r in res.fold_results])
    y = fs.y[cfg.main_horizon][idx]
    per = ic_by_period(y, q, fs.dates[idx], cfg.model.quantiles, freq="QE")
    if len(per) < 2:
        return {"ic_periods": float(len(per)), "ic_t": float("nan"),
                "ic_p": float("nan")}
    t = t_test(per["rank_ic"].to_numpy(), hac=True)
    return {"ic_periods": t["n"], "ic_t": t["t"], "ic_p": t["p"]}


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Ablation results", "",
             "Mean ± std across purged walk-forward folds "
             "(pooled OOS value in parentheses).", ""]
    header = "| rung | " + " | ".join(REPORT_METRICS) + " |"
    lines += [header, "|" + "---|" * (len(REPORT_METRICS) + 1)]
    for rung, r in df.iterrows():
        cells = [f"{r[m]:+.4f} ± {r[f'{m}_std']:.4f} ({r[f'{m}_pooled']:+.4f})"
                 for m in REPORT_METRICS]
        lines.append(f"| {rung} | " + " | ".join(cells) + " |")

    if "ic_t" in df.columns:
        lines += ["", "## Is the rank IC different from zero?", "",
                  "t-test of the quarterly IC series (Newey-West standard "
                  "errors). Consistency across periods is the evidence — a "
                  "single pooled IC has no error bar.", "",
                  "| rung | quarters | t | p |", "|---|---|---|---|"]
        for rung, r in df.iterrows():
            lines.append(f"| {rung} | {r['ic_periods']:.0f} | "
                         f"{r['ic_t']:+.2f} | {r['ic_p']:.4f} |")

    lines += ["", "---", "",
              "Per-deal predictions are in `predictions/`. For paired tests "
              "between rungs, bootstrap intervals and multiple-comparison "
              "correction, run `scripts/analyze_results.py`."]
    path.write_text("\n".join(lines) + "\n")
