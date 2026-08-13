"""The ablation ladder.

Each rung is a named set of config overrides on the same data, folds, seeds
and metrics, so differences between rungs are attributable to the component
being toggled:

  0  lgbm / xgb      trees on the engineered features (the bar to clear;
                     two engines as a robustness check on the baseline)
  0r lgbm_raw /      trees on engineered features + the raw GPR window:
     xgb_raw         does unsummarized GPR history help trees?
  1  static          static arm only
  2  static+gpr_lvl  + current GPR level
  3  static+gpr_seq  + GPR temporal encoder (GRU)
  4  static+panel    + recent-IPO temporal encoder (mean pooling)
  5  full_v1         static + panel encoder + GPR encoder
  6  full_v2         + cross-attention pooling + FiLM gating
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ipo_model.baselines import lgbm, xgb
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training import loop

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
    names = rungs if rungs is not None else [*tree_rungs, *RUNGS.keys()]
    rows = []
    for name in names:
        if verbose:
            print(f"\n=== {name} ===")
        if name in tree_rungs:
            engine, features = tree_rungs[name]
            res = engine.run(cfg, fs, features=features, verbose=verbose)
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
        rows.append(row)

    df = pd.DataFrame(rows).set_index("rung")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablations.csv")
    _write_markdown(df, out / "ablations.md")
    return df


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
    path.write_text("\n".join(lines) + "\n")
