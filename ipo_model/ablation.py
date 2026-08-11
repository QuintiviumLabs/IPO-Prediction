"""The ablation ladder.

Each rung is a named set of config overrides on the same data, folds, seeds
and metrics, so differences between rungs are attributable to the component
being toggled:

  0  lgbm            LightGBM on engineered features (the bar to clear)
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

from ipo_model.baselines import lgbm
from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.training import loop

RUNGS: dict[str, dict[str, Any]] = {
    "1_static": {"model.use_panel": False, "model.use_gpr": False},
    "2_static+gpr_level": {"model.use_panel": False, "model.use_gpr": True,
                           "model.gpr_mode": "level"},
    "3_static+gpr_seq": {"model.use_panel": False, "model.use_gpr": True,
                         "model.gpr_mode": "lstm"},
    "4_static+panel": {"model.use_panel": True, "model.use_gpr": False},
    "5_full_v1": {"model.use_panel": True, "model.use_gpr": True,
                  "model.gpr_mode": "lstm", "model.panel_pooling": "mean",
                  "model.gating": "none"},
    "6_full_v2_attn_film": {"model.use_panel": True, "model.use_gpr": True,
                            "model.gpr_mode": "lstm", "model.panel_pooling": "attn",
                            "model.gating": "film"},
}

REPORT_METRICS = ["rank_ic", "hit_rate", "decile_spread", "mae", "pinball", "coverage"]


def run_ladder(cfg: Config, fs: FeatureSet, rungs: list[str] | None = None,
               out_dir: str | Path = "results", verbose: bool = True) -> pd.DataFrame:
    names = rungs if rungs is not None else ["0_lgbm", *RUNGS.keys()]
    rows = []
    for name in names:
        if verbose:
            print(f"\n=== {name} ===")
        if name == "0_lgbm":
            res = lgbm.run(cfg, fs, verbose=verbose)
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
