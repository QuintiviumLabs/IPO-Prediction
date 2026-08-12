"""Configuration for the IPO return prediction framework.

All knobs live in one dataclass tree so an ablation rung is just a set of
overrides applied to the default config.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    # Target definition: log return from offer price to the close on the
    # h-th trading day (h=1 is the first close), in excess of the market
    # index over the same span (market_adjust=True). 1d / 3d / 1w / 1m.
    horizons: tuple[int, ...] = (1, 3, 5, 21)   # trading days; last is the main target
    market_adjust: bool = True
    # ---- Momentum factor block (F1-F4), all same-market unless noted ----
    momentum_max_deals: int = 10        # F1/F2: up to this many most recent IPOs...
    momentum_window_days: int = 90      # ...within this many calendar days
    momentum_extended: bool = True      # add deal count + IQR to the F1 spec
    momentum_horizons: tuple[int, ...] = (1, 5)   # F1/F2 measured at 1d and 1w
    supply_window_days: int = 90        # F3/F4 rolling window (~3 months)
    supply_trailing_days: int = 365     # F3_accel: trailing pace reference
    sector_30d_days: int = 30           # F4_sec_cnt_30d window (all markets)
    # GPR window (daily Caldara-Iacoviello GPR index; see scripts/prepare_gpr.py)
    gpr_window: int = 21                # trading days of GPR history for the encoder
    # Bookrunner columns are every column in ipos.csv prefixed with this.
    bookrunner_prefix: str = "bk_"
    # Binary/static feature columns (besides bookrunners and market one-hot).
    sector_cols: tuple[str, ...] = ("is_tmt", "is_healthcare")
    # Bookrunner columns rarer than this many deals in the *training* fold are
    # merged into a shared "other" bucket by the fold scaler.
    min_bookrunner_deals: int = 20


@dataclass
class SplitConfig:
    n_folds: int = 5
    # Fraction of the sample (earliest rows) reserved as the minimum training
    # window before the first test block.
    min_train_frac: float = 0.35
    # Fraction of each training window (latest rows) carved out for early
    # stopping / LightGBM early stopping.
    val_frac: float = 0.15
    # Extra calendar-day embargo beyond the label window when purging.
    embargo_days: int = 5


@dataclass
class ModelConfig:
    # Arms on/off — this is what the ablation ladder toggles.
    use_momentum: bool = True
    momentum_groups: tuple[str, ...] = ("f1", "f2", "f3", "f4")
    use_gpr: bool = True
    gpr_mode: str = "lstm"          # "level" | "engineered" | "lstm"
    gating: str = "none"            # "none" | "film" (GPR modulates static & momentum)
    # Sizes (kept deliberately small for N ~ 1.5k).
    bookrunner_emb_dim: int = 4
    static_hidden: int = 16
    static_out: int = 8
    momentum_hidden: int = 16
    momentum_out: int = 8
    gpr_hidden: int = 8
    gpr_out: int = 4
    fusion_hidden: tuple[int, ...] = (16, 8)
    dropout: float = 0.25
    # Quantile heads (pinball loss). Median head doubles as the point forecast.
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    # Multi-task loss weights, keyed by horizon; the main horizon gets 1.0.
    aux_weight: float = 0.3


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    device: str = "cpu"


@dataclass
class LGBMConfig:
    params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "huber",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
    })
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50


@dataclass
class XGBConfig:
    params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:pseudohubererror",
        "huber_slope": 1.0,
        "learning_rate": 0.05,
        "max_leaves": 31,
        "grow_policy": "lossguide",
        "tree_method": "hist",
        "min_child_weight": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
        "verbosity": 0,
    })
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)

    @property
    def main_horizon(self) -> int:
        return self.data.horizons[-1]

    def override(self, **kwargs: Any) -> "Config":
        """Return a deep copy with dotted-key overrides, e.g.
        cfg.override(**{"model.use_momentum": False, "model.gpr_mode": "level"}).
        """
        cfg = _deepcopy_dc(self)
        for key, value in kwargs.items():
            obj = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if not hasattr(obj, parts[-1]):
                raise KeyError(f"Unknown config key: {key}")
            setattr(obj, parts[-1], value)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        flat = _flatten(raw)
        return cfg.override(**flat)


def _deepcopy_dc(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return type(obj)(**{
            f.name: _deepcopy_dc(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        })
    if isinstance(obj, dict):
        return {k: _deepcopy_dc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_deepcopy_dc(v) for v in obj)
    return obj


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key not in ("lgbm.params", "xgb.params"):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = tuple(v) if isinstance(v, list) else v
    return out
