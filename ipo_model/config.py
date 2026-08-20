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
    # Target definition: log outperformance vs the IPO's OWN market's
    # benchmark index (e.g. Hang Seng for HK deals) from offer price to the
    # h-th trading day close: y = log((P_h/offer) / (I_h/I_0)), i.e. the log
    # of the outperformance multiple — exp(y) = "grew to X times what the
    # benchmark grew to". 1d / 3d / 1w / 1m. market_adjust=False -> raw.
    horizons: tuple[int, ...] = (1, 3, 5, 21)   # trading days; last is the main target
    market_adjust: bool = True
    # Where the target's return starts.
    #   "offer"       log(close_h / offer_price) — includes the first-day pop,
    #                 which typically dominates the variance at every horizon.
    #   "first_close" log(close_h / close_1) — strips the pop and isolates
    #                 aftermarket drift. Horizon 1 is degenerate under this
    #                 anchor, so use horizons like (3, 5, 21).
    # F1/F2 momentum factors stay offer-anchored either way, per the factor
    # spec — this switch changes the TARGET only.
    anchor: str = "offer"
    # Market-agnostic training: exclude the market one-hot from features so
    # the model predicts general outperformance instead of market identity.
    include_market_onehot: bool = False
    # F1/F2 momentum returns: raw-from-offer per the factor spec, or
    # benchmark-relative like the target (an ablation worth running).
    momentum_market_adjust: bool = False
    # ---- Momentum factor block (F1-F4), all same-market unless noted ----
    momentum_max_deals: int = 10        # F1/F2: up to this many most recent IPOs...
    momentum_window_days: int = 90      # ...within this many calendar days
    momentum_extended: bool = True      # add deal count + IQR to the F1 spec
    momentum_horizons: tuple[int, ...] = (1, 5)   # F1/F2 measured at 1d and 1w
    supply_window_days: int = 90        # F3 rolling window (~3 months)
    supply_trailing_days: int = 365     # F3_accel: trailing pace reference
    # Bookrunner prestige (static extra): a bank's share of same-market IPO
    # proceeds over this trailing window, percentile-ranked among active
    # banks; the deal gets the max over its bookrunners. Expanding and
    # strictly pre-pricing, so leakage-safe by construction.
    prestige_years: int = 3
    # Market regime (macro block): mean first-day pop of the prior
    # `regime_pop_window` same-market IPOs, cut into Hot/Neutral/Cold by
    # EXPANDING terciles of that market's own history. Default definition —
    # replace the cutoffs if your desk has an official one.
    regime_pop_window: int = 10
    # GPR window (daily Caldara-Iacoviello GPR index; see scripts/prepare_gpr.py)
    gpr_window: int = 21                # trading days of GPR history for the encoder
    # Bookrunner columns are every column in ipos.csv prefixed with this.
    bookrunner_prefix: str = "bk_"
    # Binary/static feature columns (besides bookrunners and market one-hot).
    sector_cols: tuple[str, ...] = ("is_tmt", "is_healthcare")
    # Bookrunner columns rarer than this many deals in the *training* fold are
    # merged into a shared "other" bucket by the fold scaler.
    min_bookrunner_deals: int = 20
    # Clip standardized continuous features at +/- this many sigma (0 = off).
    # Mean-type momentum features (e.g. the value-weighted recent-deal
    # return) carry one moonshot into every prediction of the next 90 days;
    # clipping caps a single outlier deal's reach through the features.
    feature_clip: float = 0.0


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
    # Output head.
    #   "binary"    one logit per horizon: P(outperform the benchmark), i.e.
    #               P(y_h > 0). Trained with class-weighted BCE. The ground
    #               truth's MAGNITUDE never enters the loss, so moonshots
    #               cannot dominate and there are far fewer output parameters
    #               to fit — the right shape when the continuous target is
    #               noisy at N ~ 1k.
    #   "quantile"  the original q10/50/90 pinball heads (kept for ablations
    #               and interval work; calibrate_intervals.py needs this).
    head: str = "binary"
    # Anti-overfitting on the static block (sector flags + bookrunner
    # identity stay as INPUTS; these control how hard the model may lean on
    # them):
    #   static_input_dropout  randomly zeroes static inputs (sector flags,
    #                         syndicate extras, individual banks in the
    #                         multi-hot) during training, so no single flag
    #                         or bank can become a memorized shortcut.
    #   l1_static             group-lasso on the static arm's first-layer
    #                         columns AND on each bank's embedding column —
    #                         banks/flags that don't earn their keep are
    #                         driven to exactly zero. Stacks with l1_input.
    static_input_dropout: float = 0.2
    l1_static: float = 1e-3
    # Arms on/off — this is what the ablation ladder toggles.
    use_momentum: bool = True
    # Market-state factor groups fed to the momentum arm:
    #   f1 recent-deal performance   f2 break rate/depth   f3 rolling supply
    #   m  macro block (local index state, vol/FX, global risk, regime)
    #   p1 industry-subgroup peer returns (needs peers.csv)
    momentum_groups: tuple[str, ...] = ("f1", "f2", "f3", "m", "p1")
    use_gpr: bool = True
    gpr_mode: str = "gru"           # "level" | "engineered" | "gru"
    # ("lstm" is accepted as a legacy alias for "gru" — the recurrent
    # encoder has always been an nn.GRU; the old name was a misnomer.)
    # What the gru-mode GPR arm consumes: raw standardized "level"s, or
    # day-over-day "change"s (first differences of the window). Risk shocks
    # are plausibly what matters, not the index's absolute height. Ignored by
    # the "level"/"engineered" modes (engineered already carries d5/dW).
    gpr_input: str = "level"
    gating: str = "none"            # "none" | "film" (GPR modulates static & momentum)
    # Group-lasso strength on first-layer input columns (static + momentum +
    # engineered-GPR). Unlike dropout/weight decay, this actually ZEROES the
    # weights of unhelpful features — structured pruning, reported per fold.
    # 0 disables; ~1e-3 is a sensible starting point at N ~ 1k.
    l1_input: float = 0.0
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
    # "none" | "normal_score": map labels through the training fold's
    # empirical CDF onto Gaussian scores before training, and map predicted
    # quantiles back afterwards. Outlier magnitudes then cannot influence
    # training beyond their rank — the typical (median) deal is never pulled
    # toward the moonshot tail — and back-mapped intervals inherit the
    # empirical tails. Evaluation stays in raw return space either way.
    label_transform: str = "none"
    # Binary head only. "balanced": weight the positive class by n_neg/n_pos
    # of the training fold, so a pop-heavy base rate can't be gamed by always
    # predicting "outperform" — but probabilities then centre on 0.5 instead
    # of the raw base rate (ranking/AUC unaffected; don't read p>0.5 as a
    # literal majority call). "none": raw BCE; probabilities track the base
    # rate but the model may lean lazily toward the majority class.
    class_weight: str = "balanced"
    # Weights & Biases logging (optional; wandb must be installed).
    # Logs per-epoch validation loss, per-fold metrics and the run summary.
    # On a locked-down machine set WANDB_MODE=offline and sync later.
    wandb: bool = False
    wandb_project: str = "ipo-model"
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
