"""Per-fold preprocessing.

Everything here is fit on the training indices of a fold only, then applied
to train/val/test alike — no test-set statistics ever leak into scaling or
bookrunner bucketing. (F3 and parts of the macro block arrive already
expanding-z-scored at feature build time, which uses only each row's past;
the fold scaler standardizes the momentum block again on the training
window, which is harmless and keeps F1/F2 on a comparable scale.)

- Bookrunner bucketing: bookrunner columns with fewer than
  `min_bookrunner_deals` deals in the training window are collapsed into a
  single "bk_other" column.
- Standardization: GPR sequences/features, the momentum factor block, the
  bookrunner count, and the continuous syndicate extras (n_banks,
  log_n_banks, prestige_rank_max) use train-window statistics. Binary
  extras and archetype one-hots pass through unscaled and join the static
  binary block, so the model tensors keep the same keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ipo_model.config import DataConfig
from ipo_model.data.features import FeatureSet, GPR_FEAT_NAMES


@dataclass
class FoldScaler:
    keep_bk: np.ndarray = field(default=None)   # column indices kept as-is
    rare_bk: np.ndarray = field(default=None)   # column indices pooled into "other"
    bk_out_names: list[str] = field(default_factory=list)
    gpr_mu: float = 0.0
    gpr_sd: float = 1.0
    gpr_feat_mu: np.ndarray = field(default=None)
    gpr_feat_sd: np.ndarray = field(default=None)
    mom_mu: np.ndarray = field(default=None)
    mom_sd: np.ndarray = field(default=None)
    n_bk_mu: float = 0.0
    n_bk_sd: float = 1.0
    extra_mu: np.ndarray = field(default=None)   # over continuous extras only
    extra_sd: np.ndarray = field(default=None)
    extra_continuous: np.ndarray = field(default=None)  # bool mask, all extras
    extra_names: list[str] = field(default_factory=list)
    include_market_onehot: bool = False
    feature_clip: float = 0.0

    @classmethod
    def fit(cls, fs: FeatureSet, train_idx: np.ndarray, cfg: DataConfig) -> "FoldScaler":
        s = cls()
        s.include_market_onehot = cfg.include_market_onehot
        s.feature_clip = cfg.feature_clip
        counts = fs.bk[train_idx].sum(axis=0)
        s.keep_bk = np.where(counts >= cfg.min_bookrunner_deals)[0]
        s.rare_bk = np.where(counts < cfg.min_bookrunner_deals)[0]
        s.bk_out_names = [fs.bk_names[i] for i in s.keep_bk]
        if len(s.rare_bk):
            s.bk_out_names.append("bk_other")

        g = fs.gpr_seq[train_idx]
        s.gpr_mu, s.gpr_sd = float(g.mean()), float(g.std() + 1e-8)
        gf = fs.gpr_feats[train_idx]
        s.gpr_feat_mu = gf.mean(axis=0)
        s.gpr_feat_sd = gf.std(axis=0) + 1e-8

        mom = fs.momentum[train_idx]
        s.mom_mu = mom.mean(axis=0)
        s.mom_sd = mom.std(axis=0) + 1e-8

        s.n_bk_mu = float(fs.n_bk[train_idx].mean())
        s.n_bk_sd = float(fs.n_bk[train_idx].std() + 1e-8)

        s.extra_names = list(fs.static_extra_names)
        s.extra_continuous = np.asarray(fs.static_extra_continuous, dtype=bool)
        if s.extra_continuous.any():
            cont = fs.static_extra[train_idx][:, s.extra_continuous]
            s.extra_mu = cont.mean(axis=0)
            s.extra_sd = cont.std(axis=0) + 1e-8
        return s

    # ---- transformed views (full-sample arrays; index with fold indices) ----

    def bookrunners(self, fs: FeatureSet) -> np.ndarray:
        cols = [fs.bk[:, self.keep_bk]]
        if len(self.rare_bk):
            cols.append((fs.bk[:, self.rare_bk].sum(axis=1, keepdims=True) > 0).astype(float))
        return np.concatenate(cols, axis=1) if cols[0].shape[1] or len(cols) > 1 \
            else np.zeros((len(fs), 0))

    def static_extras(self, fs: FeatureSet) -> np.ndarray:
        """Syndicate extras with the continuous ones standardized (and
        clipped) on train stats; binaries/one-hots pass through. Column order
        matches fs.static_extra_names."""
        if fs.static_extra.shape[1] == 0:
            return np.zeros((len(fs), 0), dtype=np.float32)
        out = fs.static_extra.astype(np.float64).copy()
        if self.extra_continuous is not None and self.extra_continuous.any():
            cont = (out[:, self.extra_continuous] - self.extra_mu) / self.extra_sd
            out[:, self.extra_continuous] = self._clip(cont)
        return out.astype(np.float32)

    def static_matrix(self, fs: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
        """(binary block: sector [+ market one-hot if enabled] + n_bk +
        syndicate extras, bookrunner multi-hot). Market identity is excluded
        by default so the model predicts general outperformance, not market
        membership. The extras ride in the static binary block so the model
        and training tensors keep their existing keys and shapes."""
        n_bk = ((fs.n_bk - self.n_bk_mu) / self.n_bk_sd)[:, None]
        parts = [fs.sector]
        if self.include_market_onehot:
            parts.append(fs.market_onehot)
        parts.append(n_bk)
        extras = self.static_extras(fs)
        if extras.shape[1]:
            parts.append(extras)
        binary = np.concatenate(parts, axis=1)
        return binary.astype(np.float32), self.bookrunners(fs).astype(np.float32)

    def _clip(self, z: np.ndarray) -> np.ndarray:
        if self.feature_clip > 0:
            return np.clip(z, -self.feature_clip, self.feature_clip)
        return z

    def momentum_matrix(self, fs: FeatureSet) -> np.ndarray:
        return self._clip((fs.momentum - self.mom_mu) / self.mom_sd).astype(np.float32)

    def gpr_sequences(self, fs: FeatureSet) -> np.ndarray:
        return self._clip((fs.gpr_seq - self.gpr_mu) / self.gpr_sd).astype(np.float32)

    def gpr_features(self, fs: FeatureSet) -> np.ndarray:
        return self._clip((fs.gpr_feats - self.gpr_feat_mu)
                          / self.gpr_feat_sd).astype(np.float32)


def engineered_table(fs: FeatureSet, scaler: FoldScaler) -> pd.DataFrame:
    """Flat feature table for the tree baselines (ablation 0).

    The momentum block IS engineered (F1-F4), so the trees and the deep model
    see the same market-state information; the deep model's edge, if any, must
    come from the GPR sequence encoder and learned interactions.
    """
    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(fs.sector_names):
        cols[name] = fs.sector[:, i]
    if scaler.include_market_onehot:
        for i, name in enumerate(fs.market_names):
            cols[f"mkt_{name}"] = fs.market_onehot[:, i]
    bk = scaler.bookrunners(fs)
    for i, name in enumerate(scaler.bk_out_names):
        cols[name] = bk[:, i]
    cols["n_bookrunners"] = fs.n_bk
    for i, name in enumerate(fs.static_extra_names):
        cols[name] = fs.static_extra[:, i]      # raw values; trees don't care
    for i, name in enumerate(fs.momentum_names):
        cols[name] = fs.momentum[:, i]
    for i, name in enumerate(GPR_FEAT_NAMES):
        cols[name] = fs.gpr_feats[:, i]
    return pd.DataFrame(cols)


def raw_table(fs: FeatureSet, scaler: FoldScaler) -> pd.DataFrame:
    """Engineered table plus the raw GPR window as individual columns —
    the "trees on full inputs" ablation. (The momentum block is engineered by
    definition now; only GPR still has an unsummarized sequence form.)"""
    base = engineered_table(fs, scaler)
    W = fs.gpr_seq.shape[1]
    raw = {f"gpr_m{W - t:02d}": fs.gpr_seq[:, t] for t in range(W)}  # m01 = latest
    return pd.concat([base, pd.DataFrame(raw, index=base.index)], axis=1)
