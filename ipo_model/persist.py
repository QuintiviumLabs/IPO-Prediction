"""Model persistence: save a trained bundle once, score new IPOs forever.

A bundle is a directory:

    bundle/
      meta.json        config + feature names + dims + provenance
      scaler.pkl       FoldScaler fitted on the final training window
      transform.pkl    per-horizon NormalScore transforms (or None)
      seed_<s>.pt      one state_dict per training seed (the ensemble)

`fit_final` trains on ALL labeled data (with a validation tail for early
stopping) and writes the bundle. `load_bundle(...).predict(fs)` rebuilds the
models, ensembles the seeds, back-maps through the label transform and
returns quantile forecasts per horizon — same arithmetic as the walk-forward
loop, minus the folds.

Inference on a brand-new IPO: append its row to ipos.csv (offer_price,
market, sector flags, bk_* columns, syndicate extras) — no price rows needed
— refresh peers.csv/macro csvs, then

    python scripts/predict.py --data data/ --bundle results/final_model --new-only
"""
from __future__ import annotations

import dataclasses
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ipo_model.config import Config
from ipo_model.data.features import FeatureSet
from ipo_model.data.preprocess import FoldScaler
from ipo_model.models.model import ThreeArmModel
from ipo_model.training.loop import _build_model, _slice, _tensors, train_one
from ipo_model.training.transforms import NormalScore

FORMAT_VERSION = 1


def _cfg_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def _cfg_from_dict(d: dict) -> Config:
    cfg = Config()
    flat: dict = {}

    def walk(prefix: str, obj: dict) -> None:
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict) and key not in ("lgbm.params", "xgb.params"):
                walk(f"{key}.", v)
            else:
                flat[key] = tuple(v) if isinstance(v, list) else v

    walk("", d)
    return cfg.override(**flat)


@dataclass
class Bundle:
    cfg: Config
    scaler: FoldScaler
    transform: dict[int, NormalScore] | None
    states: list[dict]                 # one state_dict per seed
    meta: dict

    def _models(self, tensors: dict[str, torch.Tensor]) -> list[ThreeArmModel]:
        models = []
        for st in self.states:
            m = _build_model(self.cfg, tensors).to(self.cfg.train.device)
            m.load_state_dict(st)
            m.eval()
            models.append(m)
        return models

    def predict(self, fs: FeatureSet,
                idx: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Ensemble forecasts per horizon.

        head='quantile': (n, Q) quantile matrix in return space.
        head='binary':   (n, 1) probability of outperforming the benchmark.

        The FeatureSet must be built from the SAME data schema the bundle was
        trained on (checked against stored feature names)."""
        self._check_schema(fs)
        binary = self.cfg.model.head == "binary"
        device = self.cfg.train.device
        tensors = _tensors(fs, self.scaler, self.cfg, device)
        if idx is not None:
            tensors = _slice(tensors, np.asarray(idx))
        out: dict[int, np.ndarray] = {}
        with torch.no_grad():
            per_model = [m(tensors) for m in self._models(tensors)]
        for h in sorted(self.cfg.data.horizons):
            if binary:
                q = np.mean([torch.sigmoid(p[h]).cpu().numpy()
                             for p in per_model], axis=0)
            else:
                q = np.mean([p[h].cpu().numpy() for p in per_model], axis=0)
            if q.ndim == 1:
                q = q[:, None]
            if self.transform is not None and not binary:
                q = self.transform[h].inverse(q)
            out[h] = q
        return out

    def _check_schema(self, fs: FeatureSet) -> None:
        problems = []
        if list(fs.momentum_names) != self.meta["momentum_names"]:
            problems.append(
                f"momentum block changed: bundle has "
                f"{len(self.meta['momentum_names'])} features, data builds "
                f"{len(fs.momentum_names)} (peers/macro csvs present at "
                f"training time must be present at inference too)")
        if list(fs.bk_names) != self.meta["bk_names"]:
            problems.append("bookrunner columns differ from training")
        if list(fs.static_extra_names) != self.meta["static_extra_names"]:
            problems.append("syndicate-extra columns differ from training")
        if list(fs.sector_names) != self.meta["sector_names"]:
            problems.append("sector columns differ from training")
        if problems:
            raise ValueError("FeatureSet does not match the bundle's schema:\n- "
                             + "\n- ".join(problems))


def save_bundle(path: str | Path, cfg: Config, scaler: FoldScaler,
                states: list[dict], fs: FeatureSet,
                transform: dict[int, NormalScore] | None,
                extra_meta: dict | None = None) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": FORMAT_VERSION,
        "config": _cfg_to_dict(cfg),
        "seeds": list(cfg.train.seeds),
        "momentum_names": list(fs.momentum_names),
        "bk_names": list(fs.bk_names),
        "sector_names": list(fs.sector_names),
        "market_names": list(fs.market_names),
        "static_extra_names": list(fs.static_extra_names),
        "n_train_rows": int(len(fs)),
        "train_date_range": [str(np.datetime_as_string(fs.dates.min(), unit="D")),
                             str(np.datetime_as_string(fs.dates.max(), unit="D"))],
        "quantiles": list(cfg.model.quantiles),
        "horizons": list(cfg.data.horizons),
    }
    meta.update(extra_meta or {})
    (p / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(p / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(p / "transform.pkl", "wb") as f:
        pickle.dump(transform, f)
    for s, st in zip(cfg.train.seeds, states):
        torch.save(st, p / f"seed_{s}.pt")
    return p


def load_bundle(path: str | Path) -> Bundle:
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text())
    if meta.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(f"Bundle {p} was written by a newer code version.")
    cfg = _cfg_from_dict(meta["config"])
    with open(p / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(p / "transform.pkl", "rb") as f:
        transform = pickle.load(f)
    states = [torch.load(p / f"seed_{s}.pt", map_location=cfg.train.device,
                         weights_only=True)
              for s in meta["seeds"]]
    return Bundle(cfg=cfg, scaler=scaler, transform=transform,
                  states=states, meta=meta)


def fit_final(cfg: Config, fs: FeatureSet, out: str | Path,
              verbose: bool = True) -> Bundle:
    """Train on ALL labeled rows (latest val_frac as the early-stopping tail)
    and save the bundle. This is the model you actually deploy; the
    walk-forward loop is how you decided to trust it."""
    from ipo_model.data.splits import Fold

    n = len(fs)
    n_val = max(int(round(n * cfg.split.val_frac)), 20)
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)
    fold = Fold(train_idx=train_idx, val_idx=val_idx,
                test_idx=np.array([], dtype=int),
                test_start=fs.dates[-1])

    scaler = FoldScaler.fit(fs, train_idx, cfg.data)
    device = cfg.train.device
    tensors = _tensors(fs, scaler, cfg, device)

    transform = None
    if cfg.model.head == "binary":
        if cfg.train.label_transform != "none":
            raise ValueError("label_transform has no meaning with "
                             "head='binary'; set train.label_transform: none")
        targets = {h: torch.tensor((y > 0).astype(np.float32), device=device)
                   for h, y in fs.y.items()}
    elif cfg.train.label_transform == "normal_score":
        transform = {h: NormalScore().fit(y[train_idx]) for h, y in fs.y.items()}
        targets = {h: torch.tensor(transform[h].transform(y),
                                   dtype=torch.float32, device=device)
                   for h, y in fs.y.items()}
    elif cfg.train.label_transform != "none":
        raise ValueError(f"Unknown label_transform: {cfg.train.label_transform}")
    else:
        targets = {h: torch.tensor(y, dtype=torch.float32, device=device)
                   for h, y in fs.y.items()}

    states, val_losses = [], []
    for seed in cfg.train.seeds:
        model, val_loss = train_one(cfg, tensors, targets, fold, seed)
        states.append({k: v.cpu() for k, v in model.state_dict().items()})
        val_losses.append(val_loss)
        if verbose:
            print(f"  seed {seed}: best val loss {val_loss:.4f}")

    path = save_bundle(out, cfg, scaler, states, fs, transform,
                       extra_meta={"final_val_losses": val_losses,
                                   "n_val_rows": int(n_val)})
    if verbose:
        print(f"  bundle written to {path}")
    return load_bundle(path)
