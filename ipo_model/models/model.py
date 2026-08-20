"""Three-arm IPO return model.

  STATIC ARM     bookrunner multi-hot -> bag-of-embeddings (mean), sector
                 binaries, market one-hot, bookrunner count -> small MLP
                 -> z_static
  MOMENTUM ARM   the engineered market-state block (F1 recent-deal
                 performance, F2 break rate/depth, F3 rolling supply,
                 macro state, subgroup peers) -> small MLP -> z_mom
  GPR ARM        "level" (last value), "engineered" (5 summaries -> MLP),
                 or "lstm" (GRU over the daily window) -> z_gpr
  GATING         optional FiLM: z_gpr modulates BOTH z_static and z_mom
                 (scale + shift, zero-initialized so training starts at
                 identity) — "the geopolitical regime decides how much deal
                 quality and market momentum matter".
  FUSION         concat -> MLP -> shared representation -> per-horizon
                 heads (1d / 3d / 1w / 1m):
                   head="binary"   one logit per horizon — P(outperform the
                                   benchmark). The default: magnitude noise
                                   and moonshots never enter the loss.
                   head="quantile" median + softplus offsets -> non-crossing
                                   (q10, q50, q90), pinball loss.
  The main horizon's output is the forecast; shorter horizons regularize.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ipo_model.config import ModelConfig


def mlp(dims: list[int], dropout: float, out_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_act:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class StaticArm(nn.Module):
    def __init__(self, n_binary: int, n_bookrunners: int, cfg: ModelConfig):
        super().__init__()
        self.has_bk = n_bookrunners > 0
        emb_dim = cfg.bookrunner_emb_dim if self.has_bk else 0
        if self.has_bk:
            self.bk_embed = nn.Linear(n_bookrunners, emb_dim, bias=False)
        # Input dropout: at train time, sector flags / syndicate extras /
        # individual banks in the multi-hot randomly vanish, so the model
        # cannot memorize "this flag or this bank => big pop" shortcuts.
        self.in_drop = nn.Dropout(cfg.static_input_dropout)
        self.net = mlp([n_binary + emb_dim, cfg.static_hidden, cfg.static_out],
                       cfg.dropout, out_act=True)

    def forward(self, binary: torch.Tensor, bk: torch.Tensor) -> torch.Tensor:
        parts = [self.in_drop(binary)]
        if self.has_bk:
            # Mean-of-embeddings over the deal's bookrunners (multi-hot bag).
            bk = self.in_drop(bk)
            emb = self.bk_embed(bk) / bk.sum(dim=1, keepdim=True).clamp(min=1.0)
            parts.append(emb)
        return self.net(torch.cat(parts, dim=1))


class MomentumArm(nn.Module):
    """Encodes the engineered market-state factor block."""

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.net = mlp([n_features, cfg.momentum_hidden, cfg.momentum_out],
                       cfg.dropout, out_act=True)

    def forward(self, momentum: torch.Tensor) -> torch.Tensor:
        return self.net(momentum)


class GPRArm(nn.Module):
    def __init__(self, cfg: ModelConfig, n_engineered: int):
        super().__init__()
        self.mode = cfg.gpr_mode
        self.input_mode = cfg.gpr_input
        if self.input_mode not in ("level", "change"):
            raise ValueError(f"Unknown gpr_input: {self.input_mode}")
        if self.mode == "level":
            self.net = nn.Linear(1, cfg.gpr_out)
        elif self.mode == "engineered":
            self.net = mlp([n_engineered, cfg.gpr_hidden, cfg.gpr_out],
                           cfg.dropout, out_act=True)
        elif self.mode == "lstm":
            self.gru = nn.GRU(1, cfg.gpr_hidden, batch_first=True)
            self.head = nn.Linear(cfg.gpr_hidden, cfg.gpr_out)
        else:
            raise ValueError(f"Unknown gpr_mode: {self.mode}")

    def forward(self, seq: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if self.mode == "level":
            return self.net(seq[:, -1:])
        if self.mode == "engineered":
            return self.net(feats)
        if self.input_mode == "change":  # day-over-day risk shocks, not height
            seq = seq[:, 1:] - seq[:, :-1]
        out, _ = self.gru(seq.unsqueeze(2))
        return self.head(out[:, -1])


class FiLM(nn.Module):
    """Feature-wise linear modulation, zero-initialized (identity at start)."""

    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.net = nn.Linear(cond_dim, 2 * target_dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)
        return z * (1.0 + gamma) + beta


class QuantileHead(nn.Module):
    """Median + non-negative offsets -> monotone (q10, q50, q90)."""

    def __init__(self, in_dim: int, quantiles: tuple[float, ...]):
        super().__init__()
        self.quantiles = sorted(quantiles)
        if 0.5 not in self.quantiles:
            raise ValueError("Quantile set must include the median (0.5).")
        self.mid = self.quantiles.index(0.5)
        self.median = nn.Linear(in_dim, 1)
        self.lo = nn.Linear(in_dim, self.mid)
        self.hi = nn.Linear(in_dim, len(self.quantiles) - self.mid - 1)

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        med = self.median(rep)
        lo = med - torch.cumsum(nn.functional.softplus(self.lo(rep)), dim=1).flip(1)
        hi = med + torch.cumsum(nn.functional.softplus(self.hi(rep)), dim=1)
        return torch.cat([lo, med, hi], dim=1)  # (B, n_quantiles), ascending


class ThreeArmModel(nn.Module):
    def __init__(self, cfg: ModelConfig, n_binary: int, n_bookrunners: int,
                 n_momentum: int, n_gpr_feats: int, horizons: tuple[int, ...]):
        super().__init__()
        self.cfg = cfg
        self.horizons = horizons
        self.static_arm = StaticArm(n_binary, n_bookrunners, cfg)
        fusion_in = cfg.static_out
        if cfg.use_momentum:
            self.momentum_arm = MomentumArm(n_momentum, cfg)
            fusion_in += cfg.momentum_out
        if cfg.use_gpr:
            self.gpr_arm = GPRArm(cfg, n_gpr_feats)
            fusion_in += cfg.gpr_out
        if cfg.gating == "film":
            if not cfg.use_gpr:
                raise ValueError("FiLM gating requires the GPR arm.")
            self.film_static = FiLM(cfg.gpr_out, cfg.static_out)
            if cfg.use_momentum:
                self.film_momentum = FiLM(cfg.gpr_out, cfg.momentum_out)
        elif cfg.gating != "none":
            raise ValueError(f"Unknown gating: {cfg.gating}")

        dims = [fusion_in, *cfg.fusion_hidden]
        self.fusion = mlp(dims, cfg.dropout, out_act=True)
        if cfg.head == "binary":
            # One logit per horizon: P(outperform). (B, 1) per head.
            self.heads = nn.ModuleDict({
                str(h): nn.Linear(dims[-1], 1) for h in horizons
            })
        elif cfg.head == "quantile":
            self.heads = nn.ModuleDict({
                str(h): QuantileHead(dims[-1], cfg.quantiles) for h in horizons
            })
        else:
            raise ValueError(f"Unknown head: {cfg.head!r} "
                             "(expected 'binary' or 'quantile')")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        z_static = self.static_arm(batch["static_binary"], batch["static_bk"])
        parts = [z_static]
        z_gpr = None
        if self.cfg.use_gpr:
            z_gpr = self.gpr_arm(batch["gpr_seq"], batch["gpr_feats"])
        if self.cfg.gating == "film":
            parts[0] = self.film_static(z_static, z_gpr)
        if self.cfg.use_momentum:
            z_mom = self.momentum_arm(batch["momentum"])
            if self.cfg.gating == "film":
                z_mom = self.film_momentum(z_mom, z_gpr)
            parts.append(z_mom)
        if z_gpr is not None:
            parts.append(z_gpr)
        rep = self.fusion(torch.cat(parts, dim=1))
        return {h: self.heads[str(h)](rep) for h in self.horizons}

    # ---- group-lasso input sparsity ----

    def _input_layers(self) -> list[nn.Linear]:
        layers = [self.static_arm.net[0]]
        if self.cfg.use_momentum:
            layers.append(self.momentum_arm.net[0])
        if self.cfg.use_gpr and self.cfg.gpr_mode == "engineered":
            layers.append(self.gpr_arm.net[0])
        return layers

    def input_l1(self) -> torch.Tensor:
        """Group-lasso penalty: the L2 norm of each input feature's first-layer
        weight column, summed. Drives whole columns to zero — i.e. actually
        removes features, which dropout and weight decay never do."""
        pen = torch.zeros((), device=next(self.parameters()).device)
        for lin in self._input_layers():
            pen = pen + lin.weight.norm(dim=0).sum()
        return pen

    def static_l1(self) -> torch.Tensor:
        """Group-lasso restricted to the static block: the static arm's
        first-layer input columns (sector flags, n_banks, syndicate extras)
        plus each BANK's embedding column. Under this penalty a bank or flag
        that doesn't reduce loss is zeroed outright — targeted containment of
        sector/bookrunner overfitting without removing the inputs."""
        pen = self.static_arm.net[0].weight.norm(dim=0).sum()
        if self.static_arm.has_bk:
            pen = pen + self.static_arm.bk_embed.weight.norm(dim=0).sum()
        return pen

    def bookrunner_embed_norms(self) -> "np.ndarray | None":
        """Per-bank embedding column norms (for prune reports)."""
        if not self.static_arm.has_bk:
            return None
        import numpy as np
        return (self.static_arm.bk_embed.weight.norm(dim=0)
                .detach().cpu().numpy())

    def momentum_input_norms(self) -> "np.ndarray | None":
        """Per-momentum-feature first-layer column norms (for prune reports)."""
        if not self.cfg.use_momentum:
            return None
        import numpy as np
        return self.momentum_arm.net[0].weight.norm(dim=0).detach().cpu().numpy()
