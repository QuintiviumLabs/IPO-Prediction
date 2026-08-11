"""Three-arm IPO return model.

  STATIC ARM     bookrunner multi-hot -> bag-of-embeddings (mean), sector
                 binaries, bookrunner count -> small MLP -> z_static
  PANEL ARM      shared GRU over each recent IPO's event-time excess-return
                 sequence (masked, variable length) + per-IPO scalars
                 (age, pop, sector) -> masked mean pooling, or cross-attention
                 with the target's static embedding as the query -> z_panel
  GPR ARM        "level" (last value), "engineered" (5 summaries -> MLP),
                 or "lstm" (GRU over the window) -> z_gpr
  GATING         optional FiLM: z_gpr modulates BOTH z_static and z_panel
                 (scale + shift, zero-initialized so training starts at
                 identity) — "the geopolitical regime decides how much recent
                 momentum and deal quality matter".
  FUSION         concat -> MLP -> shared representation -> per-horizon
                 quantile heads. Each head predicts the median directly and
                 the 10th/90th percentiles as median -/+ softplus offsets, so
                 quantiles can never cross. The median of the main horizon is
                 the point forecast.
"""
from __future__ import annotations

import math

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
        self.net = mlp([n_binary + emb_dim, cfg.static_hidden, cfg.static_out],
                       cfg.dropout, out_act=True)

    def forward(self, binary: torch.Tensor, bk: torch.Tensor) -> torch.Tensor:
        parts = [binary]
        if self.has_bk:
            # Mean-of-embeddings over the deal's bookrunners (multi-hot bag).
            emb = self.bk_embed(bk) / bk.sum(dim=1, keepdim=True).clamp(min=1.0)
            parts.append(emb)
        return self.net(torch.cat(parts, dim=1))


class PanelArm(nn.Module):
    """Shared GRU over K recent-IPO sequences + pooling across IPOs."""

    def __init__(self, n_scalars: int, cfg: ModelConfig, query_dim: int):
        super().__init__()
        self.pooling = cfg.panel_pooling
        self.gru = nn.GRU(1, cfg.panel_hidden, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(cfg.panel_hidden + n_scalars, cfg.panel_out), nn.ReLU(),
        )
        if self.pooling == "attn":
            d = cfg.panel_out
            self.wq = nn.Linear(query_dim, d, bias=False)
            self.wk = nn.Linear(cfg.panel_out, d, bias=False)
            self.scale = math.sqrt(d)
        elif self.pooling != "mean":
            raise ValueError(f"Unknown panel_pooling: {self.pooling}")

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor, scalars: torch.Tensor,
                valid: torch.Tensor, query: torch.Tensor | None) -> torch.Tensor:
        B, K, T = seq.shape
        flat = seq.reshape(B * K, T, 1)
        flat_len = lengths.reshape(B * K)
        out, _ = self.gru(flat)                                   # (B*K, T, H)
        idx = (flat_len - 1).clamp(min=0)
        h = out[torch.arange(B * K, device=seq.device), idx]      # last valid state
        h = h * (flat_len > 0).float().unsqueeze(1)               # zero empty slots
        h = h.reshape(B, K, -1)
        h = self.proj(torch.cat([h, scalars], dim=2))             # (B, K, D)

        mask = valid & (lengths > 0)                              # (B, K)
        if self.pooling == "mean":
            w = mask.float()
            return (h * w.unsqueeze(2)).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        # Cross-attention: which recent IPOs are most relevant to THIS deal?
        q = self.wq(query)                                        # (B, D)
        scores = torch.einsum("bkd,bd->bk", self.wk(h), q) / self.scale
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1) * mask.any(1, keepdim=True).float()
        return torch.einsum("bk,bkd->bd", w, h)


class GPRArm(nn.Module):
    def __init__(self, cfg: ModelConfig, n_engineered: int):
        super().__init__()
        self.mode = cfg.gpr_mode
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
                 n_panel_scalars: int, n_gpr_feats: int, horizons: tuple[int, ...]):
        super().__init__()
        self.cfg = cfg
        self.horizons = horizons
        self.static_arm = StaticArm(n_binary, n_bookrunners, cfg)
        fusion_in = cfg.static_out
        if cfg.use_panel:
            self.panel_arm = PanelArm(n_panel_scalars, cfg, query_dim=cfg.static_out)
            fusion_in += cfg.panel_out
        if cfg.use_gpr:
            self.gpr_arm = GPRArm(cfg, n_gpr_feats)
            fusion_in += cfg.gpr_out
        if cfg.gating == "film":
            if not cfg.use_gpr:
                raise ValueError("FiLM gating requires the GPR arm.")
            self.film_static = FiLM(cfg.gpr_out, cfg.static_out)
            if cfg.use_panel:
                self.film_panel = FiLM(cfg.gpr_out, cfg.panel_out)
        elif cfg.gating != "none":
            raise ValueError(f"Unknown gating: {cfg.gating}")

        dims = [fusion_in, *cfg.fusion_hidden]
        self.fusion = mlp(dims, cfg.dropout, out_act=True)
        self.heads = nn.ModuleDict({
            str(h): QuantileHead(dims[-1], cfg.quantiles) for h in horizons
        })

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        z_static = self.static_arm(batch["static_binary"], batch["static_bk"])
        parts = [z_static]
        z_gpr = None
        if self.cfg.use_gpr:
            z_gpr = self.gpr_arm(batch["gpr_seq"], batch["gpr_feats"])
        if self.cfg.gating == "film":
            parts[0] = self.film_static(z_static, z_gpr)
        if self.cfg.use_panel:
            z_panel = self.panel_arm(batch["panel_seq"], batch["panel_len"],
                                     batch["panel_scalars"], batch["panel_valid"],
                                     query=z_static)
            if self.cfg.gating == "film":
                z_panel = self.film_panel(z_panel, z_gpr)
            parts.append(z_panel)
        if z_gpr is not None:
            parts.append(z_gpr)
        rep = self.fusion(torch.cat(parts, dim=1))
        return {h: self.heads[str(h)](rep) for h in self.horizons}
