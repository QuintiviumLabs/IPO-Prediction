"""Pinball (quantile) and binary (outperform/underperform) losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 quantiles: tuple[float, ...]) -> torch.Tensor:
    """pred: (B, Q) ascending quantile forecasts; target: (B,). Mean over B and Q."""
    q = torch.as_tensor(sorted(quantiles), dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(1) - pred                      # (B, Q)
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def multi_horizon_loss(preds: dict[int, torch.Tensor], targets: dict[int, torch.Tensor],
                       quantiles: tuple[float, ...], main_horizon: int,
                       aux_weight: float) -> torch.Tensor:
    """Main horizon weighted 1.0; auxiliary horizons act as regularizers."""
    total = torch.zeros((), device=next(iter(preds.values())).device)
    for h, pred in preds.items():
        w = 1.0 if h == main_horizon else aux_weight
        total = total + w * pinball_loss(pred, targets[h], quantiles)
    return total


def binary_multi_horizon_loss(preds: dict[int, torch.Tensor],
                              targets: dict[int, torch.Tensor],
                              main_horizon: int, aux_weight: float,
                              pos_weight: dict[int, torch.Tensor] | None = None
                              ) -> torch.Tensor:
    """Class-weighted BCE per horizon; preds are (B, 1) logits, targets (B,)
    in {0, 1} (1 = outperformed the benchmark). pos_weight[h] = n_neg/n_pos of
    the TRAINING fold, so a pop-heavy base rate cannot be gamed by always
    predicting 'outperform'."""
    total = torch.zeros((), device=next(iter(preds.values())).device)
    for h, logit in preds.items():
        w = 1.0 if h == main_horizon else aux_weight
        pw = pos_weight.get(h) if pos_weight else None
        total = total + w * F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), targets[h], pos_weight=pw)
    return total
