"""Pinball (quantile) loss and the multi-horizon objective."""
from __future__ import annotations

import torch


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
