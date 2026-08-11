import numpy as np
import torch

from ipo_model.training.losses import pinball_loss
from ipo_model.training.metrics import evaluate


def test_pinball_known_values():
    pred = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([1.0])
    # err = 1 for all three quantiles: loss_q = q * 1
    expected = np.mean([0.1, 0.5, 0.9])
    assert np.isclose(float(pinball_loss(pred, target, (0.1, 0.5, 0.9))),
                      expected, atol=1e-6)

    target = torch.tensor([-1.0])
    # err = -1: loss_q = (1 - q)
    expected = np.mean([0.9, 0.5, 0.1])
    assert np.isclose(float(pinball_loss(pred, target, (0.1, 0.5, 0.9))),
                      expected, atol=1e-6)


def test_pinball_asymmetry_direction():
    """q=0.9 must penalize under-prediction more than over-prediction."""
    over = pinball_loss(torch.tensor([[1.0]]), torch.tensor([0.0]), (0.9,))
    under = pinball_loss(torch.tensor([[-1.0]]), torch.tensor([0.0]), (0.9,))
    assert under > over


def test_evaluate_perfect_ranking():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, 400)
    q = np.stack([y - 0.05, y, y + 0.05], axis=1)  # perfect median forecast
    m = evaluate(y, q, (0.1, 0.5, 0.9))
    assert np.isclose(m["rank_ic"], 1.0)
    assert np.isclose(m["hit_rate"], 1.0)
    assert m["decile_spread"] > 0
    assert np.isclose(m["coverage"], 1.0)
    assert np.isclose(m["mae"], 0.0)


def test_evaluate_antisignal():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.1, 400)
    q = np.stack([-y - 0.01, -y, -y + 0.01], axis=1)
    m = evaluate(y, q, (0.1, 0.5, 0.9))
    assert np.isclose(m["rank_ic"], -1.0)
    assert m["decile_spread"] < 0
