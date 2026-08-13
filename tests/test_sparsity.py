"""Group-lasso input sparsity and the GPR change-input option."""
import numpy as np
import torch

from ipo_model.config import Config
from ipo_model.models.model import ThreeArmModel
from ipo_model.training.losses import multi_horizon_loss


def _batch(B=64, n_mom=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "static_binary": torch.randn(B, 3, generator=g),
        "static_bk": (torch.rand(B, 6, generator=g) < 0.3).float(),
        "momentum": torch.randn(B, n_mom, generator=g),
        "gpr_seq": torch.randn(B, 21, generator=g),
        "gpr_feats": torch.randn(B, 5, generator=g),
    }


def _model(cfg, batch):
    return ThreeArmModel(cfg.model, n_binary=3, n_bookrunners=6,
                         n_momentum=batch["momentum"].shape[1], n_gpr_feats=5,
                         horizons=tuple(sorted(cfg.data.horizons)))


def _train(cfg, steps=300, seed=0):
    torch.manual_seed(seed)
    batch = _batch(seed=seed)
    # Real signal in momentum columns 0 and 1; columns 2..19 are pure noise.
    sig = 0.5 * batch["momentum"][:, 0] + 0.3 * batch["momentum"][:, 1]
    targets = {h: sig + torch.randn(64) * 0.05 for h in sorted(cfg.data.horizons)}
    model = _model(cfg, batch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = multi_horizon_loss(model(batch), targets, cfg.model.quantiles,
                                  cfg.main_horizon, cfg.model.aux_weight)
        if cfg.model.l1_input > 0:
            loss = loss + cfg.model.l1_input * model.input_l1()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def test_group_lasso_prunes_noise_but_keeps_signal():
    """With signal planted in columns 0-1, group lasso must collapse the 18
    noise columns RELATIVE to the signal columns; without it they persist."""
    plain = _train(Config())
    sparse = _train(Config().override(**{"model.l1_input": 3e-3}))
    n_plain = plain.momentum_input_norms()
    n_sparse = sparse.momentum_input_norms()

    ratio = lambda n: np.median(n[2:]) / max(n[:2].max(), 1e-12)
    assert ratio(n_sparse) < 0.4 * ratio(n_plain)  # noise collapses vs signal
    assert n_sparse[:2].max() > 5 * np.median(n_sparse[2:])  # signal survives
    dead = (n_sparse < 0.05 * n_sparse.max()).sum()
    assert dead >= 8  # most noise columns effectively pruned


def test_l1_penalty_is_differentiable_scalar():
    cfg = Config()
    model = _model(cfg, _batch())
    pen = model.input_l1()
    assert pen.dim() == 0 and pen.item() > 0
    pen.backward()
    assert model.momentum_arm.net[0].weight.grad is not None


def test_gpr_change_input_runs_and_differs():
    base = Config()
    chg = Config().override(**{"model.gpr_input": "change"})
    batch = _batch()
    torch.manual_seed(7); m1 = _model(base, batch); m1.eval()
    torch.manual_seed(7); m2 = _model(chg, batch); m2.eval()
    with torch.no_grad():
        o1 = m1(batch)[21]
        o2 = m2(batch)[21]
    assert o1.shape == o2.shape
    assert not torch.allclose(o1, o2)  # same weights, different GPR encoding

    # constant GPR level -> zero changes: the arm must be level-invariant
    b2 = {k: v.clone() for k, v in batch.items()}
    b2["gpr_seq"] = torch.full_like(b2["gpr_seq"], 5.0)
    b3 = {k: v.clone() for k, v in b2.items()}
    b3["gpr_seq"] = torch.full_like(b3["gpr_seq"], -3.0)
    with torch.no_grad():
        assert torch.allclose(m2(b2)[21], m2(b3)[21], atol=1e-6)
