"""Model mechanics: shapes, quantile monotonicity, masking, gating variants."""
import numpy as np
import torch
import pytest

from ipo_model.config import Config
from ipo_model.models.model import ThreeArmModel


def _batch(B=8, K=20, T=21, W=21, n_bin=3, n_bk=7, n_scal=4, n_gf=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, T + 1, (B, K), generator=g)
    lengths[:, 0] = T  # ensure at least one full slot
    return {
        "static_binary": torch.randn(B, n_bin, generator=g),
        "static_bk": (torch.rand(B, n_bk, generator=g) < 0.3).float(),
        "panel_seq": torch.randn(B, K, T, generator=g) * 0.02,
        "panel_len": lengths,
        "panel_scalars": torch.randn(B, K, n_scal, generator=g),
        "panel_valid": lengths > 0,
        "gpr_seq": torch.randn(B, W, generator=g),
        "gpr_feats": torch.randn(B, n_gf, generator=g),
    }


def _model(cfg: Config, batch):
    return ThreeArmModel(
        cfg.model,
        n_binary=batch["static_binary"].shape[1],
        n_bookrunners=batch["static_bk"].shape[1],
        n_panel_scalars=batch["panel_scalars"].shape[2],
        n_gpr_feats=batch["gpr_feats"].shape[1],
        horizons=tuple(sorted(cfg.data.horizons)),
    )


@pytest.mark.parametrize("overrides", [
    {},
    {"model.use_panel": False, "model.use_gpr": False},
    {"model.use_panel": False, "model.gpr_mode": "level"},
    {"model.gpr_mode": "engineered"},
    {"model.panel_pooling": "attn", "model.gating": "film"},
])
def test_forward_shapes_and_monotone_quantiles(overrides):
    cfg = Config().override(**overrides)
    batch = _batch()
    model = _model(cfg, batch)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert set(out) == set(cfg.data.horizons)
    for q in out.values():
        assert q.shape == (8, len(cfg.model.quantiles))
        assert torch.isfinite(q).all()
        assert (q[:, 1:] >= q[:, :-1] - 1e-6).all(), "quantiles crossed"


def test_empty_panel_slots_do_not_contribute():
    """Changing values inside masked-out panel slots must not change output."""
    cfg = Config()
    batch = _batch()
    batch["panel_len"][:, 5:] = 0
    batch["panel_valid"] = batch["panel_len"] > 0
    model = _model(cfg, batch)
    model.eval()
    with torch.no_grad():
        out1 = model(batch)[cfg.main_horizon]
        batch2 = {k: v.clone() for k, v in batch.items()}
        batch2["panel_seq"][:, 5:] = 99.0
        batch2["panel_scalars"][:, 5:] = -99.0
        out2 = model(batch2)[cfg.main_horizon]
    assert torch.allclose(out1, out2, atol=1e-5)


def test_film_starts_at_identity():
    from ipo_model.models.model import FiLM
    torch.manual_seed(3)
    film = FiLM(cond_dim=8, target_dim=16)
    z = torch.randn(4, 16)
    cond = torch.randn(4, 8)
    assert torch.allclose(film(z, cond), z, atol=1e-7), \
        "zero-init FiLM must start as identity"
    # And it must stop being the identity once parameters move.
    with torch.no_grad():
        film.net.weight.add_(0.1)
    assert not torch.allclose(film(z, cond), z, atol=1e-4)


def test_truncated_sequence_ignores_padding():
    """Padding beyond a slot's length must not affect its representation."""
    cfg = Config()
    batch = _batch()
    batch["panel_len"][:, :] = 7
    batch["panel_valid"][:, :] = True
    model = _model(cfg, batch)
    model.eval()
    with torch.no_grad():
        out1 = model(batch)[cfg.main_horizon]
        batch2 = {k: v.clone() for k, v in batch.items()}
        batch2["panel_seq"][:, :, 7:] = 123.0
        out2 = model(batch2)[cfg.main_horizon]
    assert torch.allclose(out1, out2, atol=1e-5)
