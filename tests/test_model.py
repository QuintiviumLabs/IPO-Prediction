"""Model mechanics: shapes, quantile monotonicity, gating variants."""
import torch
import pytest

from ipo_model.config import Config
from ipo_model.models.model import ThreeArmModel


def _batch(B=8, n_bin=6, n_bk=7, n_mom=20, W=21, n_gf=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "static_binary": torch.randn(B, n_bin, generator=g),
        "static_bk": (torch.rand(B, n_bk, generator=g) < 0.3).float(),
        "momentum": torch.randn(B, n_mom, generator=g),
        "gpr_seq": torch.randn(B, W, generator=g),
        "gpr_feats": torch.randn(B, n_gf, generator=g),
    }


def _model(cfg: Config, batch):
    return ThreeArmModel(
        cfg.model,
        n_binary=batch["static_binary"].shape[1],
        n_bookrunners=batch["static_bk"].shape[1],
        n_momentum=batch["momentum"].shape[1],
        n_gpr_feats=batch["gpr_feats"].shape[1],
        horizons=tuple(sorted(cfg.data.horizons)),
    )


@pytest.mark.parametrize("overrides", [
    {},
    {"model.use_momentum": False, "model.use_gpr": False},
    {"model.use_momentum": False, "model.gpr_mode": "level"},
    {"model.gpr_mode": "engineered"},
    {"model.gating": "film"},
])
def test_forward_shapes_and_monotone_quantiles(overrides):
    cfg = Config().override(**overrides)
    batch = _batch()
    model = _model(cfg, batch)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert set(out) == set(cfg.data.horizons)  # 1d, 3d, 1w, 1m
    for q in out.values():
        assert q.shape == (8, len(cfg.model.quantiles))
        assert torch.isfinite(q).all()
        assert (q[:, 1:] >= q[:, :-1] - 1e-6).all(), "quantiles crossed"


def test_film_starts_at_identity():
    from ipo_model.models.model import FiLM
    torch.manual_seed(3)
    film = FiLM(cond_dim=8, target_dim=16)
    z = torch.randn(4, 16)
    cond = torch.randn(4, 8)
    assert torch.allclose(film(z, cond), z, atol=1e-7), \
        "zero-init FiLM must start as identity"
    with torch.no_grad():
        film.net.weight.add_(0.1)
    assert not torch.allclose(film(z, cond), z, atol=1e-4)


def test_momentum_group_subsetting(fs, cfg):
    """momentum_columns must slice exactly the requested factor groups."""
    f1 = fs.momentum_columns(("f1",))
    all_groups = fs.momentum_columns(("f1", "f2", "f3", "m", "p1"))
    n_f1 = sum(1 for n in fs.momentum_names if n.startswith("f1_"))
    assert f1.shape == (len(fs), n_f1)
    assert all_groups.shape == (len(fs), len(fs.momentum_names))
    m_only = fs.momentum_columns(("m",))
    assert m_only.shape[1] == sum(1 for n in fs.momentum_names
                                  if n.startswith("m_"))
