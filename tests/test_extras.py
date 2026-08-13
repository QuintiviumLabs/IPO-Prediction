"""Optional add-on rungs. The linear rung runs for real; the TabPFN rung is
exercised with a stub regressor so the wrapper logic (context assembly,
quantile-orientation handling, fallback path) is tested without needing the
pretrained weights."""
import numpy as np
import pytest

from ipo_model.extras import tabpfn_model
from ipo_model.extras.tabpfn_model import _as_quantile_matrix

sklearn = pytest.importorskip("sklearn")


def _fast_cfg(cfg):
    return cfg.override(**{"split.n_folds": 2, "train.seeds": (0,)})


# ---------------------------------------------------------------- linear ---

def test_linear_rung_runs(fs, cfg):
    from ipo_model.extras import linear
    res = linear.run(_fast_cfg(cfg), fs, model="ridge", verbose=False)
    assert len(res.fold_results) == 2
    assert np.isfinite(res.pooled["mae"])
    for r in res.fold_results:
        assert (np.diff(r.q_pred, axis=1) >= 0).all(), "quantiles crossed"


def test_linear_lasso_selects_features(fs, cfg):
    from ipo_model.extras import linear
    res = linear.run(_fast_cfg(cfg), fs, model="lasso", verbose=False)
    assert np.isfinite(res.pooled["rank_ic"]) or np.isnan(res.pooled["rank_ic"])
    coefs = linear.coefficients(_fast_cfg(cfg), fs, model="lasso")
    assert {"feature", "coef", "coef_std"} == set(coefs.columns)
    # sorted by |coef| descending
    a = coefs["coef"].abs().to_numpy()
    assert (a[:-1] >= a[1:] - 1e-12).all()


def test_linear_quantreg_gives_varying_widths(fs, cfg):
    from ipo_model.extras import linear
    small = cfg.override(**{"split.n_folds": 2, "split.min_train_frac": 0.6})
    res = linear.run(small, fs, model="ridge", quantile_method="quantreg",
                     verbose=False)
    widths = np.concatenate([r.q_pred[:, -1] - r.q_pred[:, 0]
                             for r in res.fold_results])
    assert widths.std() > 0, "quantreg should give deal-specific band widths"


def test_linear_residual_band_is_constant_width(fs, cfg):
    from ipo_model.extras import linear
    res = linear.run(_fast_cfg(cfg), fs, model="ridge",
                     quantile_method="residual", verbose=False)
    w = res.fold_results[0].q_pred[:, -1] - res.fold_results[0].q_pred[:, 0]
    assert np.allclose(w, w[0]), "residual method yields one width per fold"


# ---------------------------------------------------------------- tabpfn ---

class _StubRegressor:
    """Mimics TabPFNRegressor: records the context it was given."""
    last_context = {}

    def __init__(self, mode="quantiles"):
        self.mode = mode

    def fit(self, X, y):
        _StubRegressor.last_context = {"n": len(X), "p": X.shape[1]}
        self._mean = float(np.mean(y))
        return self

    def predict(self, X, output_type="mean", quantiles=None):
        n = len(X)
        if output_type == "quantiles":
            if self.mode != "quantiles":
                raise ValueError("quantile output not supported")
            # TabPFN returns one array per quantile -> (n_quantiles, n_rows)
            return [np.full(n, self._mean + (q - 0.5)) for q in quantiles]
        return np.full(n, self._mean)


def test_quantile_matrix_orientation():
    n_rows, n_q = 7, 3
    per_q = [np.arange(n_rows) + k for k in range(n_q)]      # (n_q, n_rows)
    out = _as_quantile_matrix(per_q, n_rows, n_q)
    assert out.shape == (n_rows, n_q)
    assert (out[:, 1] - out[:, 0] == 1).all()
    already = np.zeros((n_rows, n_q))                        # (n_rows, n_q)
    assert _as_quantile_matrix(already, n_rows, n_q).shape == (n_rows, n_q)
    with pytest.raises(ValueError):
        _as_quantile_matrix(np.zeros((5, 5)), n_rows, n_q)


def test_tabpfn_rung_with_stub(fs, cfg, monkeypatch):
    monkeypatch.setattr(tabpfn_model, "_make_regressor",
                        lambda device, n_estimators, seed: _StubRegressor())
    res = tabpfn_model.run(_fast_cfg(cfg), fs, verbose=False)
    assert len(res.fold_results) == 2
    for r in res.fold_results:
        assert r.q_pred.shape[1] == len(cfg.model.quantiles)
        assert (np.diff(r.q_pred, axis=1) >= 0).all()


def test_tabpfn_uses_train_plus_val_as_context(fs, cfg, monkeypatch):
    monkeypatch.setattr(tabpfn_model, "_make_regressor",
                        lambda device, n_estimators, seed: _StubRegressor())
    fast = _fast_cfg(cfg)
    tabpfn_model.run(fast, fs, verbose=False)
    from ipo_model.data.splits import purged_walk_forward
    last = purged_walk_forward(fs.dates, fs.label_end, fast.split)[-1]
    assert _StubRegressor.last_context["n"] == len(last.train_idx) + len(last.val_idx)


def test_tabpfn_truncates_to_most_recent(fs, cfg, monkeypatch):
    monkeypatch.setattr(tabpfn_model, "_make_regressor",
                        lambda device, n_estimators, seed: _StubRegressor())
    tabpfn_model.run(_fast_cfg(cfg), fs, max_train=40, verbose=False)
    assert _StubRegressor.last_context["n"] == 40


def test_tabpfn_falls_back_when_quantiles_unsupported(fs, cfg, monkeypatch):
    monkeypatch.setattr(tabpfn_model, "_make_regressor",
                        lambda device, n_estimators, seed: _StubRegressor(mode="point"))
    res = tabpfn_model.run(_fast_cfg(cfg), fs, verbose=False)
    for r in res.fold_results:
        assert np.isfinite(r.q_pred).all()
        assert (np.diff(r.q_pred, axis=1) >= 0).all()
