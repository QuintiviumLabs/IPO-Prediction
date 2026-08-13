# Optional add-on rungs

Self-contained model variants that sit outside the core pipeline. Each
imports its third-party dependency **lazily**, so the default ladder and all
existing scripts run unchanged whether or not these are installed.

```powershell
pip install -r requirements-extras.txt
```

Both reuse `baselines.common.run_folds`, so they see exactly the same purged
walk-forward folds, per-fold scalers, engineered feature table and metrics as
every other rung — the numbers drop straight into your ablation table.

## `linear.py` — Ridge / Lasso / ElasticNet

At ~1k rows and ~25 features this is arguably the right complexity class, and
it is the most defensible model in the ladder. **If it matches the trees and
the neural net, that is a finding**, not a failure: the relationship is
approximately linear in these factors.

```powershell
python scripts\run_extras.py --data data --config configs\small.yaml --what linear --model ridge
python scripts\run_extras.py --data data --config configs\small.yaml --what linear --model lasso
```

- Features are standardized on **training-fold** statistics before fitting
  (the engineered table is raw-scaled and regularization is scale-sensitive).
- The penalty strength is chosen per fold on the **purged validation slice**,
  never on test.
- `--quantile-method residual` (default) takes empirical quantiles of the
  validation residuals as constant offsets — fast and robust, but every deal
  gets the same band width. `--quantile-method quantreg` fits a separate
  L1-penalized quantile regression per level for deal-specific widths;
  slower, and noisy in the tails at small N.
- Writes `results\x_<model>_coefficients.csv` — signed standardized
  coefficients with their across-fold standard deviation. Read the std: a
  coefficient that flips sign between folds is noise, not a finding.
- Lasso doubles as feature selection — the printed nonzero count per fold
  tells you how many factors survive.

## `tabpfn_model.py` — TabPFN in-context learning

A transformer pre-trained on millions of synthetic tabular tasks. It does not
train on your data; it conditions on it in one forward pass. That is exactly
the few-shot regime this dataset sits in, and it emits a full predictive
distribution, so the quantile band comes natively.

```powershell
python scripts\run_extras.py --data data --config configs\small.yaml --what tabpfn
```

Two deliberate differences from other rungs, both documented in the module:

- **Train and validation rows are concatenated into the in-context set.**
  There is no fitting and no early stopping, so a validation slice protects
  against nothing — and both blocks are already purged against the test
  block, so this adds context without leaking.
- **If the context exceeds `max_train`, the most recent rows are kept.**
  Truncating the distant past is the right bias for a time series.

### Installation, and the gotcha worth knowing

`requirements-extras.txt` pins **`tabpfn==2.2.1` deliberately**: its weights
are a plain unauthenticated download. Version 8.x works with the same code,
but its weights are **gated on HuggingFace** — you must accept the terms at
`huggingface.co/Prior-Labs/tabpfn_3` and authenticate (`hf auth login` or an
`HF_TOKEN`), which is often blocked on managed laptops.

**Firewalled machine?** Download the checkpoint in a browser from
`https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt`
and place it at `%USERPROFILE%\.cache\tabpfn\tabpfn-v2-regressor.ckpt`.

### Caveats

- TabPFN assumes i.i.d. tabular data — it has no notion of time. The purged
  walk-forward wrapper is what keeps the evaluation honest; do not trust it
  to handle regime drift on its own.
- It degrades past ~10k rows and ~500 features. You are far inside the row
  limit; watch the feature count if you have many bookrunner columns (the
  module warns).
- First run downloads weights (a few hundred MB) and is slow; later runs
  reuse the cache. `--device cuda` if the laptop has a usable GPU.
- **Not verified end-to-end here** — this sandbox cannot reach HuggingFace,
  so the wrapper is tested against a stub regressor (shape handling, context
  assembly, fallback path). Run it on your machine and sanity-check the first
  fold's output before trusting the numbers.

## Using them as ablation rungs

They are excluded from the default ladder on purpose, so a missing dependency
can never break `run_ablations.py`. Request them explicitly:

```powershell
python scripts\run_ablations.py --data data --config configs\small.yaml ^
    --rungs 0_lgbm 0_xgb x_ridge x_lasso x_tabpfn 3_static+momentum 5_full_v1
```

Available: `x_ridge`, `x_lasso`, `x_elasticnet`, `x_tabpfn`.
