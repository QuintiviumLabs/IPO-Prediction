# IPO aftermarket return prediction

> **Setting this up on a Windows laptop with Bloomberg?** Open
> **[docs/runbook.html](docs/runbook.html)** — the full run-book: environment
> install, connecting your dataset, validation, the runs, reading the
> results, and troubleshooting. Text versions:
> [WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md) (setup) and
> [docs/BLOOMBERG_DATA.md](docs/BLOOMBERG_DATA.md) (where each field comes
> from on the terminal).

A framework for predicting IPO aftermarket returns from three information
sources, with gradient-boosted baselines the deep model has to beat:

- **Static arm** — sector flags (TMT, healthcare), bookrunner identity as a
  multi-hot binary vector (deals with 1+ banks) fed through a
  bag-of-embeddings projection, bookrunner count, plus the **syndicate
  block** when present in `ipos.csv` (`scripts/build_bookrunner_features.py`):
  size (`n_banks`, `log_n_banks`), bank-class flags (`has_bb`, `has_global`,
  `has_regional`, `has_local`, `has_domestic`), interactions
  (`combo_bb_x_dom`, `combo_bb_x_reg`), structure (`solo_book`,
  `jumbo_synd`), an 8-way `archetype_*` one-hot, and `prestige_rank_max` —
  each bookrunner's trailing-3y percentile share of same-market IPO
  proceeds, computed expanding in the pipeline so it is leakage-safe.
  (Market one-hot available behind `data.include_market_onehot`, off by
  default for market-agnostic training.)
- **Momentum arm** — an engineered market-state factor block (all same-market,
  all strictly pre-pricing), encoded by a small MLP:
  - **F1 · Recent deal aftermarket performance**: over the ≤10 most recent
    same-market IPOs within 90 days, median and value-weighted (deal proceeds)
    average of realised returns from offer at 1d and 1w — a deal's h-day
    return enters only if its h-th close predates the target's pricing.
    Extended stats (on by default): observable deal count and IQR.
  - **F2 · Break-issue rate**: share of that deal set trading below offer at
    1d / 1w, and average downside return among the breakers.
  - **F3 · Rolling supply**: same-market deal count, log proceeds, IPO count
    (90d) and issuance acceleration vs the trailing year — as
    expanding-window market-specific z-scores.
  - **M · Macro block**: local benchmark state (63d momentum, 252d drawdown,
    21d realized vol), GPR level/vol (63d), and — when the optional macro
    files exist — local vol index z-score and 21d FX move (`macro.csv`),
    global VIX z/change, HY OAS z, EM-vs-DM relative momentum
    (`macro_global.csv`), plus a Hot/Neutral/Cold issuance regime: mean
    first-day pop of the prior 10 same-market IPOs, cut by expanding
    terciles of that market's own history.
  - **P1 · Industry-subgroup peers** (`peers.csv`, from
    `scripts/pull_peers.py`): median/IQR of the as-of trailing 21d and 63d
    returns of up to K listed companies in the deal's Bloomberg industry
    subgroup, plus the peer count — validated strictly pre-pricing via the
    `asof` column.
- **GPR arm** — the last 21 days of the daily
  [Caldara–Iacoviello GPR index](https://www.matteoiacoviello.com/gpr.htm)
  encoded by a GRU, or as engineered summaries, or the bare level (an explicit
  representation ablation). `scripts/prepare_gpr.py` converts the downloaded
  file into the expected `gpr.csv`.

Fused by a small MLP into per-horizon **quantile heads** (10/50/90, pinball
loss, structurally non-crossing) at **1d / 3d / 1w / 1m** (trading days) —
the 1-month head is the target, shorter horizons are auxiliary regularizers.
Optional **FiLM gating** (V2) lets the GPR representation modulate the static
and momentum representations.

## Target definition

`y = log( (close_h / offer_price) / (I_h / I_0) )`

Log **outperformance vs the IPO's own market's benchmark index** (e.g. Hang
Seng for HK deals, S&P 500 for US) from offer price to the h-th daily close;
I_0 is the benchmark close just before the first trade. `exp(y)` is the
outperformance multiple — exp(y)=1.03 means the deal grew to 1.03× what its
benchmark grew to. This makes the target market-agnostic: the model predicts
general outperformance, not each market's index moves. The market one-hot is
likewise excluded from features by default (`data.include_market_onehot`).
Set `data.market_adjust: false` for raw returns. F1/F2 momentum returns are
raw-from-offer per the factor spec (`data.momentum_market_adjust: true`
switches them to benchmark-relative — a worthwhile ablation).

## Evaluation

**Purged walk-forward CV.** Labels are 21-day forward windows, so a row enters
a training/validation set only if its label window (plus a 5-day embargo) ends
before the period it would leak into. Five expanding-window folds over the
later ~65% of the sample; early stopping on a purged validation slice; one
model per seed, quantile forecasts ensembled across seeds. Expanding-window
statistics (F3 z-scores, regime terciles, prestige ranks) use only each
row's past by construction — verified by a prefix-stability test.

**Metrics** (`ipo_model/training/metrics.py`). Ordering — Spearman rank IC
(with p-value), Pearson IC, decile/quintile spread, top and bottom bucket
means, hit rate. Level — MAE, median AE, RMSE, bias, and an OOS R² against
the natural null of predicting zero outperformance. Distribution — pinball
loss, interval coverage vs nominal, which tail is missing
(`frac_below_lo` / `frac_above_hi`), mean interval width, and the Winkler
interval score. For a cross-sectional signal, rank IC and bucket spread are
the ones that decide.

**Significance.** A point estimate without an error bar is not a result at
this sample size, so the pipeline ships the machinery to test every number:
`ic_by_period` turns one pooled IC into a quarterly IC series, `t_test`
applies Newey-West standard errors, `bootstrap_ci` gives block-aware
distribution-free intervals, `compare_models` runs a paired
Diebold-Mariano test plus a paired IC bootstrap between two rungs on
identical deals, and `benjamini_hochberg` controls the false-discovery rate
when many rungs are compared at once.

> **Read the fold-mean IC, not the pooled IC.** Pooling mixes deals across
> folds; because each fold's model is trained on different data, level
> differences between folds can inflate the pooled correlation with no real
> within-fold skill. The `00_naive_const` rung demonstrates this — pooled IC
> looks positive, fold-wise IC is correctly undefined.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# synthetic data in the expected schema (~4000 IPOs, 3 markets, planted signal)
python scripts/make_synthetic_data.py --out data

# real GPR: download the daily file from matteoiacoviello.com/gpr.htm, then
python scripts/prepare_gpr.py --in data_gpr_daily_recent.csv --out data/gpr.csv

# ablation 0: trees on the engineered features — the bar to clear
python scripts/run_baseline.py --data data --engine both

# full three-arm model (V1); add --film for V2 gating
python scripts/run_full.py --data data --seeds 0 1 2

# the whole ablation ladder -> results/ablations.{csv,md}
# (also saves per-deal predictions to results/predictions/)
python scripts/run_ablations.py --data data --seeds 0 1 2

# significance tests, paired comparisons, FDR — no retraining needed
python scripts/analyze_results.py --results results

# deploy: fit once on ALL labeled data, save weights + scaler + schema...
python scripts/fit_final.py --data data --out results/final_model
# ...then score any new deal (append its row to ipos.csv; no prices needed)
python scripts/predict.py --data data --bundle results/final_model --new-only

pytest
```

Weights & Biases logging is off by default; enable with `train.wandb: true`
in a YAML config (`pip install wandb`; on a locked-down machine set
`WANDB_MODE=offline` and `wandb sync` later). It logs per-epoch validation
loss per fold/seed, per-fold metrics, and the run summary.

## Ablation ladder

| rung | what it tests |
|---|---|
| 00_naive_const | predict the training median for every deal — no ordering at all, so it floors the *distribution* metrics |
| 00_naive_f1 | one-factor regression on recent same-market IPO performance — the floor that matters: does the full model beat the single obvious signal? |
| 0_lgbm / 0_xgb | engineered features + trees, two engines: the baseline, engine-robust |
| 0_lgbm_raw / 0_xgb_raw | + the raw GPR window as columns: does unsummarized GPR help trees? |
| 1_static | deal characteristics alone |
| 2_static+f1 | + F1 only — the headline momentum factor |
| 3_static+momentum | + the full factor block (F1–F3, macro, peers) |
| 4_momentum+gpr_level | + the *level* of GPR |
| 5_full_v1 | + the GPR temporal encoder (does GPR *dynamics* beat its level?) |
| 6_full_v2_film | + FiLM gating (does the risk regime modulate the other arms?) |

Optional add-on rungs live in [`ipo_model/extras/`](ipo_model/extras/README.md)
and are **excluded from the default ladder** so a missing dependency can never
break it (`pip install -r requirements-extras.txt`, then request them by name):

| rung | what it tests |
|---|---|
| `x_ridge` / `x_lasso` / `x_elasticnet` | regularized linear — arguably the right complexity at ~1k rows; if it ties the net, that is a finding |
| `x_tabpfn` | TabPFN in-context learning, purpose-built for small tabular data; emits its own predictive distribution |

```bash
python scripts/run_extras.py --data data --config configs/small.yaml --what linear tabpfn
python scripts/run_ablations.py --data data --rungs 0_lgbm x_ridge x_tabpfn 5_full_v1
```

## Plugging in real data

Drop CSVs in a directory (see `ipo_model/data/features.py` for details):

| file | columns |
|---|---|
| `ipos.csv` | `ipo_id`, `first_trade_date`, `offer_price`, `market`, `deal_size` (proceeds), `is_tmt`, `is_healthcare`, `bk_*` (one binary column per bookrunner); optional `is_target` (0 = momentum-universe only: feeds the factors, never modelled), `subgroup` (`scripts/pull_subgroups.py`), and the syndicate block (`scripts/build_bookrunner_features.py`) |
| `prices.csv` | `ipo_id`, `date`, `close` (daily closes; ~26 trading days per IPO suffices) |
| `gpr.csv` | `date`, `gpr` (daily index level — see `scripts/prepare_gpr.py`) |
| `market.csv` | `date`, `market`, `close` — one benchmark index per market (Hang Seng rows for HK, S&P rows for US, …); a single global `date`, `close` series is also accepted |
| `deals.csv` *(optional)* | `date`, `market`, `proceeds` — all deal types (IPO, follow-on, convertible) for F3; without it, supply factors fall back to the IPO record only |
| `macro.csv` *(optional)* | `date`, `market`, `vol_index`, `fx` — local vol index and FX vs USD per market (`scripts/pull_macro.py`); missing columns just skip their features |
| `macro_global.csv` *(optional)* | `date`, `vix`, `hy_oas`, `em`, `acwi` — global risk series (`scripts/pull_macro.py`) |
| `peers.csv` *(optional)* | `ipo_id`, `peer`, `ret_21d`, `ret_63d`, `asof` — as-of trailing peer returns (`scripts/pull_peers.py`; needs `data/peer_universe.csv` exported from the Terminal) |

Data caveats the code cannot check for you:

- **Survivorship**: build `ipos.csv` from a point-in-time listing record
  including deals that later delisted — a current-constituent list quietly
  drops the worst performers and inflates the momentum signal.
- **GPR vintage**: use values as they were observable on each date. A
  retrospectively revised index leaks future information even though the code
  windows it correctly.
- Rows near the end of the sample without 21 closes are dropped automatically;
  bookrunner columns with < `min_bookrunner_deals` deals in a training window
  are pooled into a shared "other" bucket per fold.

## Layout

```
ipo_model/
  config.py              all knobs; ablation rungs are config overrides
  data/synthetic.py      synthetic data generator (schema + planted signal)
  data/features.py       leakage-safe features: targets, factor blocks
                         (F1-F3, macro, peers), syndicate extras, GPR
  data/preprocess.py     per-fold scalers + rare-bookrunner bucketing
  data/splits.py         purged walk-forward CV
  models/model.py        three arms, FiLM, quantile heads
  training/{losses,metrics,loop}.py
  results_io.py          per-deal prediction persistence (post-hoc analysis)
                         — every run script saves to results/predictions/;
                         analysis then needs no retraining
  persist.py             deployable model bundles: fit_final / save / load /
                         predict (weights + scaler + transform + schema)
  baselines/{lgbm,xgb}.py  ablation 0 (two tree engines, shared fold loop)
  extras/                optional rungs: regularized linear, TabPFN
  diagnostics.py         feature importance + failure slices
  ablation.py            the ladder
configs/bank_classes.csv seed bank→class/home-region map (REVIEW BY HAND)
scripts/                 CLI entry points: pulls (pull_bloomberg, pull_macro,
                         pull_subgroups, pull_peers, prepare_gpr), feature
                         build (build_bookrunner_features), runs, analysis,
                         deploy (fit_final, predict)
docs/architecture.html   the architecture diagram (also published as an artifact)
tests/                   leakage/purging properties, factor recomputation,
                         prefix-stability (no-lookahead), model mechanics
```
