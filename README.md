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
  bag-of-embeddings projection, bookrunner count. (Market one-hot available
  behind `data.include_market_onehot`, off by default for market-agnostic
  training.)
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
  - **F4 · Sector issuance density**: same-market same-sector (TMT /
    healthcare / other) count, proceeds share, log proceeds (90d), plus the
    all-markets same-sector count (30d) — expanding-window z-scores.
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
z-scores (F3/F4) use only each row's past by construction — verified by a
prefix-stability test.

Metrics: Spearman rank IC, hit rate, decile spread, MAE/RMSE, pinball loss,
10–90 interval coverage. For a cross-sectional signal, rank IC and decile
spread are the ones that matter.

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
python scripts/run_ablations.py --data data --seeds 0 1 2

pytest
```

## Ablation ladder

| rung | what it tests |
|---|---|
| 0_lgbm / 0_xgb | engineered features + trees, two engines: the baseline, engine-robust |
| 0_lgbm_raw / 0_xgb_raw | + the raw GPR window as columns: does unsummarized GPR help trees? |
| 1_static | deal characteristics alone |
| 2_static+f1 | + F1 only — the headline momentum factor |
| 3_static+momentum | + the full F1–F4 factor block |
| 4_momentum+gpr_level | + the *level* of GPR |
| 5_full_v1 | + the GPR temporal encoder (does GPR *dynamics* beat its level?) |
| 6_full_v2_film | + FiLM gating (does the risk regime modulate the other arms?) |

## Plugging in real data

Drop CSVs in a directory (see `ipo_model/data/features.py` for details):

| file | columns |
|---|---|
| `ipos.csv` | `ipo_id`, `first_trade_date`, `offer_price`, `market`, `deal_size` (proceeds), `is_tmt`, `is_healthcare`, `bk_*` (one binary column per bookrunner); optional `is_target` (0 = momentum-universe only: feeds F1–F4, never modelled) |
| `prices.csv` | `ipo_id`, `date`, `close` (daily closes; ~26 trading days per IPO suffices) |
| `gpr.csv` | `date`, `gpr` (daily index level — see `scripts/prepare_gpr.py`) |
| `market.csv` | `date`, `market`, `close` — one benchmark index per market (Hang Seng rows for HK, S&P rows for US, …); a single global `date`, `close` series is also accepted |
| `deals.csv` *(optional)* | `date`, `market`, `sector` (`tmt`/`healthcare`/`other`), `proceeds` — all deal types (IPO, follow-on, convertible) for F3/F4; without it, supply factors fall back to the IPO record only |

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
  data/features.py       leakage-safe features: targets, F1-F4 factors, GPR
  data/preprocess.py     per-fold scalers + rare-bookrunner bucketing
  data/splits.py         purged walk-forward CV
  models/model.py        three arms, FiLM, quantile heads
  training/{losses,metrics,loop}.py
  baselines/{lgbm,xgb}.py  ablation 0 (two tree engines, shared fold loop)
  ablation.py            the ladder
scripts/                 CLI entry points (incl. prepare_gpr.py)
docs/architecture.html   the architecture diagram (also published as an artifact)
tests/                   leakage/purging properties, factor recomputation,
                         prefix-stability (no-lookahead), model mechanics
```
