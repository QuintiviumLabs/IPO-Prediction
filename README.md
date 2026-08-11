# IPO aftermarket return prediction

A framework for predicting IPO aftermarket returns from three information
sources, with a gradient-boosted baseline that the deep model has to beat:

- **Static arm** — sector flags (TMT, healthcare), bookrunner identity as a
  multi-hot binary vector (deals with 1+ banks) fed through a bag-of-embeddings
  projection, bookrunner count.
- **Recent-IPO panel arm** — the 20 most recent prior IPOs, each as an
  event-time sequence of market-excess daily log returns (up to 21 days,
  truncated to what was observable at pricing), encoded by one shared GRU,
  pooled by masked mean (V1) or cross-attention with the target's static
  embedding as query (V2: "how did recent deals *like this one* trade?").
- **GPR arm** — the last 21 days of the geopolitical-risk index, encoded by a
  GRU, or as engineered summaries, or the bare level (an explicit
  representation ablation).

Fused by a small MLP into per-horizon **quantile heads** (10/50/90, pinball
loss, structurally non-crossing) at **1, 5, and 21 trading days** — the 21-day
head is the target, the short horizons are auxiliary regularizers. Optional
**FiLM gating** lets the GPR representation modulate the static and panel
representations ("the geopolitical regime decides how much recent momentum
and deal quality matter").

## Target definition

`y = log(close_21 / offer_price) − market log return over the same span`

i.e. the 21-trading-day log return from the offer price, in excess of the
market index (anchored at the last market close before the first trade). Set
`data.market_adjust: false` for raw returns. Horizon 1 is the first-day pop.

## Evaluation

**Purged walk-forward CV.** Labels are 21-day forward windows, so a row enters
a training/validation set only if its label window (plus a 5-day embargo) ends
before the period it would leak into. Five expanding-window folds over the
later ~65% of the sample; early stopping on a purged validation slice; one
model per seed, quantile forecasts ensembled across seeds.

Metrics: Spearman rank IC, hit rate, decile spread (top-minus-bottom bucket
realized return), MAE/RMSE, pinball loss, 10–90 interval coverage. For a
cross-sectional signal, rank IC and decile spread are the ones that matter.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# synthetic data in the expected schema (~4000 IPOs, planted signal)
python scripts/make_synthetic_data.py --out data

# ablation 0: LightGBM on engineered features — the bar to clear
python scripts/run_baseline.py --data data

# full three-arm model (V1); add --attn --film for V2
python scripts/run_full.py --data data --seeds 0 1 2

# the whole ablation ladder -> results/ablations.{csv,md}
python scripts/run_ablations.py --data data --seeds 0 1 2

pytest
```

## Ablation ladder

| rung | what it tests |
|---|---|
| 0_lgbm | engineered features + trees: do sequences add anything at all? |
| 1_static | deal characteristics alone |
| 2_static+gpr_level | does the *level* of GPR matter? |
| 3_static+gpr_seq | does GPR *dynamics* beat its level? |
| 4_static+panel | does recent-IPO momentum matter? |
| 5_full_v1 | both temporal arms (mean pooling, no gating) |
| 6_full_v2_attn_film | cross-attention pooling + FiLM gating |

## Plugging in real data

Drop four CSVs in a directory (see `ipo_model/data/features.py` for details):

| file | columns |
|---|---|
| `ipos.csv` | `ipo_id`, `first_trade_date`, `offer_price`, `is_tmt`, `is_healthcare`, `bk_*` (one binary column per bookrunner) |
| `prices.csv` | `ipo_id`, `date`, `close` (daily closes; ~26 trading days per IPO suffices) |
| `gpr.csv` | `date`, `gpr` (daily index level) |
| `market.csv` | `date`, `close` (index level for market adjustment) |

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
  data/features.py       leakage-safe feature construction
  data/preprocess.py     per-fold scalers + rare-bookrunner bucketing
  data/splits.py         purged walk-forward CV
  models/model.py        three arms, FiLM, quantile heads
  training/{losses,metrics,loop}.py
  baselines/lgbm.py      ablation 0
  ablation.py            the ladder
scripts/                 CLI entry points
tests/                   leakage/purging properties, model mechanics, smoke tests
```
