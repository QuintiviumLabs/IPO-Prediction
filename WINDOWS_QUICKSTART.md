# Running this on a Windows laptop (with Bloomberg)

The pipeline is plain Python — no databases, no services. Four **core CSV
files** make it runnable; four more optional files (peers, macro, syndicate
columns, all-deal supply) switch on the newer factor blocks. Moving
machines = clone the repo + recreate the venv + copy your `data/` folder.
Nothing else to migrate.

## How the repo works, in one paragraph

`ipo_model/data/features.py` reads the CSVs and builds, for every IPO, only
information observable before its pricing date: the market-state factor
blocks (F1–F3 recent deals/breaks/supply, macro state, industry-subgroup
peers), the bookrunner syndicate block, a GPR window, and the 1d/3d/1w/1m
targets. `ipo_model/data/splits.py` cuts time-ordered train/test folds with
the overlap purged so no future leaks. Then two model tracks run on
identical folds: gradient-boosted trees on the engineered features
(`scripts/run_baseline.py` — the bar to clear) and the three-arm neural net
(`scripts/run_full.py`); both now predict **P(outperform the benchmark)**
by default (`model.head: binary`). `scripts/run_ablations.py` runs the
whole ladder and writes a results table; `scripts/diagnose.py` tells you
which features matter and where the model fails; `scripts/fit_final.py` +
`scripts/predict.py` turn the finished model into a saved bundle that
scores brand-new deals. `configs/small.yaml` has the right settings for
~1.5k rows.

## 1. One-time setup

1. Install **Python 3.11** from https://www.python.org/downloads/ — tick
   **"Add python.exe to PATH"** during install.
2. Install **Git** from https://git-scm.com/download/win (or use
   GitHub → Code → Download ZIP and unzip it).
3. In **PowerShell**, in the folder where you want the project:

```powershell
git clone https://github.com/QuintiviumLabs/IPO-Prediction.git ipo-model
cd ipo-model
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -r requirements-extras.txt   # wandb, openpyxl (.xlsx), sklearn, tabpfn
pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
pip install xbbg
```

If PowerShell refuses to run the activate script, run this once and retry:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Sanity check the install (no real data needed — the test suite generates
synthetic data itself, ~1 min; then a demo run, ~2 min):

```powershell
pytest -q
python scripts\make_synthetic_data.py --out demo_data --n-ipos 400
python scripts\run_baseline.py --data demo_data
```

With the Terminal running and logged in, confirm the API too:

```powershell
python -c "from xbbg import blp; print(blp.bdp('AAPL US Equity','PX_LAST'))"
```

If that prints per-fold metrics, the environment works. Every later session:
just `cd ipo-model` and `.venv\Scripts\activate` again.

## 2. Build your data folder

Create a folder `data\` containing these files (exact column names):

### ipos.csv — from your company CSV
| column | format |
|---|---|
| `ipo_id` | any unique string (ticker is fine) |
| `first_trade_date` | YYYY-MM-DD |
| `offer_price` | number |
| `market` | market/exchange label, e.g. `US`, `HK` |
| `deal_size` | IPO proceeds (any consistent unit) |
| `is_tmt`, `is_healthcare` | 0 or 1 |
| `bk_goldman`, `bk_ms`, … | one 0/1 column per bookrunner (any names starting `bk_`) |
| `is_target` *(optional)* | 1 = model this deal; 0 = momentum-universe only |
| `bbg_ticker` *(optional)* | Bloomberg ticker if different from `ipo_id` |

`is_target=0` rows are other IPOs in the same market that you are **not**
modelling. They make the F1/F2 "recent deals" pool reflect the whole market
instead of just your sample, need only ~6 trading days of prices, and can
leave `deal_size` / sector / bookrunner columns blank. See
[docs/BLOOMBERG_DATA.md](docs/BLOOMBERG_DATA.md).

### prices.csv — from Bloomberg
One row per IPO per day, first ~30 trading days after listing:
`ipo_id,date,close`. In Excel per ticker:
`=BDH("TICKER Equity","PX_LAST",first_trade_date,first_trade_date+45,"Days=T")`
then stack the results into one long CSV (dates YYYY-MM-DD).

For 1,500+ deals, script it instead (Terminal must be running):
```powershell
pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
pip install xbbg
python scripts\pull_bloomberg.py --data data --benchmarks "HK=HSI Index,US=SPX Index" --limit 5
```
Inspect those 5 deals, then re-run without `--limit`. It also writes
`market.csv`, resumes if interrupted, and logs bad tickers rather than
stopping.

### market.csv — from Bloomberg
One benchmark index **per market**, daily over your whole sample, **starting
well before each market's first IPO**: `date,market,close` (long format —
stack the indices). One BDH per index, e.g.:
`=BDH("HSI Index","PX_LAST",start,end)` for `market=HK`,
`=BDH("SPX Index","PX_LAST",start,end)` for `market=US`.
The `market` labels must match the ones in ipos.csv. Targets become
outperformance vs this benchmark ("grew to X× what the index grew to").

### gpr.csv — from the internet
```powershell
python scripts\fetch_gpr.py --out data\gpr.csv
```
If your firewall blocks the download: open
https://www.matteoiacoviello.com/gpr.htm in a browser, download the **daily**
GPR data file, then:
```powershell
python scripts\prepare_gpr.py --in Downloads\data_gpr_daily_recent.xls --out data\gpr.csv
```

### deals.csv — optional, recommended (your follow-on data)
All deal types (IPO, follow-on, convertible) for the F3 supply factors:
`date,market,proceeds`. Without this file, supply factors use IPOs only.

## 2b. The newer factor data (in this order)

Each step is optional — features whose file is missing are simply skipped —
but run them in this order because later ones depend on earlier ones. Full
commands and Terminal specifics: [docs/BLOOMBERG_DATA.md](docs/BLOOMBERG_DATA.md).

```powershell
# 1. Syndicate block (no Bloomberg needed) — FIRST review configs\bank_classes.csv
#    and add every bank appearing in your bk_* columns (it lists unmatched names)
python scripts\build_bookrunner_features.py --file data\ipos.csv

# 2. Industry subgroup per company (Terminal running; resumable; smoke test first)
python scripts\pull_subgroups.py --file data\ipos.csv --limit 5
python scripts\pull_subgroups.py --file data\ipos.csv

# 3. Peer universe — the ONE manual Terminal step: EQS <GO>, screen your
#    markets, output ticker + Industry Subgroup (+ market, market_cap) to
#    Excel, save as data\peer_universe.csv

# 4. Peer trailing returns (resumable; writes the mandatory asof column)
python scripts\pull_peers.py --data data --k 8 --limit 5
python scripts\pull_peers.py --data data --k 8

# 5. Macro state (edit the ticker maps to your markets)
python scripts\pull_macro.py --data data --vol "HK=VHSI Index,US=VIX Index" --fx "HK=USDHKD Curncy,JP=USDJPY Curncy"
```

## 3. Check, sanity-test, then run

```powershell
python scripts\check_data.py --data data          # fix anything it flags, re-run until OK
python scripts\sanity_check.py --data data        # shuffled-label IC must be ~0

python scripts\run_baseline.py --data data --config configs\small.yaml --engine both --tune 40
python scripts\run_full.py --data data --config configs\small.yaml
python scripts\run_ablations.py --data data --config configs\small.yaml
python scripts\analyze_results.py --results results
python scripts\diagnose.py --data data --config configs\small.yaml
python scripts\make_report.py --results results   # PM-facing HTML summary
```

Results land in `results\` (`ablations.md` is the summary table). Rough
timings on a laptop CPU: baselines under a minute, the full ladder with the
small config ~30–60 min. Under the default binary head, look first at
`rank_ic`, `auc`, and `quintile_spread` — the deep rungs must beat
`0_lgbm_tuned` and `00_naive_f1` to justify themselves.

## 4. Deploy: score new deals without retraining

```powershell
python scripts\fit_final.py --data data --out results\final_model
# for each new deal: append its row to ipos.csv (no prices needed),
# refresh peers/macro, then
python scripts\predict.py --data data --bundle results\final_model --new-only
```

Optional experiment tracking: `pip install wandb`, set `WANDB_MODE=offline`
(locked-down machine), and put `train: {wandb: true}` in your YAML config;
`wandb sync` the run folder later from a machine with internet.

## Moving to another device later

Everything that matters is the git repo + your `data\` folder. On the new
machine: repeat step 1, copy `data\` over, run `check_data.py`. The `.venv\`
folder should NOT be copied — always recreate it (it hard-codes paths).
