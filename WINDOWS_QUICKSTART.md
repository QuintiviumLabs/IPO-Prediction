# Running this on a Windows laptop (with Bloomberg)

The pipeline is plain Python — no databases, no services. You give it **four
CSV files** in one folder, it does everything else. Moving machines = clone
the repo + recreate the venv + copy your `data/` folder. Nothing else to
migrate.

## How the repo works, in one paragraph

`ipo_model/data/features.py` reads the four CSVs and builds, for every IPO,
only information observable before its pricing date: the F1–F4 market-state
factors, a GPR window, and the 1d/3d/1w/1m return targets.
`ipo_model/data/splits.py` cuts time-ordered train/test folds with the
overlap purged so no future leaks. Then two model tracks run on identical
folds: gradient-boosted trees on the engineered features
(`scripts/run_baseline.py` — the bar to clear) and the three-arm neural net
(`scripts/run_full.py`). `scripts/run_ablations.py` runs the whole ladder and
writes a results table; `scripts/diagnose.py` tells you which features matter
and where the model fails. `configs/small.yaml` has the right settings for
~1.5k rows.

## 1. One-time setup

1. Install **Python 3.11** from https://www.python.org/downloads/ — tick
   **"Add python.exe to PATH"** during install.
2. Install **Git** from https://git-scm.com/download/win (or use
   GitHub → Code → Download ZIP and unzip it).
3. In **PowerShell**, in the folder where you want the project:

```powershell
git clone https://github.com/QuintiviumLabs/Test.git ipo-model
cd ipo-model
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell refuses to run the activate script, run this once and retry:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Sanity check the install (uses generated fake data, ~2 min):

```powershell
python scripts\make_synthetic_data.py --out demo_data --n-ipos 400
python scripts\run_baseline.py --data demo_data
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

### prices.csv — from Bloomberg
One row per IPO per day, first ~30 trading days after listing:
`ipo_id,date,close`. In Excel per ticker:
`=BDH("TICKER Equity","PX_LAST",first_trade_date,first_trade_date+45,"Days=T")`
then stack the results into one long CSV (dates YYYY-MM-DD).

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
All deal types (IPO, follow-on, convertible) for the supply factors:
`date,market,sector,proceeds` with sector one of `tmt` / `healthcare` /
`other`. Without this file, supply factors use IPOs only.

## 3. Check, then run

```powershell
python scripts\check_data.py --data data          # fix anything it flags
python scripts\run_baseline.py --data data --config configs\small.yaml --engine both
python scripts\run_ablations.py --data data --config configs\small.yaml
python scripts\diagnose.py --data data --config configs\small.yaml
```

Results land in `results\` (`ablations.md` is the summary table). Rough
timings on a laptop CPU: baselines under a minute, the full ladder with the
small config ~30–60 min. Metrics to look at first: `rank_ic` and
`decile_spread` — the deep rungs must beat rung `0_lgbm` to justify
themselves.

## Moving to another device later

Everything that matters is the git repo + your `data\` folder. On the new
machine: repeat step 1, copy `data\` over, run `check_data.py`. The `.venv\`
folder should NOT be copied — always recreate it (it hard-codes paths).
