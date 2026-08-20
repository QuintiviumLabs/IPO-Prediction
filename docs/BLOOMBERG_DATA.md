# Sourcing the data from Bloomberg

What each file is for, and how to pull it. **Verify every field mnemonic on
your terminal with `FLDS <GO>` before a big pull** — mnemonics vary by asset
class and change between Bloomberg releases.

## The key point about F1/F2

The momentum factors need **no separate data source**. F1 (median and
value-weighted recent-deal returns) and F2 (break rate, downside depth) are
computed inside the pipeline from `ipos.csv` + `prices.csv`: for each target
deal, the code takes the ≤10 most recent IPOs **in the same market** within
90 days and reads their 1-day and 1-week returns from offer price.

So the only thing that determines F1/F2 quality is **how complete your IPO
list is per market**. If your modelled deals are a filtered subset (size
threshold, sector focus, data availability), then "the 10 most recent HK
IPOs" silently means "…that survived my filter", which misreads the market.

**Fix: the `is_target` column.** Add the market's other IPOs to `ipos.csv`
with `is_target=0`. They feed F1–F4 but are never trained or tested on.
These rows are cheap:

| they need | they don't need |
|---|---|
| `ipo_id`, `first_trade_date`, `market`, `offer_price` | 21 days of prices — ~6 trading days covers 1d + 1w |
| `deal_size` (for the value-weighted leg; blank ⇒ no weight) | bookrunner columns — leave blank |
| `is_tmt` / `is_healthcare` if you want them in F4 sector density | full label history |

Your modelled deals keep `is_target=1` (or omit the column entirely if every
row is a target). `check_data.py` reports the split.

## 1. The IPO universe → `ipos.csv`

`IPO <GO>` is the IPO Center: filter by region/exchange and a date range,
set status to priced/completed, then export to Excel. `NIM <GO>` (New Issue
Monitor) covers the same ground plus follow-ons and convertibles.
Alternatively screen in `EQS <GO>` on IPO date.

Pull, per deal: ticker, first trade date, offer price, exchange/market, deal
size (proceeds), sector, bookrunners. Useful reference fields to check in
`FLDS`: `EQY_INIT_PO_DT` (IPO date), `EQY_INIT_PO_PX` (offer price) — confirm
the exact names on your terminal.

**Survivorship warning.** Build the list from a historical IPO screen, which
includes companies that later delisted or were acquired. A list derived from
*current* index membership or a current-constituent screen silently drops the
failures and will inflate every momentum factor. For delisted names, the
plain ticker may no longer resolve — capture the **FIGI / BBGID** at export
time and use it for the price pull.

Rename columns to the pipeline's schema (`ipo_id`, `first_trade_date`,
`offer_price`, `market`, `deal_size`, `is_tmt`, `is_healthcare`, `bk_*`, and
optionally `is_target`, `bbg_ticker`).

## 2. Prices → `prices.csv`

Long format `ipo_id,date,close`, roughly the first 30 trading days per deal
(targets need 21 + buffer; `is_target=0` rows need only ~6).

**Excel**, one deal at a time:
```
=BDH("0700 HK Equity","PX_LAST",first_trade_date,first_trade_date+45,"Days=T","Fill=P")
```
then stack the blocks into one long sheet with the `ipo_id` beside each.

**Scripted** (much better for 1,500+ deals) — with the Terminal running:
```powershell
pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
pip install xbbg
python scripts\pull_bloomberg.py --data data --benchmarks "HK=HSI Index,US=SPX Index" --limit 5
```
Check the output on those 5, then drop `--limit` for the full run. The script
resumes if interrupted and writes unresolvable tickers to
`bloomberg_failures.csv` instead of dying. Note terminals have a daily data
limit; ~1,500 deals × 30 days ≈ 45k points is normally well inside it, but
spread very large pulls across days if you hit a cap.

## 3. Benchmarks → `market.csv`

Long format `date,market,close` — **one index per market**, since the target
is outperformance vs each deal's own benchmark:
```
=BDH("HSI Index","PX_LAST",start,end)     ' market = HK
=BDH("SPX Index","PX_LAST",start,end)     ' market = US
```
`pull_bloomberg.py` does this from `--benchmarks`. The `market` labels must
match `ipos.csv` exactly, and each series must start before that market's
first IPO.

## 4. All deal types → `deals.csv` (optional, for F3)

`date,market,proceeds` covering IPOs **and** follow-ons and convertibles —
from `NIM <GO>`. This is where your follow-on dataset belongs: it upgrades
the supply factor to the all-deal-types version the factor spec calls for.
Without the file, F3 falls back to IPOs only.

## 5. GPR → `gpr.csv`

Not Bloomberg: `python scripts\fetch_gpr.py --out data\gpr.csv` (see
`WINDOWS_QUICKSTART.md` for the manual fallback).

## 6. Industry subgroup + peers → `subgroup` column, `peers.csv`

```powershell
python scripts\pull_subgroups.py --file data\ipos.csv     # BDP INDUSTRY_SUBGROUP
python scripts\pull_peers.py --data data --k 8            # peer returns, as-of
```
`pull_peers.py` needs `data\peer_universe.csv` (`ticker,subgroup[,market][,market_cap]`)
exported ONCE from the Terminal — an EQS screen of listed names with their
subgroup — because the API cannot enumerate subgroup members. Both scripts
resume where they stopped and also accept your `.xlsx` deal file directly.

## 7. Macro state → `macro.csv`, `macro_global.csv`

```powershell
python scripts\pull_macro.py --data data ^
    --vol "HK=VHSI Index,US=VIX Index" --fx "HK=USDHKD Curncy,JP=USDJPY Curncy"
```
Globals default to `VIX Index`, `LF98OAS Index` (US HY OAS), `MXEF Index`,
`MXWD Index` — override on the command line if your desk prefers other
series. Any market/series you skip simply drops those features.

## 8. Bookrunner syndicate block → columns in `ipos.csv`

```powershell
python scripts\build_bookrunner_features.py --file data\ipos.csv
```
No Bloomberg needed, but it reads `configs\bank_classes.csv` — a
hand-maintained map of bank → class (bb/global/regional/local) + home
region. The shipped file is a SEED: review it and add every bank that
appears in your `bk_*` columns (the script lists unmatched names).
`prestige_rank_max` is computed inside the pipeline (expanding, pre-pricing
only) — nothing to pull.

## Then

```powershell
python scripts\check_data.py --data data
```
Fix whatever it flags before running any model.
