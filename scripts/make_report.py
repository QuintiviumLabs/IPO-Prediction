#!/usr/bin/env python3
"""Build a stakeholder-facing HTML report from the results directory.

Reads whatever `run_ablations.py`, `analyze_results.py` and `diagnose.py`
produced and writes results/report.html — a self-contained page that leads
with the finding in plain language, then the evidence, then the limitations.

    python scripts\\run_ablations.py --data data --config configs\\small.yaml
    python scripts\\analyze_results.py --results results
    python scripts\\diagnose.py --data data --config configs\\small.yaml
    python scripts\\make_report.py --results results

Missing inputs degrade gracefully — sections are skipped, never faked.
"""
import argparse
import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

CSS = """
:root{--bg:#F5F7F4;--surface:#fff;--ink:#1F2B31;--muted:#5D6D72;--line:#C8D1CE;
--good:#10756A;--good-soft:#DBEDE8;--warn:#96660D;--warn-soft:#F7EEDC;--bad:#A83A28;
--display:"Avenir Next",Avenir,"Segoe UI",system-ui,sans-serif;
--body:Charter,Cambria,Georgia,serif;--mono:"SF Mono",Consolas,ui-monospace,monospace}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131A1D;
--surface:#1B2327;--ink:#E5EAE7;--muted:#92A19D;--line:#3A474B;--good:#46B4A3;
--good-soft:#17302C;--warn:#D5A554;--warn-soft:#2A2418;--bad:#DF7258}}
:root[data-theme="dark"]{--bg:#131A1D;--surface:#1B2327;--ink:#E5EAE7;--muted:#92A19D;
--line:#3A474B;--good:#46B4A3;--good-soft:#17302C;--warn:#D5A554;--warn-soft:#2A2418;--bad:#DF7258}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font-family:var(--body);
margin:0;line-height:1.6}.wrap{max-width:920px;margin:0 auto;padding:44px 22px 80px}
header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
color:var(--muted);margin:0 0 10px}h1{font-family:var(--display);font-weight:600;
font-size:clamp(27px,5vw,38px);line-height:1.1;margin:0 0 12px;text-wrap:balance}
.standfirst{font-size:17px;color:var(--muted);max-width:64ch;margin:0}
h2{font-family:var(--display);font-weight:600;font-size:21px;margin:44px 0 6px}
h3{font-family:var(--display);font-weight:600;font-size:16px;margin:26px 0 6px}
p{max-width:74ch}.lede{color:var(--muted);font-size:15.5px;margin:0 0 16px;max-width:74ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:16px 18px}
.kpi .v{font-family:var(--display);font-size:30px;font-weight:600;line-height:1.1;
font-variant-numeric:tabular-nums}.kpi .l{font-family:var(--mono);font-size:11px;
letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:6px}
.kpi .s{font-size:13px;color:var(--muted);margin-top:7px}
.kpi.hero{border-color:var(--good);border-width:2px}.kpi.hero .v{color:var(--good)}
.tablewrap{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;
font-size:14.5px;min-width:460px}th,td{text-align:left;padding:9px 12px;
border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--display);font-weight:600;font-size:12.5px;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);border-bottom:1.5px solid var(--line)}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
tr.hl td{background:var(--good-soft)}
.callout{border-left:3px solid var(--good);background:var(--good-soft);padding:13px 16px;
border-radius:0 7px 7px 0;margin:18px 0;font-size:15px}
.callout.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.callout p{margin:0;max-width:none}.callout p+p{margin-top:8px}
.callout b{font-family:var(--display)}
code{font-family:var(--mono);font-size:.87em;background:var(--surface);
border:1px solid var(--line);border-radius:4px;padding:1px 5px}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;
border-radius:11px;border:1px solid currentColor}
.tag.yes{color:var(--good)}.tag.no{color:var(--bad)}.tag.meh{color:var(--warn)}
footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);
font-family:var(--mono);font-size:12.5px;color:var(--muted);line-height:1.9}
"""


def pct(x):
    return "n/a" if not np.isfinite(x) else f"{np.expm1(x) * 100:+.1f}%"


def num(x, d=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def _read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def build(results: Path, title: str) -> str:
    abl = _read(results / "ablations.csv")
    sig = _read(results / "significance.csv")
    cmp_ = _read(results / "comparisons.csv")
    slices = _read(results / "diagnostics" / "slice_metrics.csv")
    perm = _read(results / "diagnostics" / "permutation_importance.csv")
    if abl is None and sig is None:
        raise SystemExit(f"Nothing to report on in {results}/. Run "
                         "run_ablations.py (and analyze_results.py) first.")

    P = [f'<title>{html.escape(title)}</title><style>{CSS}</style>',
         '<div class="wrap"><header>',
         f'<p class="eyebrow">Model evaluation · {date.today().isoformat()}</p>',
         f'<h1>{html.escape(title)}</h1>',
         '<p class="standfirst">Can we predict, at pricing, which IPOs will '
         'outperform their own market index over the following month? This '
         'reports what the model achieves out of sample, how it compares with '
         'simpler alternatives, and how confident we can be that the result is '
         'not chance.</p></header>']

    # ---------- pick the headline model ----------
    best_row, best_name = None, None
    if sig is not None and len(sig):
        s = sig.dropna(subset=["ic_fold_mean"])
        if len(s):
            best_row = s.sort_values("ic_fold_mean", ascending=False).iloc[0]
            best_name = best_row["rung"]
    elif abl is not None and len(abl):
        a = abl.dropna(subset=["rank_ic"])
        if len(a):
            best_row = a.sort_values("rank_ic", ascending=False).iloc[0]
            best_name = best_row.get("rung", a.index[0])

    spread = np.nan
    if abl is not None and best_name is not None:
        m = abl[abl.iloc[:, 0].astype(str) == str(best_name)]
        if len(m):
            for col in ("quintile_spread", "decile_spread"):
                if col in m.columns and np.isfinite(m.iloc[0][col]):
                    spread = float(m.iloc[0][col])
                    break

    # ---------- KPI strip ----------
    P.append('<div class="kpis">')
    if best_row is not None:
        ic = float(best_row.get("ic_fold_mean", best_row.get("rank_ic", np.nan)))
        P.append(f'<div class="kpi hero"><div class="v">{num(ic)}</div>'
                 f'<div class="l">Rank IC</div><div class="s">Correlation between '
                 f'predicted and actual ranking of deals. Above +0.05 is '
                 f'commercially meaningful in equities.</div></div>')
        if np.isfinite(spread):
            P.append(f'<div class="kpi"><div class="v">{pct(spread)}</div>'
                     f'<div class="l">Top vs bottom bucket</div><div class="s">'
                     f'Difference in realised outperformance between the deals the '
                     f'model rates best and worst.</div></div>')
        p = float(best_row.get("ic_p", np.nan))
        if np.isfinite(p):
            verdict = ("Very unlikely to be chance" if p < 0.01 else
                       "Unlikely to be chance" if p < 0.05 else
                       "NOT distinguishable from chance")
            P.append(f'<div class="kpi"><div class="v">{p:.3f}</div>'
                     f'<div class="l">p-value</div><div class="s">{verdict}. '
                     f'Tested on the quarter-by-quarter record, not one pooled '
                     f'number.</div></div>')
        n = float(best_row.get("n_deals", np.nan))
        if np.isfinite(n):
            P.append(f'<div class="kpi"><div class="v">{n:,.0f}</div>'
                     f'<div class="l">Deals scored</div><div class="s">Every one '
                     f'predicted before pricing, by a model that never saw it or '
                     f'anything after it.</div></div>')
    P.append('</div>')

    # ---------- what it means ----------
    if best_row is not None:
        ic = float(best_row.get("ic_fold_mean", best_row.get("rank_ic", np.nan)))
        P += ['<h2>What the number means</h2>',
              f'<p>The headline model is <code>{html.escape(str(best_name))}</code>. '
              f'A rank IC of {num(ic)} says that when the model ranks two deals, '
              f'the better-rated one really does perform better more often than '
              f'not. It is a measure of <em>ordering</em>, not of precision: the '
              f'model is useful for choosing between deals, not for forecasting '
              f'any single deal\'s return.</p>']
        if np.isfinite(spread):
            P.append(f'<p>In practical terms, deals in the model\'s top bucket '
                     f'outperformed those in its bottom bucket by <strong>'
                     f'{pct(spread)}</strong> relative to their benchmarks over one '
                     f'month, out of sample.</p>')

    # ---------- evidence: the ladder ----------
    if abl is not None and len(abl):
        name_col = abl.columns[0]
        P += ['<h2>Compared with simpler alternatives</h2>',
              '<p class="lede">A good score means nothing on its own — the '
              'question is whether it beats what you would get without the '
              'model. Each row sees the same deals, the same features and the '
              'same time-ordered testing.</p>',
              '<div class="tablewrap"><table><thead><tr><th>Approach</th>'
              '<th>Rank IC</th><th>Consistency (± across periods)</th>'
              '<th>Top-vs-bottom</th></tr></thead><tbody>']
        sc = "quintile_spread" if "quintile_spread" in abl.columns else "decile_spread"
        for _, r in abl.iterrows():
            nm = str(r[name_col])
            hl = ' class="hl"' if nm == str(best_name) else ""
            P.append(f'<tr{hl}><td>{html.escape(_friendly(nm))}</td>'
                     f'<td class="n">{num(r.get("rank_ic", np.nan))}</td>'
                     f'<td class="n">{num(r.get("rank_ic_std", np.nan), 3)}</td>'
                     f'<td class="n">{pct(r.get(sc, np.nan))}</td></tr>')
        P.append('</tbody></table></div>')
        P.append('<div class="callout"><p><b>Why the comparison rows matter.</b> '
                 'The naive rows are deliberate floors: one predicts the same '
                 'number for every deal, another uses only how recent IPOs in '
                 'that market performed. The gradient-boosted rows are '
                 'conventional machine learning on the same inputs, with '
                 'hyperparameters searched on held-out data so the comparison is '
                 'not rigged in the model\'s favour.</p></div>')

    # ---------- evidence: paired tests ----------
    if cmp_ is not None and len(cmp_):
        base = str(cmp_.iloc[0].get("model_b", "the baseline"))
        P += ['<h2>Is the improvement real?</h2>',
              f'<p class="lede">Each approach tested against <code>'
              f'{html.escape(base)}</code> on <em>identical deals</em>, which is '
              f'far more sensitive than comparing two separate averages. '
              f'"Adjusted p" accounts for the fact that testing many approaches '
              f'gives many chances at a fluke.</p>',
              '<div class="tablewrap"><table><thead><tr><th>Approach</th>'
              '<th>Improvement in rank IC</th><th>95% range</th>'
              '<th>Adjusted p</th><th>Verdict</th></tr></thead><tbody>']
        for _, r in cmp_.iterrows():
            q = float(r.get("delta_q_value", np.nan))
            better = float(r.get("rank_ic_delta", 0)) > 0
            if np.isfinite(q) and q < 0.05 and better:
                tag = '<span class="tag yes">holds up</span>'
            elif np.isfinite(q) and q < 0.05:
                tag = '<span class="tag no">worse</span>'
            else:
                tag = '<span class="tag meh">not proven</span>'
            P.append(f'<tr><td>{html.escape(_friendly(str(r["model_a"])))}</td>'
                     f'<td class="n">{num(r.get("rank_ic_delta", np.nan))}</td>'
                     f'<td class="n">[{num(r.get("rank_ic_delta_lo", np.nan))}, '
                     f'{num(r.get("rank_ic_delta_hi", np.nan))}]</td>'
                     f'<td class="n">{q:.4f}</td><td>{tag}</td></tr>')
        P.append('</tbody></table></div>')

    # ---------- where it works ----------
    if slices is not None and len(slices):
        P += ['<h2>Where it works, and where it does not</h2>',
              '<p class="lede">The same predictions, split by market, sector and '
              'risk environment. Uneven performance is normal; it tells you where '
              'to trust the model and where to apply judgement.</p>',
              '<div class="tablewrap"><table><thead><tr><th>Segment</th>'
              '<th>Deals</th><th>Rank IC</th><th>Top-vs-bottom</th></tr>'
              '</thead><tbody>']
        sc = ("quintile_spread" if "quintile_spread" in slices.columns else None)
        for _, r in slices.sort_values("rank_ic", ascending=False).iterrows():
            sv = pct(r[sc]) if sc else "—"
            P.append(f'<tr><td>{html.escape(_friendly(str(r["slice"])))}</td>'
                     f'<td class="n">{r["n"]:.0f}</td>'
                     f'<td class="n">{num(r["rank_ic"])}</td>'
                     f'<td class="n">{sv}</td></tr>')
        P.append('</tbody></table></div>')

    # ---------- what drives it ----------
    if perm is not None and len(perm):
        top = perm.sort_values("delta_ic", ascending=False).head(8)
        P += ['<h2>What the model is actually using</h2>',
              '<p class="lede">Measured by scrambling each input and seeing how '
              'much accuracy is lost. A negative value means the input is noise '
              'the model would be better off without.</p>',
              '<div class="tablewrap"><table><thead><tr><th>Input</th>'
              '<th>Accuracy lost when scrambled</th></tr></thead><tbody>']
        for _, r in top.iterrows():
            P.append(f'<tr><td>{html.escape(_friendly(str(r["feature"])))}</td>'
                     f'<td class="n">{num(r["delta_ic"])}</td></tr>')
        P.append('</tbody></table></div>')

    # ---------- limitations ----------
    P += ['<h2>What this does not show</h2>',
          '<div class="callout warn"><p><b>Read these before acting on the '
          'number.</b></p>',
          '<p><b>Ordering, not returns.</b> The model ranks deals; it does not '
          'forecast how much any one will make. Individual errors are large and '
          'always will be — IPO outcomes are dominated by deal-specific noise.</p>',
          '<p><b>Sample size.</b> The evidence rests on roughly a thousand deals '
          'over a limited window. It has not been tested through every kind of '
          'market, and performance in an unseen regime is not guaranteed.</p>',
          '<p><b>Survivorship.</b> Deals delisting before the one-month mark have '
          'no measurable outcome and drop out. That removes some of the worst '
          'cases, which flatters every model here equally.</p>',
          '<p><b>No trading costs.</b> Bucket spreads are gross of allocation '
          'constraints, fees and the practical difficulty of getting IPO '
          'allocations at scale.</p></div>']

    P += ['<footer>Generated by <code>scripts/make_report.py</code> from '
          f'<code>{html.escape(str(results))}/</code>. Every figure is '
          'out-of-sample under purged walk-forward validation: each model is '
          'trained only on deals priced before the ones it is scored on, with an '
          'embargo so no overlapping outcome leaks backwards.</footer></div>']
    return "\n".join(P)


FRIENDLY = {
    "00_naive_const": "Naive: same prediction for every deal",
    "00_naive_f1": "Naive: recent IPO performance only",
    "0_lgbm": "Gradient boosting (LightGBM)",
    "0_xgb": "Gradient boosting (XGBoost)",
    "0_lgbm_tuned": "Gradient boosting, tuned",
    "0_xgb_tuned": "Gradient boosting, tuned (XGBoost)",
    "0_lgbm_raw": "Gradient boosting + raw risk history",
    "0_xgb_raw": "Gradient boosting + raw risk history (XGBoost)",
    "1_static": "Deal characteristics only",
    "2_static+f1": "+ recent IPO performance",
    "3_static+momentum": "+ full market-state factors",
    "4_momentum+gpr_level": "+ geopolitical risk level",
    "5_full_v1": "Full model",
    "6_full_v2_film": "Full model + risk-regime gating",
    "full_v1": "Full model",
    "full_v2_film": "Full model + risk-regime gating",
    "x_ridge": "Regularized linear model",
    "x_lasso": "Regularized linear model (Lasso)",
    "x_tabpfn": "TabPFN (pretrained tabular model)",
    "gpr": "Geopolitical risk",
    "bookrunners": "Bookrunner identity",
    "n_bookrunners": "Number of bookrunners",
    "is_tmt": "TMT sector", "is_healthcare": "Healthcare sector",
}


def _friendly(name: str) -> str:
    if name in FRIENDLY:
        return FRIENDLY[name]
    s = name
    for a, b in (("f1_", "Recent IPO perf: "), ("f2_", "Break rate: "),
                 ("f3_", "Issuance supply: "), ("m_", "Macro: "),
                 ("p1_", "Sector peers: "), ("archetype_", "Syndicate type: "),
                 ("gpr_", "Geopolitical risk: "), ("market=", "Market "),
                 ("sector=", "Sector "), ("momentum=", "Momentum "),
                 ("gpr=", "Risk environment ")):
        if s.startswith(a):     # prefixes only — "m_" must not hit "geo_mom_"
            s = b + s[len(a):]
            break
    return s.replace("_", " ")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--title", default="IPO Outperformance Model")
    p.add_argument("--out", default=None, help="default: <results>/report.html")
    args = p.parse_args()
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "report.html"
    out.write_text(build(results, args.title), encoding="utf-8")
    print(f"Wrote {out}\nOpen it in a browser, or publish it as an artifact.")


if __name__ == "__main__":
    main()
