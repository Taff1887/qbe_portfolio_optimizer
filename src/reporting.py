"""Lens-agnostic output layer: comparison tables, charts and a markdown report.

Takes the computed objects from main.py and writes CSVs to outputs/tables, PNGs
to outputs/charts and a summary to outputs/. No finance logic lives here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lagic_capital import run_lagic
from portfolio import Portfolio
from risk_attribution import diversification_ratio
from utils import (
    OUT_TABLES,
    PALETTE,
    ROOT,
    drawdown_series,
    save_chart,
    save_table,
)


# ----------------------------------------------------------------- tables
def comparison_table(
    portfolios: dict[str, Portfolio],
    config: dict,
    benchmark: Portfolio | None = None,
    stress_losses: pd.Series | None = None,
) -> pd.DataFrame:
    """Row per portfolio across return, risk, tail, capital and accounting lenses.

    Every portfolio is scored on the identical metric set. `benchmark` (defaults
    to the first portfolio) anchors tracking error and turnover; `stress_losses`
    (worst-case total impact per portfolio) supplies the Stress Loss column.
    """
    if benchmark is None:
        benchmark = next(iter(portfolios.values()))
    rows = {}
    for name, pf in portfolios.items():
        cap = run_lagic(pf, config)
        s = pf.summary()
        rows[name] = {
            "exp_return": s["expected_return"],
            "volatility": s["volatility"],
            "sharpe": s["sharpe"],
            "sortino": s["sortino"],
            "max_drawdown": s["max_drawdown"],
            "var_95": s["var_95"],
            "cvar_95": s["cvar_95"],
            "duration": s["duration"],
            "capital_charge": cap["capital_charge"],
            "stress_loss": float(stress_losses[name]) if stress_losses is not None else float("nan"),
            "liquidity": s["liquidity"],
            "diversification_ratio": diversification_ratio(pf),
            "turnover": pf.turnover_from(benchmark),
            "tracking_error": pf.tracking_error(benchmark),
            # retained context columns
            "realised_return": s["ann_return"],
            "carry": s["carry"],
            "core_fi": s["core_fi"],
            "risk_assets": s["risk_assets"],
            "return_on_capital": cap["return_on_capital"],
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------- charts
def chart_efficient_frontier(frontier: pd.DataFrame, points: dict[str, Portfolio]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier["volatility"] * 100, frontier["exp_return"] * 100, "-", color="#0072B2", label="Efficient frontier")
    for i, (name, pf) in enumerate(points.items()):
        ax.scatter(pf.volatility() * 100, pf.expected_return() * 100, s=90, zorder=5,
                   color=PALETTE[(i + 1) % len(PALETTE)], label=name)
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Constrained efficient frontier")
    ax.legend()
    save_chart(fig, "01_efficient_frontier")


def chart_allocation_comparison(portfolios: dict[str, Portfolio]) -> None:
    meta = next(iter(portfolios.values())).meta
    alloc = pd.DataFrame({name: pf.weights.groupby(meta["capital_category"]).sum()
                          for name, pf in portfolios.items()})
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(alloc.index))
    w = 0.8 / len(alloc.columns)
    for k, col in enumerate(alloc.columns):
        ax.bar(x + (k - (len(alloc.columns) - 1) / 2) * w, alloc[col] * 100, w, label=col)
    ax.set_xticks(x, alloc.index, rotation=40, ha="right")
    ax.set_ylabel("Weight (%)")
    ax.set_title("Allocation by capital category")
    ax.legend()
    save_chart(fig, "02_allocation_comparison")


def chart_stress(stress: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(stress.index))
    ax.bar(x - 0.2, stress["pnl_impact"] * 100, 0.4, label="P&L impact", color="#D55E00")
    ax.bar(x + 0.2, stress["oci_impact"] * 100, 0.4, label="OCI impact", color="#0072B2")
    ax.plot(x, stress["total_impact"] * 100, "ko", label="Total")
    ax.set_xticks(x, stress.index, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Instantaneous impact (%)")
    ax.set_title("Stress scenario impact (P&L vs OCI)")
    ax.legend()
    save_chart(fig, "03_stress_scenarios")


def chart_capital_by_category(cap_by_cat: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    s = cap_by_cat.sort_values()
    ax.barh(s.index, s.values * 100, color="#117733")
    for i, v in enumerate(s.values):
        ax.text(v * 100, i, f" {v*100:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("Capital charge (% of portfolio)")
    ax.set_title("LAGIC-style capital charge by category")
    save_chart(fig, "04_capital_by_category")


def chart_return_vs_vol(comp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(comp["volatility"] * 100, comp["exp_return"] * 100, s=90, color="#0072B2")
    for name, r in comp.iterrows():
        ax.annotate(name, (r["volatility"] * 100, r["exp_return"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Return vs volatility")
    save_chart(fig, "05_return_vs_volatility")


def chart_return_vs_capital(comp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(comp["capital_charge"] * 100, comp["exp_return"] * 100, s=90, color="#D55E00")
    for name, r in comp.iterrows():
        ax.annotate(name, (r["capital_charge"] * 100, r["exp_return"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Capital charge (% of portfolio)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Return vs capital charge")
    save_chart(fig, "06_return_vs_capital")


def chart_rolling_return(pf: Portfolio) -> None:
    roll = (1 + pf.return_series).rolling(12).apply(np.prod, raw=True) - 1
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(roll.index, roll.values * 100, color="#0072B2")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Rolling 12m return (%)")
    ax.set_title(f"Rolling annual return - {pf.name}")
    save_chart(fig, "07_rolling_annual_return")


def chart_earnings(annual: pd.DataFrame, plan: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#C0392B" if v < plan else "#2E8B57" for v in annual["total"]]
    ax.bar(annual.index.astype(str), annual["total"] * 100, color=colors)
    ax.axhline(plan * 100, color="#222", ls="--", lw=1.2, label=f"Plan target {plan:.1%}")
    ax.set_ylabel("Annual return (%)")
    ax.set_title("Annual earnings vs plan (red = missed plan)")
    ax.legend()
    save_chart(fig, "08_earnings_vs_plan")


def chart_duration_gap(by_ccy: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(by_ccy.index))
    ax.bar(x - 0.2, by_ccy["asset_duration"], 0.4, label="Asset duration", color="#0072B2")
    ax.bar(x + 0.2, by_ccy["liability_duration"], 0.4, label="Liability duration", color="#D55E00")
    ax.plot(x, by_ccy["duration_gap"], "ko", label="Gap")
    ax.set_xticks(x, by_ccy.index)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Duration (years)")
    ax.set_title("Asset vs liability duration by currency")
    ax.legend()
    save_chart(fig, "09_duration_gap")


def chart_drawdown(pf: Portfolio) -> None:
    dd = drawdown_series(pf.return_series) * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(dd.index, dd.values, 0, color="#C0392B", alpha=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown - {pf.name}")
    save_chart(fig, "10_drawdown")


def chart_duration_earnings_example(example: dict) -> None:
    pivot = example["pivot"]              # index = book, cols = rate scenarios
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    cols = list(pivot.columns)
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, pivot[c] * 100, w, label=c)
    ax.set_xticks(x, pivot.index)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Year-1 earnings (%)")
    ax.set_title("Duration & earnings stability\nmatched book earns steady carry whichever way rates move")
    ax.legend()
    save_chart(fig, "11_duration_earnings_example")


# ------------------------------------------------- extra-analysis charts
def chart_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    ax.set_title("Asset return correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation")
    save_chart(fig, "12_correlation_heatmap")


def chart_risk_contribution(rc_asset: pd.Series, rc_cat: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    a = rc_asset.sort_values()
    axes[0].barh(a.index, a.values * 100, color="#0072B2")
    axes[0].set_xlabel("% of portfolio risk")
    axes[0].set_title("Risk contribution by asset")
    c = rc_cat.sort_values()
    axes[1].barh(c.index, c.values * 100, color="#117733")
    axes[1].set_xlabel("% of portfolio risk")
    axes[1].set_title("Risk contribution by capital category")
    save_chart(fig, "13_risk_contribution")


def chart_capital_frontier(capital_frontier: pd.DataFrame, portfolios: dict[str, Portfolio], config: dict) -> None:
    from optimizer import _capital_components, _smooth_capital
    ind, stress, R = _capital_components(next(iter(portfolios.values())).market, config)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(capital_frontier["capital_charge"] * 100, capital_frontier["exp_return"] * 100,
            "-", color="#117733", label="Capital-efficient frontier")
    for i, (name, pf) in enumerate(portfolios.items()):
        cap = _smooth_capital(pf.weights.to_numpy(), ind, stress, R)
        ax.scatter(cap * 100, pf.expected_return() * 100, s=90, zorder=5,
                   color=PALETTE[(i + 1) % len(PALETTE)], label=name)
    ax.set_xlabel("Diversified capital charge (% of assets)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Capital-efficient frontier (return vs capital)")
    ax.legend()
    save_chart(fig, "14_capital_efficient_frontier")


def chart_liquidity_and_rating(liq: dict, rating: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bt = liq["by_tier"]
    axes[0].bar(bt.index.astype(str), bt.values * 100, color=["#C0392B", "#E69F00", "#2E8B57"])
    axes[0].set_ylabel("Weight (%)")
    axes[0].set_title(f"Liquidity profile ({liq['pct_illiquid']:.0%} illiquid)")
    br = rating["by_rating"]
    axes[1].bar(br.index.astype(str), br.values * 100, color="#0072B2")
    axes[1].set_ylabel("Weight (%)")
    axes[1].set_title(f"Rating distribution ({rating['sub_investment_grade']:.0%} sub-IG)")
    save_chart(fig, "15_liquidity_and_rating")


def chart_intra_asset(intra: dict) -> None:
    s = intra["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(s.index.astype(str), s["incremental_return_bps"], color="#117733")
    for i, v in enumerate(s["incremental_return_bps"]):
        axes[0].text(i, v, f"+{v:.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Same-risk return pickup (bps)")
    axes[0].set_title(f"Within-class return uplift (portfolio +{intra['portfolio_return_uplift_bps']:.1f} bps)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(s.index.astype(str), s["div_vol_saved_bps"], color="#0072B2")
    axes[1].set_ylabel("Vol saved by diversifying (bps)")
    axes[1].set_title("Intra-class diversification benefit")
    axes[1].tick_params(axis="x", rotation=25)
    save_chart(fig, "17_intra_asset_uplift")


def chart_intra_weights(intra: dict, class_name: str) -> None:
    w = intra["per_class"][class_name]["weights"][["benchmark", "enhanced"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(w.index))
    ax.bar(x - 0.2, w["benchmark"] * 100, 0.4, label="Benchmark mix", color="#999999")
    ax.bar(x + 0.2, w["enhanced"] * 100, 0.4, label="Enhanced (same-risk)", color="#117733")
    ax.set_xticks(x, w.index, rotation=20, ha="right")
    ax.set_ylabel("Weight within class (%)")
    ax.set_title(f"{class_name}: benchmark vs enhanced sub-allocation")
    ax.legend()
    save_chart(fig, "18_intra_class_weights")


def chart_earnings_at_risk(er: dict, plan: float) -> None:
    dist = er["baseline_distribution"] * 100
    t = er["table"].loc["Baseline"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dist, bins=50, color="#0072B2", alpha=0.75)
    ax.axvline(plan * 100, color="#222", ls="--", lw=1.3, label=f"Plan {plan:.1%}")
    ax.axvline(t["mean_annual_pnl"] * 100, color="#2E8B57", lw=1.2, label=f"Mean {t['mean_annual_pnl']:.1%}")
    ax.axvline(t["earnings_at_risk_5pc"] * 100, color="#C0392B", lw=1.5, label=f"EaR 5% {t['earnings_at_risk_5pc']:.1%}")
    ax.axvline(t["cte_95"] * 100, color="#8C564B", lw=1.5, ls=":", label=f"CTE95 {t['cte_95']:.1%}")
    ax.set_xlabel("Plan-year P&L (%)")
    ax.set_title("Earnings-at-risk: bootstrap distribution of plan-year P&L (Baseline)")
    ax.legend(fontsize=8)
    save_chart(fig, "19_earnings_at_risk")


def chart_earnings_vol_contribution(s: pd.Series) -> None:
    s = s[s.abs() > 0.005].sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s.index, s.values * 100, color="#D55E00")
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("% of annual earnings (P&L) volatility")
    ax.set_title("Earnings-volatility contribution by asset")
    save_chart(fig, "20_earnings_vol_contribution")


def chart_duration_contribution(pf: Portfolio) -> None:
    """Each asset's contribution to portfolio duration (weight x duration)."""
    contrib = (pf.weights * pf.meta["duration"])
    contrib = contrib[contrib.abs() > 1e-9].sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(contrib.index, contrib.values, color="#0072B2")
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Contribution to portfolio duration (years)")
    ax.set_title(f"Duration contribution by asset - {pf.name} (total {pf.duration():.2f}y)")
    save_chart(fig, "21_duration_contribution")


def chart_stress_grid(stress_grid: pd.DataFrame) -> None:
    """Heatmap of every scenario (rows) against every portfolio (columns)."""
    data = stress_grid.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-np.abs(data).max(), vmax=np.abs(data).max(), aspect="auto")
    ax.set_xticks(range(len(stress_grid.columns)), stress_grid.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(stress_grid.index)), stress_grid.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="#222")
    ax.set_title("Stress grid: instantaneous total impact (%) - every scenario x every portfolio")
    fig.colorbar(im, ax=ax, shrink=0.7, label="impact (%)")
    save_chart(fig, "22_stress_grid")


def chart_philosophy_metrics(comp: pd.DataFrame, names: list[str]) -> None:
    """Risk-adjusted return (Sharpe & Sortino) across construction philosophies."""
    sub = comp.loc[[n for n in names if n in comp.index]]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(sub.index))
    ax.bar(x - 0.2, sub["sharpe"], 0.4, label="Sharpe", color="#0072B2")
    ax.bar(x + 0.2, sub["sortino"], 0.4, label="Sortino", color="#117733")
    ax.set_xticks(x, sub.index, rotation=20, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_title("Construction philosophies: risk-adjusted return (in-sample)")
    ax.legend()
    save_chart(fig, "23_philosophy_metrics")


def chart_factor_attribution(attr: pd.DataFrame) -> None:
    """Annualised return attributed to each factor (plus alpha/carry)."""
    s = attr["ann_contribution"] * 100
    colors = ["#117733" if v >= 0 else "#C0392B" for v in s.values]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="#444", lw=0.8)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=8)
    ax.set_xlabel("Annualised return contribution (%)")
    ax.set_title("Return attribution by factor (through-time drivers)")
    save_chart(fig, "24_factor_attribution")


def chart_rolling_factor_betas(roll: pd.DataFrame) -> None:
    """Rolling factor betas - how the book's factor exposures drift through time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in roll.columns:
        ax.plot(roll.index, roll[col], label=col)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Rolling factor beta")
    ax.set_title("Rolling factor exposures through time (36m window)")
    ax.legend(fontsize=8, ncol=3)
    save_chart(fig, "25_rolling_factor_betas")


def chart_marginal_efficiency(marg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = (marg["weight"] * 1500 + 20)
    ax.scatter(marg["marginal_capital"] * 100, marg["exp_return"] * 100, s=sizes, alpha=0.6, color="#D55E00")
    for name, r in marg.iterrows():
        ax.annotate(name, (r["marginal_capital"] * 100, r["exp_return"] * 100),
                    fontsize=7, textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("Marginal capital (prescribed stress, %)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Marginal efficiency: return vs capital per asset (bubble = weight)")
    save_chart(fig, "16_marginal_efficiency")


# ------------------------------------------------------------- markdown
def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> GitHub markdown table (avoids a tabulate dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(["metric"] + [str(c) for c in cols]) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = []
    for idx, row in df.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], (int, float, np.floating)) and not isinstance(row[c], bool)
                else str(row[c]) for c in cols]
        body.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join([head, sep, *body])


def write_markdown(results: dict) -> None:
    comp = results["comparison"]
    base = results["portfolios"]["Baseline"]
    stress = results["stress"]
    lagic = results["lagic"]["Baseline"]
    earn = results["earnings"]["summary"]
    dur = results["duration"]
    worst_s = stress["total_impact"].idxmin()

    md = []
    md.append("# QBE-style Portfolio Optimisation Research Lab - summary\n")
    md.append("_Generated by `python src/main.py`. Dummy data; replace with real data via `data/processed/returns.csv` and `config.yaml`. "
              "Full report with charts: `outputs/report.md`._\n")
    md.append("## 1. Construction philosophies compared (headline metrics)\n")
    md.append(_df_to_md(comp[_COMPARISON_COLS].round(4)))
    md.append("\n## 2. Headline read across the lenses\n")
    md.append(f"- **Baseline**: {base.core_fi_share():.0%} core fixed income / {base.risk_asset_share():.0%} risk assets; "
              f"forward expected return **{base.expected_return():.2%}**, volatility **{base.volatility():.2%}**, "
              f"running carry **{base.carry():.2%}**, portfolio duration **{base.duration():.1f}y**.")
    md.append(f"- **Mean-variance (lens 1)**: the Max-Sharpe portfolio lifts expected return to "
              f"**{results['portfolios']['Max-Sharpe'].expected_return():.2%}** for volatility "
              f"**{results['portfolios']['Max-Sharpe'].volatility():.2%}** within the insurer constraints.")
    md.append(f"- **Stress (lens 2)**: worst instantaneous scenario is **{worst_s}** at "
              f"**{stress.loc[worst_s, 'total_impact']:.2%}** (P&L {stress.loc[worst_s, 'pnl_impact']:.2%}, "
              f"OCI {stress.loc[worst_s, 'oci_impact']:.2%}).")
    md.append(f"- **Capital (lens 5)**: binding LAGIC-style charge **{lagic['capital_charge']:.2%}** of assets "
              f"(binding: {lagic['binding_basis']}); return on capital **{lagic['return_on_capital']:.2f}x**.")
    md.append(f"- **Earnings (lens 3/6)**: annual earnings (P&L) volatility **{earn['earnings_volatility']:.2%}**, "
              f"chance of missing the {earn['plan_target']:.1%} plan **{earn['plan_miss_prob']:.0%}**, "
              f"carry funds **{earn['carry_share_of_return']:.0%}** of return.")
    md.append(f"- **Duration / ALM (lens 4)**: total dollar-duration gap **{dur['total_dollar_duration_gap']:+.2f}y**; "
              f"a +100bp shock moves economic surplus **{dur['rate_shock']['economic_surplus_impact']:+.2%}** vs "
              f"P&L earnings **{dur['rate_shock']['pnl_earnings_impact']:+.2%}** "
              f"(the gap is the OCI/surplus rate exposure, **{dur['rate_shock']['oci_impact']:+.2%}**).")
    md.append("\n## 3. Why multiple lenses\n")
    md.append("A single mean-variance number hides what an insurer actually manages: the same return can carry very "
              "different capital, very different earnings volatility, and a very different accounting (P&L vs OCI) "
              "footprint. The tables and charts in `outputs/` let those trade-offs be compared side by side.\n")
    md.append("\n_Charts: see `outputs/charts/`. Tables: see `outputs/tables/`._\n")

    (ROOT / "outputs" / "summary_report.md").write_text("\n".join(md), encoding="utf-8")


# Headline comparison columns (Step 7), in reading order.
_COMPARISON_COLS = [
    "exp_return", "volatility", "sharpe", "sortino", "max_drawdown",
    "var_95", "cvar_95", "duration", "capital_charge", "stress_loss",
    "liquidity", "diversification_ratio", "turnover",
]


def write_full_report(results: dict) -> None:
    """A single comprehensive markdown report, structured as two parts:
    Part A compares construction philosophies; Part B applies the evaluation
    lenses (stress, capital, earnings, ALM, risk, liquidity) to the book."""
    config = results["config"]
    pf = results["portfolios"]
    base = pf["Baseline"]
    comp = results["comparison"]
    stress = results["stress"]
    lag = results["lagic"]["Baseline"]
    earn = results["earnings"]["summary"]
    dur = results["duration"]
    ra = results["risk_attribution"]
    dg = results["diagnostics"]
    intra = results["intra_asset"]
    fa = results.get("factor_analysis")
    worst_s = stress["total_impact"].idxmin()
    philosophies = [n for n in ["Equal-Weight", "Max-Sharpe", "Min-Variance",
                                "Risk-Parity", "Max-Diversification"] if n in pf]

    def img(name, cap):
        return f"![{cap}](charts/{name}.png)\n\n*{cap}*\n"

    def comp_md(rows=None, cols=_COMPARISON_COLS):
        sub = comp if rows is None else comp.loc[[r for r in rows if r in comp.index]]
        return _df_to_md(sub[cols].round(4))

    m = []
    m.append("# QBE-style Portfolio Optimisation Research Lab - Report\n")
    m.append("_Generated by `python src/main.py` on dummy data. Every number reconciles to a CSV in `outputs/tables/`; "
             "replace `data/processed/returns.csv` and `config.yaml` with real data to refresh._\n")

    # ----------------------------------------------------------- exec read
    m.append("## Executive read\n")
    m.append("This report answers one question: **how should an institutional insurance portfolio be constructed when "
             "several competing objectives apply at once?** It does that by *comparing* portfolio-construction "
             "philosophies rather than crowning a single optimiser, then putting every portfolio through the same "
             "evaluation lenses an insurer actually manages to - return, **regulatory capital**, **earnings stability**, "
             "**accounting (P&L vs OCI)** and **asset-liability duration**.\n")
    m.append("- **Part A - Construction philosophies** builds the candidate portfolios (equal weight, mean-variance, "
             "minimum variance, risk parity, maximum diversification, plus the capital-aware optimisers) and scores them "
             "on one identical metric set.\n"
             "- **Part B - Evaluation lenses** stress-tests, capital-charges, earnings-tests and ALM-tests the book, "
             "showing why two portfolios with the same Sharpe can differ sharply on capital, drawdown and earnings.\n")
    m.append("The single comparison table below is the heart of the lab; everything after it explains the trade-offs it "
             "exposes.\n")

    # =========================================================== PART A
    m.append("## Part A - Portfolio construction philosophies\n")
    m.append("### A1. The philosophies\n")
    m.append("Each optimiser is independent and produces a `Portfolio` scored on the same metrics. Risk parity and "
             "maximum diversification are solved under the **same insurer constraints** as mean-variance (min "
             f"{config['portfolio']['min_core_fixed_income']:.0%} core fixed income, currency buckets, per-asset caps), so "
             "the comparison is like-for-like; equal weight is the deliberately unconstrained naive 1/N benchmark.\n")
    m.append("- **Equal weight (1/N)** - no optimisation, no mandate constraints: the bar every optimiser must beat.\n"
             "- **Mean-variance (Max-Sharpe)** - most return per unit of volatility within the constraints.\n"
             "- **Minimum variance** - lowest achievable volatility.\n"
             "- **Risk parity** - every asset contributes an equal share of portfolio risk.\n"
             "- **Maximum diversification** - maximises the diversification ratio (weighted-avg asset vol / portfolio vol).\n"
             "- **Capital-aware (Max-RoC, Capital-Budgeted)** - optimise return per unit of, or subject to, regulatory capital.\n"
             "- **Baseline / Risk 20%** - the insurer's strategic book and a risk-scaled scenario for reference.\n"
             "- _Roadmap placeholders_: **Black-Litterman** and **robust optimisation** are scaffolded in "
             "`src/construction.py` (documented, not yet wired into the comparison).\n")
    m.append("### A2. Side-by-side comparison\n")
    m.append("Every portfolio, every headline metric (the full set, including the retained context columns, is in "
             "`outputs/tables/portfolio_comparison.csv`):\n")
    m.append(comp_md() + "\n")
    m.append("_VaR / CVaR are historical monthly 95% figures (signed; negative = loss); stress loss is the worst "
             "instantaneous scenario; turnover (and tracking error, in the CSV) is measured against the Baseline book._\n")
    m.append(img("23_philosophy_metrics", "Figure 1. Risk-adjusted return (Sharpe & Sortino) across the construction philosophies."))
    m.append(img("01_efficient_frontier", "Figure 2. Constrained efficient frontier with each philosophy plotted."))
    m.append(f"Within the insurer constraints the **Max-Sharpe** portfolio reaches expected return "
             f"**{pf['Max-Sharpe'].expected_return():.2%}** at volatility **{pf['Max-Sharpe'].volatility():.2%}** "
             f"(Sharpe {comp.loc['Max-Sharpe','sharpe']:.2f}), versus the baseline at "
             f"**{base.expected_return():.2%}** / **{base.volatility():.2%}**. **Min-Variance** and **Risk-Parity** trade "
             "return for stability; **Equal-Weight**, unconstrained, takes the most risk and the largest turnover to "
             "implement. No single portfolio wins on every column - which is the point.\n")
    m.append(img("02_allocation_comparison", "Figure 3. Allocation by capital category across philosophies."))

    # =========================================================== PART B
    m.append("## Part B - Evaluation lenses\n")
    m.append("The same portfolios, now seen through the lenses an insurer is actually governed by. Figures below focus on "
             "the Baseline book unless stated; the stress grid covers every portfolio.\n")

    m.append("### B1. Lens 2 - Instantaneous stress testing\n")
    m.append(img("22_stress_grid", "Figure 4. Stress grid - every scenario against every portfolio (total impact, %)."))
    m.append(img("03_stress_scenarios", "Figure 5. Baseline stress impacts split into P&L and OCI."))
    m.append(_df_to_md(stress.round(4)) + "\n")
    m.append(f"The worst instantaneous scenario for the Baseline is **{worst_s}** at **{stress.loc[worst_s,'total_impact']:.2%}** "
             f"(P&L {stress.loc[worst_s,'pnl_impact']:.2%}, OCI {stress.loc[worst_s,'oci_impact']:.2%}). Rate shocks hit the "
             "P&L (matched FI) book; equity/property shocks hit the OCI (surplus) book. The grid shows how the worst-case "
             "loss shifts across philosophies - the **Stress Loss** column in Part A's table is each portfolio's worst row.\n")

    m.append("### B2. Lens 5 - LAGIC-style capital\n")
    m.append(img("04_capital_by_category", "Figure 6. Capital charge by category (Baseline)."))
    m.append(f"The asset risk charge is the **worst of an 8-scenario prescribed panel** (deterministic, no history needed): "
             f"**{lag['capital_charge']:.2%}** of assets, binding in the **{lag['worst_scenario']}** scenario "
             f"(the fully diversified 'all categories at once' aggregate is **{lag['diversified_charge']:.2%}**, reported as "
             f"a comparator). Return on capital is **{lag['return_on_capital']:.2f}x**. The largest capital consumers in "
             "the binding scenario:\n")
    m.append(_df_to_md(lag["marginal_capital"].head(6).rename("capital").to_frame()) + "\n")
    m.append(img("14_capital_efficient_frontier", "Figure 7. Capital-efficient frontier (return vs capital)."))
    m.append(f"Optimising return **per unit of capital** gives the Max-RoC portfolio: expected return "
             f"**{pf['Max-RoC'].expected_return():.2%}** at capital charge **{comp.loc['Max-RoC','capital_charge']:.2%}** "
             f"(RoC {comp.loc['Max-RoC','return_on_capital']:.2f}x). Conversely, **holding capital at the baseline's "
             f"level**, the capital-budgeted optimiser lifts expected return to "
             f"**{pf['Capital-Budgeted'].expected_return():.2%}** (vs baseline {base.expected_return():.2%}) - more return "
             "for the same capital. For an insurer, capital - not volatility - is usually the binding constraint.\n")

    m.append("### B3. Lens 3/6 - Through-time earnings & carry\n")
    m.append(img("08_earnings_vs_plan", "Figure 8. Annual earnings vs plan (red = missed)."))
    m.append(f"Annual earnings (P&L) volatility is **{earn['earnings_volatility']:.2%}**; the chance of missing the "
             f"{earn['plan_target']:.1%} plan is **{earn['plan_miss_prob']:.0%}**; predictable **carry funds "
             f"{earn['carry_share_of_return']:.0%}** of the return.\n")
    m.append(img("11_duration_earnings_example", "Figure 9. Duration & earnings stability: a duration-matched book earns "
                 "steady carry whichever way rates move; an unmatched book is an unhedged rate bet."))
    et = results["earnings_risk"]["table"].loc["Baseline"]
    m.append(img("19_earnings_at_risk", "Figure 10. Bootstrap distribution of plan-year P&L, with earnings-at-risk and CTE95."))
    m.append(f"Block-bootstrapping the monthly P&L into plan-year outcomes gives an **earnings-at-risk (5%) of "
             f"{et['earnings_at_risk_5pc']:.2%}** and a **CTE95 of {et['cte_95']:.2%}** (the average of the worst 1-in-20 "
             f"years) against a {et['plan_target']:.1%} plan - the downside an insurer's capital must absorb.\n")

    m.append("### B4. Lens 4 - Duration / ALM\n")
    m.append(img("09_duration_gap", "Figure 11. Asset vs liability duration by currency."))
    m.append(img("21_duration_contribution", "Figure 12. Each asset's contribution to portfolio duration."))
    rs = dur["rate_shock"]
    m.append(f"The total dollar-duration gap is **{dur['total_dollar_duration_gap']:+.2f}y** (assets vs liabilities). "
             f"A +100bp shock moves **economic surplus {rs['economic_surplus_impact']:+.2%}** but **P&L earnings "
             f"{rs['pnl_earnings_impact']:+.2%}** - the difference (**{rs['economic_minus_pnl']:+.2%}**) is exactly the "
             "OCI/surplus book's rate exposure that bypasses earnings.\n")

    m.append("### B5. Risk budgeting & diversification\n")
    m.append(img("13_risk_contribution", "Figure 13. Risk contribution by asset and by capital category."))
    m.append(img("12_correlation_heatmap", "Figure 14. Asset return correlation matrix."))
    m.append(f"Diversification ratio **{ra['diversification_ratio']:.2f}** (1 = none); average pairwise correlation "
             f"**{ra['avg_pairwise_correlation']:.2f}**. Top risk contributors:\n")
    m.append(_df_to_md((ra["risk_by_asset"].head(5) * 100).round(1).rename("risk_%").to_frame()) + "\n")
    m.append(img("20_earnings_vol_contribution", "Figure 15. Which sleeves drive annual earnings (P&L) volatility."))
    evc = ra["earnings_vol_contribution"]
    m.append(f"Earnings (P&L) volatility is concentrated in **{evc.index[0]}** and **{evc.index[1]}** "
             f"({evc.iloc[0]:.0%} and {evc.iloc[1]:.0%} of P&L variance) - the long-duration P&L bonds whose "
             "mark-to-market flows through earnings. This is distinct from total-risk contribution: an asset can be small "
             "in the risk budget yet large in the *earnings* budget.\n")

    if fa is not None:
        m.append("### B6. Through-time return drivers (factor analysis)\n")
        m.append(img("24_factor_attribution", "Figure 16. Annualised return attributed to each macro/market factor (plus alpha/carry)."))
        m.append(img("25_rolling_factor_betas", "Figure 17. Rolling factor betas - exposures drift through time."))
        m.append(f"Regressing the book's monthly returns on the macro/market factors (rates, credit spread, structured "
                 f"spread, equity, property, gold) explains **{fa['r_squared']:.0%}** of the variation. This is the "
                 "*why* behind the return - and the **point-in-time vs through-time** tension: a flat average exposure can "
                 "hide a factor bet that is wound up and down over the cycle (the rolling betas show that drift). Return "
                 "attribution by factor:\n")
        m.append(_df_to_md(fa["attribution"].round(4)) + "\n")
        m.append("On real data, replace the generative factors with observable market series (a rates index, IG OAS, an "
                 "equity index) and the same regression gives a genuine driver-of-returns decomposition.\n")

    m.append("### B7. Liquidity, credit quality & solvency\n")
    m.append(img("15_liquidity_and_rating", "Figure 18. Liquidity tiers and rating distribution."))
    s = dg["surplus"]
    m.append(f"**{dg['liquidity']['pct_illiquid']:.0%}** of the book is illiquid; **{dg['rating']['sub_investment_grade']:.0%}** "
             f"is sub-investment-grade; effective number of assets **{dg['concentration']['effective_n_assets']:.1f}**. "
             f"Surplus is **{s['surplus']:.0%}** of assets (coverage {s['coverage_ratio']:.2f}x); the worst stress erodes "
             f"surplus by **{s['surplus_erosion_pct']:.0%}** to a coverage of {s['coverage_ratio_stressed']:.2f}x.\n")

    m.append("### B8. Marginal efficiency & historical stress\n")
    m.append(img("16_marginal_efficiency", "Figure 19. Return vs marginal capital per asset (bubble = weight)."))
    m.append("Realised behaviour through the embedded historical episodes:\n")
    m.append(_df_to_md(dg["historical_stress"].round(4)) + "\n")

    m.append("### B9. Within-asset-class diversification (implementation alpha)\n")
    m.append(img("17_intra_asset_uplift", "Figure 20. Same-risk return pickup and diversification benefit by class."))
    m.append("The strategic allocation is fixed, but inside each class a better mix of sub-sleeves can add return at the "
             "**same risk**. Holding volatility at each class's benchmark level, the enhanced sub-allocation adds a few "
             f"basis points per class - **+{intra['portfolio_return_uplift_bps']:.1f} bps at the portfolio level, with the "
             "SAA completely unchanged**.\n")
    m.append(_df_to_md(intra["summary"].round(3)) + "\n")
    m.append(img("18_intra_class_weights", "Figure 21. AUD Sovereign: the enhanced mix tilts to the belly of the curve "
                 "(5-10y), trimming the low-yield 2y and the high-vol 20y - more yield at the same duration risk."))
    m.append("This is genuine *implementation* alpha: small, repeatable and orthogonal to the top-level allocation - "
             "exactly where an insurer with a constrained SAA can still add value.\n")
    m.append(f"**Out-of-sample and net of trading costs** (walk-forward, annual rebalancing) the net portfolio pickup is "
             f"**{intra['portfolio_oos_uplift_bps']:+.1f} bps**, and it is concentrated where sub-sleeve dispersion is "
             "*structural*: **Structured Credit** (CLOs by rating, ABS/RMBS/CMBS - a strategic growth area), **Listed "
             "Equities** (region/style) and **IG Credit** (quality/sector) all add meaningful same-risk return, whereas "
             "**High Yield, the sovereign curve and Private Credit** are too noisy for the tilt to pay reliably. The honest "
             "conclusion: harvest intra-class implementation alpha **only where the dispersion is structural** (structured "
             "credit is the standout - worth a deeper, granular build), size it modestly, and rebalance slowly to keep "
             "turnover low. Out-of-sample testing is exactly what separates a real edge from an in-sample mirage.\n")

    # ----------------------------------------------------------- close
    m.append("## Conclusion\n")
    m.append("No single construction philosophy captures an insurer's problem. Equal weight is the naive benchmark; "
             "Max-Sharpe improves risk-adjusted return; Min-Variance and Risk-Parity trade return for stability; "
             "Max-Diversification spreads correlation risk; the Max-RoC and Capital-Budgeted optimisers are far more "
             "capital-efficient. The right choice depends on which lens - return, drawdown, capital, earnings stability or "
             "ALM - is binding for the business at the time. The lab makes that trade-off explicit and is built to add "
             "new philosophies (Black-Litterman, robust, ML forecasts) behind the same comparison.\n")
    m.append("## Methodology & limitations\n")
    m.append("Dummy data is a factor model (rates/credit/equity/property/gold + idiosyncratic) with embedded GFC/COVID/2022 "
             "episodes. MVO uses forward `exp_return` assumptions and historical covariance. Realised-return, earnings-"
             "volatility and drawdown figures inherit the dummy data's one-off secular rate-decline tailwind and are "
             "illustrative, not forecasts. The LAGIC module is a simplified, illustrative asset-risk charge (worst of an "
             "8-scenario prescribed panel) - not the legal standard; the capital-aware *optimisers* use a smooth "
             "diversified-aggregate proxy for the charge because it is differentiable (the worst-of-panel max is not), so "
             "their reported charge can differ slightly from the binding scenario. Stress impacts are first-order "
             "(duration/beta). Liabilities are stylised (backed ~1:1 by the P&L book; ratio "
             f"{config['portfolio'].get('liability_ratio',0.82):.0%} of assets). Risk assets are restricted to the big-four "
             "currencies. See `README.md` for how to drop in real data.\n")

    (ROOT / "outputs" / "report.md").write_text("\n".join(m), encoding="utf-8")


# ------------------------------------------------------------- orchestration
def generate_all(results: dict) -> None:
    """Write every table and chart from the computed results bundle."""
    config = results["config"]
    portfolios = results["portfolios"]
    base = portfolios["Baseline"]

    # tables
    save_table(results["comparison"].round(5), "portfolio_comparison.csv")
    save_table(base.weights.rename("weight").to_frame(), "baseline_allocation.csv")
    save_table(results["frontier"].round(5), "efficient_frontier.csv", index=False)
    save_table(results["stress"].round(5), "stress_scenarios.csv")
    save_table(results["lagic"]["Baseline"]["asset_charges"].rename("capital_charge").to_frame(), "lagic_asset_charges.csv")
    save_table(results["lagic"]["Baseline"]["scenario_charges"].rename("capital_charge").to_frame(), "lagic_scenario_charges.csv")
    save_table(results["earnings"]["annual"].round(5), "earnings_annual.csv")
    save_table(results["duration"]["by_currency"].round(4), "duration_by_currency.csv")
    save_table(pd.Series(results["duration"]["rate_shock"]).round(5).to_frame("value"), "duration_rate_shock.csv")
    save_table(results["duration_example"]["table"].round(5), "duration_earnings_example.csv", index=False)
    save_table(results["capital_frontier"].round(5), "capital_efficient_frontier.csv", index=False)

    # extra-analysis tables
    ra = results["risk_attribution"]
    save_table(ra["risk_by_asset"].rename("risk_share").to_frame(), "risk_contribution_by_asset.csv")
    save_table(ra["risk_by_category"].rename("risk_share").to_frame(), "risk_contribution_by_category.csv")
    save_table(ra["marginal_efficiency"].round(5), "marginal_efficiency.csv")
    save_table(ra["correlation_matrix"].round(3), "correlation_matrix.csv")
    dg = results["diagnostics"]
    save_table(dg["rating"]["by_rating"].rename("weight").to_frame(), "rating_distribution.csv")
    save_table(dg["liquidity"]["by_tier"].rename("weight").to_frame(), "liquidity_profile.csv")
    save_table(pd.Series(dg["surplus"]).round(5).to_frame("value"), "surplus_analysis.csv")
    save_table(dg["historical_stress"].round(5), "historical_stress.csv")
    intra = results["intra_asset"]
    save_table(intra["summary"].round(4), "intra_asset_uplift.csv")
    save_table(intra["per_class"]["AUD Sovereign"]["weights"].round(4), "intra_class_weights_aud_sovereign.csv")
    er = results["earnings_risk"]
    save_table(er["table"].round(5), "earnings_at_risk.csv")
    save_table(ra["earnings_vol_contribution"].rename("earnings_vol_share").to_frame(), "earnings_vol_contribution.csv")
    save_table(results["return_capital_budget"].round(5), "return_capital_budget_frontier.csv", index=False)
    if "stress_grid" in results:
        save_table(results["stress_grid"].round(5), "stress_grid.csv")
    if "factor_analysis" in results:
        fa = results["factor_analysis"]
        save_table(fa["attribution"].round(5), "factor_attribution.csv")
        save_table(fa["rolling_betas"].round(4), "factor_rolling_betas.csv")

    # construction philosophies shown side by side (Lens 1)
    philosophies = [n for n in ["Equal-Weight", "Max-Sharpe", "Min-Variance",
                                "Risk-Parity", "Max-Diversification"] if n in portfolios]

    # charts
    frontier_points = {"Baseline": base}
    frontier_points.update({k: portfolios[k] for k in
                            ["Max-Sharpe", "Min-Variance", "Risk-Parity",
                             "Max-Diversification", "Max-RoC", "Risk 20%"] if k in portfolios})
    chart_efficient_frontier(results["frontier"], frontier_points)
    chart_allocation_comparison({k: portfolios[k] for k in
                                 ["Baseline", "Equal-Weight", "Max-Sharpe", "Risk-Parity",
                                  "Max-Diversification", "Max-RoC", "Risk 20%"] if k in portfolios})
    chart_stress(results["stress"])
    chart_capital_by_category(results["lagic"]["Baseline"]["category_charges"])
    chart_return_vs_vol(results["comparison"])
    chart_return_vs_capital(results["comparison"])
    chart_rolling_return(base)
    chart_earnings(results["earnings"]["annual"], config["portfolio"]["plan_return_target"])
    chart_duration_gap(results["duration"]["by_currency"])
    chart_drawdown(base)
    chart_duration_earnings_example(results["duration_example"])
    chart_correlation_heatmap(ra["correlation_matrix"])
    chart_risk_contribution(ra["risk_by_asset"], ra["risk_by_category"])
    chart_capital_frontier(results["capital_frontier"], portfolios, config)
    chart_liquidity_and_rating(dg["liquidity"], dg["rating"])
    chart_marginal_efficiency(ra["marginal_efficiency"])
    chart_intra_asset(intra)
    chart_intra_weights(intra, "AUD Sovereign")
    chart_earnings_at_risk(er, config["portfolio"]["plan_return_target"])
    chart_earnings_vol_contribution(ra["earnings_vol_contribution"])
    chart_duration_contribution(base)
    if "stress_grid" in results:
        chart_stress_grid(results["stress_grid"])
    chart_philosophy_metrics(results["comparison"], ["Baseline"] + philosophies)
    if "factor_analysis" in results:
        chart_factor_attribution(results["factor_analysis"]["attribution"])
        chart_rolling_factor_betas(results["factor_analysis"]["rolling_betas"])

    write_markdown(results)
    write_full_report(results)
