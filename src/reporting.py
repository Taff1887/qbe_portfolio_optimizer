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
from utils import (
    OUT_TABLES,
    PALETTE,
    ROOT,
    drawdown_series,
    save_chart,
    save_table,
)


# ----------------------------------------------------------------- tables
def comparison_table(portfolios: dict[str, Portfolio], config: dict) -> pd.DataFrame:
    """Row per portfolio across return, risk, capital and accounting lenses."""
    rows = {}
    for name, pf in portfolios.items():
        cap = run_lagic(pf, config)
        s = pf.summary()
        rows[name] = {
            "exp_return": s["expected_return"],
            "volatility": s["volatility"],
            "realised_return": s["ann_return"],
            "sharpe": s["sharpe"],
            "max_drawdown": s["max_drawdown"],
            "carry": s["carry"],
            "duration": s["duration"],
            "core_fi": s["core_fi"],
            "risk_assets": s["risk_assets"],
            "liquidity": s["liquidity"],
            "capital_charge": cap["capital_charge"],
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
    md.append("# QBE-style multi-lens portfolio analysis - summary\n")
    md.append("_Generated by `python src/main.py`. Dummy data; replace with real data via `data/processed/returns.csv` and `config.yaml`._\n")
    md.append("## 1. Portfolio comparison\n")
    md.append(_df_to_md(comp.round(4)))
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


def write_full_report(results: dict) -> None:
    """A single comprehensive markdown report across every lens, with charts."""
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
    worst_s = stress["total_impact"].idxmin()

    def img(name, cap):
        return f"![{cap}](charts/{name}.png)\n\n*{cap}*\n"

    m = []
    m.append("# QBE-style Multi-Lens Portfolio Report\n")
    m.append("_Generated by `python src/main.py` on dummy data. Every number reconciles to a CSV in `outputs/tables/`; "
             "replace `data/processed/returns.csv` and `config.yaml` with real data to refresh._\n")
    m.append("## Executive read\n")
    m.append("Classic mean-variance optimisation answers one question - return per unit of volatility. An insurer must "
             "also see **regulatory capital**, **earnings stability**, **accounting (P&L vs OCI)** and **asset-liability "
             "duration** at once. The table below shows portfolios with similar expected returns that differ sharply on "
             "capital, drawdown and capital efficiency.\n")
    m.append("## 1. Portfolio comparison\n")
    m.append(_df_to_md(comp.round(4)) + "\n")
    m.append("## 2. Lens 1 - Mean-variance optimisation\n")
    m.append(img("01_efficient_frontier", "Figure 1. Constrained efficient frontier with key portfolios."))
    m.append(f"Within the insurer constraints (min {config['portfolio']['min_core_fixed_income']:.0%} core fixed income, "
             f"currency buckets, per-asset caps), the **Max-Sharpe** portfolio reaches expected return "
             f"**{pf['Max-Sharpe'].expected_return():.2%}** at volatility **{pf['Max-Sharpe'].volatility():.2%}** "
             f"(Sharpe {pf['Max-Sharpe'].summary()['sharpe']:.2f}), versus the baseline at "
             f"**{base.expected_return():.2%}** / **{base.volatility():.2%}**.\n")
    m.append(img("02_allocation_comparison", "Figure 2. Allocation by capital category across portfolios."))
    m.append("## 3. Lens 2 - Instantaneous stress testing\n")
    m.append(img("03_stress_scenarios", "Figure 3. Stress impacts split into P&L and OCI."))
    m.append(_df_to_md(stress.round(4)) + "\n")
    m.append(f"The worst instantaneous scenario is **{worst_s}** at **{stress.loc[worst_s,'total_impact']:.2%}** "
             f"(P&L {stress.loc[worst_s,'pnl_impact']:.2%}, OCI {stress.loc[worst_s,'oci_impact']:.2%}). Rate shocks hit the "
             "P&L (matched FI) book; equity/property shocks hit the OCI (surplus) book.\n")
    m.append("## 4. Lens 5 - LAGIC-style capital\n")
    m.append(img("04_capital_by_category", "Figure 4. Capital charge by category."))
    m.append(f"The binding capital charge is **{lag['capital_charge']:.2%}** of assets "
             f"(binding basis: {lag['binding_basis']}; worst single scenario: {lag['worst_scenario']}). "
             f"Return on capital is **{lag['return_on_capital']:.2f}x**. The largest capital consumers:\n")
    m.append(_df_to_md(lag["marginal_capital"].head(6).rename("capital").to_frame()) + "\n")
    m.append(img("14_capital_efficient_frontier", "Figure 5. Capital-efficient frontier (return vs capital)."))
    m.append(f"Optimising return **per unit of capital** gives the Max-RoC portfolio: expected return "
             f"**{pf['Max-RoC'].expected_return():.2%}** at capital charge **{comp.loc['Max-RoC','capital_charge']:.2%}** "
             f"(RoC {comp.loc['Max-RoC','return_on_capital']:.2f}x).\n")
    m.append("## 5. Lens 3/6 - Through-time earnings & carry\n")
    m.append(img("08_earnings_vs_plan", "Figure 6. Annual earnings vs plan (red = missed)."))
    m.append(f"Annual earnings (P&L) volatility is **{earn['earnings_volatility']:.2%}**; the chance of missing the "
             f"{earn['plan_target']:.1%} plan is **{earn['plan_miss_prob']:.0%}**; predictable **carry funds "
             f"{earn['carry_share_of_return']:.0%}** of the return.\n")
    m.append(img("11_duration_earnings_example", "Figure 7. Duration & earnings stability: a duration-matched book earns "
                 "steady carry whichever way rates move; an unmatched book is an unhedged rate bet."))
    m.append("## 6. Lens 4 - Duration / ALM\n")
    m.append(img("09_duration_gap", "Figure 8. Asset vs liability duration by currency."))
    rs = dur["rate_shock"]
    m.append(f"The total dollar-duration gap is **{dur['total_dollar_duration_gap']:+.2f}y** (assets vs liabilities). "
             f"A +100bp shock moves **economic surplus {rs['economic_surplus_impact']:+.2%}** but **P&L earnings "
             f"{rs['pnl_earnings_impact']:+.2%}** - the difference (**{rs['economic_minus_pnl']:+.2%}**) is exactly the "
             "OCI/surplus book's rate exposure that bypasses earnings.\n")
    m.append("## 7. Risk budgeting & diversification\n")
    m.append(img("13_risk_contribution", "Figure 9. Risk contribution by asset and by capital category."))
    m.append(img("12_correlation_heatmap", "Figure 10. Asset return correlation matrix."))
    m.append(f"Diversification ratio **{ra['diversification_ratio']:.2f}** (1 = none); average pairwise correlation "
             f"**{ra['avg_pairwise_correlation']:.2f}**. Top risk contributors:\n")
    m.append(_df_to_md((ra["risk_by_asset"].head(5) * 100).round(1).rename("risk_%").to_frame()) + "\n")
    m.append("## 8. Liquidity, credit quality & solvency\n")
    m.append(img("15_liquidity_and_rating", "Figure 11. Liquidity tiers and rating distribution."))
    s = dg["surplus"]
    m.append(f"**{dg['liquidity']['pct_illiquid']:.0%}** of the book is illiquid; **{dg['rating']['sub_investment_grade']:.0%}** "
             f"is sub-investment-grade; effective number of assets **{dg['concentration']['effective_n_assets']:.1f}**. "
             f"Surplus is **{s['surplus']:.0%}** of assets (coverage {s['coverage_ratio']:.2f}x); the worst stress erodes "
             f"surplus by **{s['surplus_erosion_pct']:.0%}** to a coverage of {s['coverage_ratio_stressed']:.2f}x.\n")
    m.append("## 9. Marginal efficiency & historical stress\n")
    m.append(img("16_marginal_efficiency", "Figure 12. Return vs marginal capital per asset (bubble = weight)."))
    m.append("Realised behaviour through the embedded historical episodes:\n")
    m.append(_df_to_md(dg["historical_stress"].round(4)) + "\n")
    m.append("## 10. Conclusion\n")
    m.append("No single optimisation captures an insurer's problem. The Min-Variance and Max-RoC portfolios are far more "
             "capital-efficient than the baseline; the Max-Sharpe portfolio improves risk-adjusted return; raising risk to "
             "20% lifts return but costs capital and Sharpe. The right choice depends on which lens - return, drawdown, "
             "capital, earnings stability or ALM - is binding for the business at the time. This framework makes that "
             "trade-off explicit.\n")
    m.append("## Methodology & limitations\n")
    m.append("Dummy data is a factor model (rates/credit/equity/property + idiosyncratic) with embedded GFC/COVID/2022 "
             "episodes. MVO uses forward `exp_return` assumptions and historical covariance. Realised-return, earnings-"
             "volatility and drawdown figures inherit the dummy data's one-off secular rate-decline tailwind and are "
             "illustrative, not forecasts. The LAGIC module is a "
             "simplified, illustrative asset-risk charge - not the legal standard. Stress impacts are first-order "
             "(duration/beta). Liabilities are stylised (backed ~1:1 by the P&L book; ratio "
             f"{config['portfolio'].get('liability_ratio',0.82):.0%} of assets). See `README.md` for how to drop in real data.\n")

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

    # charts
    chart_efficient_frontier(results["frontier"], {
        "Baseline": base, "Max-Sharpe": portfolios["Max-Sharpe"],
        "Min-Variance": portfolios["Min-Variance"], "Max-RoC": portfolios["Max-RoC"],
        "Risk 20%": portfolios["Risk 20%"]})
    chart_allocation_comparison({k: portfolios[k] for k in ["Baseline", "Max-Sharpe", "Max-RoC", "Risk 20%"]})
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

    write_markdown(results)
    write_full_report(results)
