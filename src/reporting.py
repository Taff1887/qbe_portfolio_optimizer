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


def chart_capital_modules(lf: dict) -> None:
    """The fuller LAGIC build: risk modules, diversification credit and total."""
    mods = lf["modules"]
    labels = list(mods.keys()) + ["diversification", "concentration", "TOTAL"]
    div_credit = -(sum(mods.values()) - lf["diversified_modules"])
    vals = list(mods.values()) + [div_credit, lf["concentration_addon"], lf["total_capital_requirement"]]
    colors = (["#0072B2"] * len(mods)) + ["#2E8B57", "#E69F00", "#C0392B"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(vals)), [v * 100 for v in vals], color=colors)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Capital (% of assets)")
    ax.set_title("Fuller LAGIC-style capital: risk modules -> diversified total")
    save_chart(fig, "28_capital_modules")


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


def chart_structured_credit(sc: dict) -> None:
    """Two panels: the tranche map (return vs vol, bubble = capital) and the
    benchmark vs same-risk vs capital-efficient sub-allocation."""
    tr = sc["tranches"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sizes = tr["capital"] * 1500 + 25
    sccr = axes[0].scatter(tr["ann_vol"] * 100, tr["exp_return"] * 100, s=sizes, alpha=0.65, color="#0072B2")
    for nm, r in tr.iterrows():
        axes[0].annotate(nm, (r["ann_vol"] * 100, r["exp_return"] * 100), fontsize=6.5,
                         textcoords="offset points", xytext=(4, 3))
    axes[0].set_xlabel("Volatility (%)")
    axes[0].set_ylabel("Expected return (%)")
    axes[0].set_title("Structured-credit tranche map (bubble = capital charge)")

    w = sc["weights"][["benchmark", "same_risk", "capital_efficient"]]
    x = np.arange(len(w.index))
    bw = 0.27
    for k, col in enumerate(w.columns):
        axes[1].bar(x + (k - 1) * bw, w[col] * 100, bw, label=col)
    axes[1].set_xticks(x, w.index, rotation=40, ha="right", fontsize=7)
    axes[1].set_ylabel("Weight (%)")
    axes[1].set_title("Benchmark vs same-risk vs capital-efficient mix")
    axes[1].legend(fontsize=8)
    save_chart(fig, "29_structured_credit")


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


def chart_regimes(rg: dict) -> None:
    """Two panels: the market proxy coloured by regime, and key risk stats by regime."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    proxy, regime = rg["proxy"], rg["regime"]
    cum = (1 + proxy).cumprod()
    off = regime == "Risk-off"
    axes[0].plot(cum.index, cum.values, color="#0072B2", lw=1.0)
    axes[0].fill_between(cum.index, cum.min(), cum.max(), where=off.values,
                         color="#C0392B", alpha=0.18, label="Risk-off")
    axes[0].set_title("Risk-asset proxy with risk-off regimes shaded")
    axes[0].set_ylabel("Growth of 1")
    axes[0].legend(fontsize=8)

    t = rg["table"]
    x = np.arange(len(t.index))
    axes[1].bar(x - 0.2, t["baseline_vol"] * 100, 0.4, label="Baseline vol (%)", color="#D55E00")
    axes[1].bar(x + 0.2, t["diversification_ratio"], 0.4, label="Diversification ratio", color="#117733")
    axes[1].set_xticks(x, t.index)
    axes[1].set_title("Risk by regime: volatility up, diversification down in risk-off")
    axes[1].legend(fontsize=8)
    save_chart(fig, "30_regimes")


def chart_glide_path(gp: dict) -> None:
    """Two panels: the duration schedule through the year, and the resulting
    plan-year earnings distribution, per policy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    months = gp["months"]
    for i, (name, sched) in enumerate(gp["schedules"].items()):
        axes[0].plot(months, sched, marker="o", color=PALETTE[i % len(PALETTE)], label=name)
    axes[0].set_xlabel("Month of plan year")
    axes[0].set_ylabel("Duration held (years)")
    axes[0].set_title("Duration policy through the year")
    axes[0].legend(fontsize=8)

    for i, (name, dist) in enumerate(gp["distributions"].items()):
        axes[1].hist(dist * 100, bins=60, histtype="step", lw=1.6,
                     color=PALETTE[i % len(PALETTE)], label=name)
    axes[1].axvline(gp["plan"] * 100, color="#222", ls="--", lw=1.2, label=f"Plan {gp['plan']:.1%}")
    axes[1].set_xlabel("Plan-year earnings (%)")
    axes[1].set_title("Distribution of plan-year earnings by policy")
    axes[1].legend(fontsize=8)
    save_chart(fig, "27_glide_path")


def chart_pareto(pareto: dict) -> None:
    """Capital vs return: candidates, the baseline, and the books that dominate it."""
    t = pareto["table"]
    base = pareto["baseline_objectives"]
    dom = t["dominates_baseline"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(t.loc[~dom, "capital"] * 100, t.loc[~dom, "exp_return"] * 100,
               s=40, color="#BBBBBB", label="candidate (no improvement)")
    ax.scatter(t.loc[dom, "capital"] * 100, t.loc[dom, "exp_return"] * 100,
               s=70, color="#117733", label="Pareto-dominates baseline")
    ax.scatter(base["capital"] * 100, base["exp_return"] * 100, s=220, marker="*",
               color="#C0392B", zorder=5, label="Baseline (current book)")
    # shade the region that is unambiguously better than the baseline (more return,
    # less capital) - any green point here also beats it on vol/earnings/stress.
    ax.axhline(base["exp_return"] * 100, color="#C0392B", lw=0.7, ls=":")
    ax.axvline(base["capital"] * 100, color="#C0392B", lw=0.7, ls=":")
    ax.set_xlabel("Capital charge (% of assets)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Pareto search: more return for less capital (and lower vol / earnings vol / stress)")
    ax.legend(fontsize=8)
    save_chart(fig, "26_pareto_improvements")


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
def story(why: str = "", means: str = "", how: str = "", found: str = "", nxt: str = "") -> str:
    """Render a section's narrative as a consistent set of labelled beats:
    why the lens is looked at, what it means, how it is calculated, what we found,
    and what to study next. Empty beats are skipped."""
    beats = [
        ("Why this lens", why),
        ("What it means", means),
        ("How it is calculated", how),
        ("What we found", found),
        ("What to study next", nxt),
    ]
    return "\n\n".join(f"**{label}.** {text}" for label, text in beats if text) + "\n"


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
                                "Risk-Parity", "Max-Diversification", "Black-Litterman",
                                "Robust", "ML-Forecast"] if n in pf]

    def img(name, cap):
        return f"![{cap}](charts/{name}.png)\n\n*{cap}*\n"

    def comp_md(rows=None, cols=_COMPARISON_COLS):
        sub = comp if rows is None else comp.loc[[r for r in rows if r in comp.index]]
        return _df_to_md(sub[cols].round(4))

    prov = results.get("market").provenance if results.get("market") is not None else {}
    src = prov.get("source", "unknown")
    if src == "real":
        banner = f"**Data: REAL market data** (provider: {prov.get('provider')}). "
    elif src == "synthetic-fallback":
        banner = ("**Data: SYNTHETIC fallback** - a real-data fetch was configured but unavailable in this environment "
                  f"({prov.get('reason','')}), so the transparent factor model was used. Numbers are illustrative; they "
                  "become real the moment the configured provider (FMP/Yahoo) is reachable or an FMP key is supplied. ")
    else:
        banner = "**Data: synthetic** (transparent factor model). "

    m = []
    m.append("# QBE-style Portfolio Optimisation Research Lab - Report\n")
    m.append(f"_{banner}Generated by `python src/main.py`. Every number reconciles to a CSV in `outputs/tables/`; "
             "real data is a config change (`data.source: real` + `market_data` tickers)._\n")

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
    m.append("The framing follows the way the problem was put to us: there are really **two big lenses**. The first is "
             "*mean-variance* - well understood, many ways to do it - which here becomes a whole family of construction "
             "philosophies (Part A). The second is everything mean-variance ignores: shocks do not happen in isolation, an "
             "insurer is judged on its **annual earnings plan** not a point-in-time Sharpe, and **capital** (not "
             "volatility) is usually the binding constraint (Part B). The aim is not a portfolio that is optimal on one "
             "axis but one that is **acceptable across all of them** - and, where possible, a change that improves several "
             "at once (growing the pie rather than re-slicing it).\n")
    m.append("**How to read each section below.** Every lens is written to the same five beats so the reasoning is "
             "explicit: _Why this lens_ (why it is looked at), _What it means_, _How it is calculated_ (how it was "
             "researched and computed), _What we found_ (the result on this book), and _What to study next_.\n")

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
             "- **Black-Litterman** - equilibrium returns (reverse-optimised from the book) blended with explicit views.\n"
             "- **Robust** - resampled (Michaud) max-Sharpe, stable to estimation error.\n"
             "- **ML-Forecast** - expected returns from a pooled ridge on momentum / reversal / carry / vol signals.\n"
             "- **Capital-aware (Max-RoC, Capital-Budgeted, Min-EarningsVol)** - optimise return per unit of, or subject to, capital / earnings stability.\n"
             "- **Pareto-Balanced** - the best book that dominates the baseline on every objective (see A3).\n"
             "- **Baseline / Risk 20%** - the insurer's strategic book and a risk-scaled scenario for reference.\n")
    m.append(story(
        why="Every philosophy encodes a different *belief* about what is knowable. Mean-variance trusts the return "
            "forecast; minimum variance and risk parity distrust it and lean only on the covariance; equal weight "
            "distrusts everything. Building them side by side stops us over-fitting to one worldview - exactly the "
            "'don't get stuck in one way of thinking' instruction.",
        how="Each optimiser is an independent function in `src/optimizer.py` / `src/construction.py` that returns a "
            "`Portfolio`. Forward `exp_return` assumptions drive return; the historical covariance drives risk. Risk "
            "parity and maximum diversification are solved (SLSQP) under the *same* mandate constraints as mean-variance "
            "so the contest is fair; equal weight is left unconstrained as the naive control.",
        nxt="Wire in the scaffolded **Black-Litterman** (blend equilibrium with views) and **robust** optimisers, then "
            "newer philosophies - ML return forecasts, regime-switching, Bayesian / multi-objective - all behind the "
            "same comparison so they are judged on identical metrics."))
    m.append("### A2. Side-by-side comparison\n")
    m.append("Every portfolio, every headline metric (the full set, including the retained context columns, is in "
             "`outputs/tables/portfolio_comparison.csv`):\n")
    m.append(comp_md() + "\n")
    m.append("_VaR / CVaR are historical monthly 95% figures (signed; negative = loss); stress loss is the worst "
             "instantaneous scenario; turnover (and tracking error, in the CSV) is measured against the Baseline book._\n")
    m.append(img("23_philosophy_metrics", "Figure 1. Risk-adjusted return (Sharpe & Sortino) across the construction philosophies."))
    m.append(img("01_efficient_frontier", "Figure 2. Constrained efficient frontier with each philosophy plotted."))
    m.append(story(
        means="Read each row as a portfolio and each column as a lens. The right-most columns (capital charge, stress "
              "loss, turnover) are where philosophies that look identical on return/volatility separate - which is the "
              "whole argument for not optimising a single number.",
        found=f"Within the insurer constraints the **Max-Sharpe** book reaches expected return "
              f"**{pf['Max-Sharpe'].expected_return():.2%}** at volatility **{pf['Max-Sharpe'].volatility():.2%}** "
              f"(Sharpe {comp.loc['Max-Sharpe','sharpe']:.2f}) versus the baseline at "
              f"**{base.expected_return():.2%}** / **{base.volatility():.2%}**; **Min-Variance** and **Risk-Parity** trade "
              "return for stability, while **Equal-Weight** takes the most risk and the largest turnover to implement. No "
              "single portfolio wins every column.",
        nxt="Add the objective that is actually binding for the business each period (capital, earnings stability) as a "
            "constraint and search for **Pareto improvements** - moves that help one lens without hurting the others - "
            "rather than a single 'optimal' point."))
    m.append(img("02_allocation_comparison", "Figure 3. Allocation by capital category across philosophies."))

    # ----------------------------------------------------- A3 Pareto search
    pareto = results.get("pareto")
    if pareto is not None:
        bo = pareto["baseline_objectives"]
        m.append("### A3. Pareto improvements over the current book\n")
        m.append(story(
            why="The philosophies above each optimise *one* thing. The sharper question for the business is whether a "
                "book exists that is **at least as good as today's on every objective at once** - return, volatility, "
                "capital, earnings volatility and worst-case stress - and strictly better on some. That is a Pareto "
                "improvement: growing the pie, not re-slicing it.",
            how="Epsilon-constraint search: maximise expected return subject to hard caps on **capital** and **earnings "
                "(P&L) volatility** (the smooth quadratic proxy) and total volatility, sweeping those caps from their "
                "best-achievable level up to the baseline's. Every solution is then scored on all five objectives and "
                "tested for Pareto-dominance over the current book.",
            found=f"**{pareto['n_dominating']} of the searched portfolios dominate the baseline** on every objective. The "
                  f"current book sits at return {bo['exp_return']:.2%}, vol {bo['volatility']:.2%}, capital "
                  f"{bo['capital']:.2%}, earnings-vol {bo['earnings_vol']:.2%}, worst-stress {bo['worst_stress']:.2%} - "
                  "and is strictly beaten on all of them. The best-balanced dominating book (**Pareto-Balanced**, shown in "
                  "the Part A table) earns the same-or-more return at materially lower capital, earnings volatility and "
                  "stress.",
            nxt="Add turnover and tracking-error to the dominated set so the cheapest *implementable* improvement is "
                "chosen, then re-run as constraints tighten (e.g. a hard capital budget) to trace the efficient surface, "
                "not just dominance over one point."))
        m.append(img("26_pareto_improvements", "Figure 3b. Pareto search - books beating the current one on return AND capital "
                     "(and, by construction, on volatility, earnings volatility and stress)."))
        dom_tbl = pareto["table"][pareto["table"]["dominates_baseline"]].drop(columns="dominates_baseline")
        if len(dom_tbl):
            m.append("Portfolios that dominate the baseline (all objectives at least as good, some strictly better):\n")
            m.append(_df_to_md(dom_tbl.round(4)) + "\n")

    # =========================================================== PART B
    m.append("## Part B - Evaluation lenses\n")
    m.append("The same portfolios, now seen through the lenses an insurer is actually governed by. Figures below focus on "
             "the Baseline book unless stated; the stress grid covers every portfolio.\n")

    m.append("### B1. Lens 2 - Instantaneous stress testing\n")
    m.append(img("22_stress_grid", "Figure 4. Stress grid - every scenario against every portfolio (total impact, %)."))
    m.append(img("03_stress_scenarios", "Figure 5. Baseline stress impacts split into P&L and OCI."))
    m.append(_df_to_md(stress.round(4)) + "\n")
    m.append(story(
        why="Volatility describes a normal month; it says nothing about a specific, sudden move in rates, spreads or "
            "equities - the events that actually threaten the balance sheet and the earnings plan. Deterministic stress "
            "tests answer 'if *this* happens tomorrow, what do we lose, and does it hit earnings or reserves?'",
        means="Each loss is split into **P&L** (FVTPL - hits this year's earnings immediately) and **OCI** (FVOCI - sits "
              "in reserves and bypasses earnings). A rate shock that is painful economically can be nearly invisible to "
              "earnings if it lands in the OCI book, and vice versa - the distinction an insurer manages to.",
        how="For every scenario, each asset is repriced first-order from its exposures: "
            "`-rate_duration x dRate - spread_duration x dSpread + equity_beta x dEquity + property_beta x dProperty`, "
            "weighted up to the portfolio. The panel deliberately includes single-factor *and* combined scenarios (shocks "
            "do not arrive in isolation), and the grid re-runs all of them against every portfolio.",
        found=f"The worst instantaneous scenario for the Baseline is **{worst_s}** at "
              f"**{stress.loc[worst_s,'total_impact']:.2%}** (P&L {stress.loc[worst_s,'pnl_impact']:.2%}, "
              f"OCI {stress.loc[worst_s,'oci_impact']:.2%}). Rate shocks concentrate in the P&L (matched FI) book; "
              "equity/property shocks land in the OCI (surplus) book. The **Stress Loss** column in Part A is each "
              "portfolio's worst row here.",
        nxt="Add second-order (convexity) terms for large moves, correlated multi-factor shocks calibrated to history, "
            "and a **reverse stress test** - solve for the scenario that breaches a chosen earnings or surplus limit, "
            "rather than guessing scenarios up front."))

    rev = results.get("reverse_stress")
    if rev is not None:
        m.append("**Reverse stress test.** Instead of guessing scenarios, solve for the single-factor move that *breaches* "
                 "a chosen limit (loss = limit / sensitivity). It answers 'how big a move wipes out surplus, or costs 2% of "
                 "earnings?' - and reads across to where the book is fragile vs resilient:\n")
        m.append(_df_to_md(rev.round(1)) + "\n")
        m.append("_Equity/property moves beyond -100% mean that factor **cannot** breach the limit on its own. The book is "
                 "very solvency-resilient to any single factor (the 18% surplus is a deep buffer), but its **P&L is most "
                 "sensitive to rates** - a ~48bp move alone costs 2% of earnings, which is exactly why the in-year duration "
                 "glide path (B10) matters._\n")

    m.append("### B2. Lens 5 - LAGIC-style capital\n")
    m.append(img("04_capital_by_category", "Figure 6. Capital charge by category (Baseline)."))
    m.append(story(
        why="For a regulated insurer the scarce resource is **regulatory capital**, not volatility. Two books with the "
            "same Sharpe can tie up very different amounts of capital, so the honest measure of efficiency is **return on "
            "capital**. The lab originally lacked this lens - it is the gap that motivated the build.",
        means="The asset risk charge is the capital the regulator makes you hold against a prescribed worst case. It is "
              "fully **deterministic** - no return history is needed - so it is driven by *what you own*, not how it "
              "happened to trade.",
        how="Each capital category carries a prescribed stress loss. A **panel of 8 scenarios** scales those stresses for "
            "different states of the world (equity/property crash, credit crisis, IG widening, structured shock, unlisted "
            "revaluation, rates/inflation, severe equity, broad recession); each scenario is aggregated with a category "
            "correlation, and the **worst of the 8 is the binding charge**. (The capital-aware *optimisers* use a smooth "
            "diversified-aggregate proxy because the worst-of-panel max is not differentiable.)",
        found=f"The binding charge is **{lag['capital_charge']:.2%}** of assets, set by the **{lag['worst_scenario']}** "
              f"scenario (diversified-aggregate comparator **{lag['diversified_charge']:.2%}**); return on capital is "
              f"**{lag['return_on_capital']:.2f}x**. Largest consumers of capital in the binding scenario:"))
    m.append(_df_to_md(lag["marginal_capital"].head(6).rename("capital").to_frame()) + "\n")
    m.append(img("14_capital_efficient_frontier", "Figure 7. Capital-efficient frontier (return vs capital)."))
    m.append(story(
        found=f"Optimising return **per unit of capital** (Max-RoC) reaches **{pf['Max-RoC'].expected_return():.2%}** at a "
              f"charge of just **{comp.loc['Max-RoC','capital_charge']:.2%}** (RoC {comp.loc['Max-RoC','return_on_capital']:.2f}x). "
              f"More usefully, **holding capital at the baseline's level**, the capital-budgeted optimiser lifts expected "
              f"return to **{pf['Capital-Budgeted'].expected_return():.2%}** (vs {base.expected_return():.2%}) - a clean "
              "'grow the pie' result: more return for the *same* capital.",
        nxt="Build toward a fuller LAGIC: add the interest-rate stress computed **net of liabilities** (not just assets), "
            "insurance- and operational-risk charges, the prescribed correlation matrix, and asset-concentration / "
            "counterparty add-ons. Then make capital a hard constraint in every optimiser, not just the capital-aware ones."))

    lf = results.get("lagic_full")
    if lf is not None:
        mods = lf["modules"]
        m.append(img("28_capital_modules", "Figure 7b. Fuller LAGIC-style build: risk modules aggregated (with "
                     "diversification) to a total capital requirement."))
        m.append(story(
            why="The asset risk charge above is only one module. A fuller capital requirement also holds capital for "
                "**interest-rate risk net of liabilities**, **insurance risk**, **operational risk** and **asset "
                "concentration** - and these do not all peak together, so they aggregate with a diversification credit.",
            how="Each module is a prescribed charge: asset risk (worst-of-panel), the net asset-minus-liability duration "
                "gap x a prescribed rate move, insurance/operational risk scaled to liabilities/premiums, and an additive "
                "single-name concentration add-on. Modules combine via sqrt(m' R m) with a prescribed inter-module "
                "correlation.",
            found=f"Modules (% of assets): asset {mods['asset_risk']:.2%}, **rate-net-of-liabilities "
                  f"{mods['rate_risk_net']:.2%}** (small - the matched book does its job), insurance "
                  f"{mods['insurance_risk']:.2%}, operational {mods['operational_risk']:.2%}. Diversified across modules "
                  f"that is **{lf['diversified_modules']:.2%}**, plus a {lf['concentration_addon']:.2%} concentration "
                  f"add-on, for a **total capital requirement of {lf['total_capital_requirement']:.2%}** of assets "
                  f"(return on total capital {lf['return_on_total_capital']:.2f}x).",
            nxt="Replace the illustrative factors with the prescribed standard, add a real liability cash-flow model for "
                "the rate module, and counterparty-grade and reinsurance-recovery risk. The key structural point already "
                "holds: a duration-matched book keeps the rate module small."))

    m.append("### B3. Lens 3/6 - Through-time earnings & carry\n")
    et = results["earnings_risk"]["table"].loc["Baseline"]
    m.append(story(
        why="An insurer is judged on whether it makes its **annual earnings plan**, not on a point-in-time Sharpe. That "
            "is the core point-in-time-vs-through-time tension: a portfolio can look efficient instantaneously yet miss "
            "the plan often because its return is volatile mark-to-market rather than predictable income.",
        means="Return splits into **carry** (predictable running yield, banked whatever happens) and **mark-to-market** "
              "(volatile price moves). The higher the carry share, the more reliably the plan is met. Carry on both books "
              "hits P&L; only the *price* move of OCI assets bypasses earnings.",
        how="Each month is decomposed into carry vs MtM and P&L vs OCI, then aggregated to calendar-year earnings. For the "
            "tail we **block-bootstrap** the realised monthly P&L into thousands of synthetic plan-years (preserving "
            "short-run autocorrelation) and read off earnings-at-risk and CTE95 - resampling observed returns rather than "
            "assuming a bell curve."))
    m.append(img("08_earnings_vs_plan", "Figure 8. Annual earnings vs plan (red = missed)."))
    m.append(img("11_duration_earnings_example", "Figure 9. Duration & earnings stability: a duration-matched book earns "
                 "steady carry whichever way rates move; an unmatched book is an unhedged rate bet."))
    m.append(img("19_earnings_at_risk", "Figure 10. Bootstrap distribution of plan-year P&L, with earnings-at-risk and CTE95."))
    m.append(story(
        found=f"Annual earnings (P&L) volatility is **{earn['earnings_volatility']:.2%}**; the chance of missing the "
              f"{earn['plan_target']:.1%} plan is **{earn['plan_miss_prob']:.0%}**; predictable **carry funds "
              f"{earn['carry_share_of_return']:.0%}** of the return. The bootstrap gives an **earnings-at-risk (5%) of "
              f"{et['earnings_at_risk_5pc']:.2%}** and a **CTE95 of {et['cte_95']:.2%}** (mean of the worst 1-in-20 years) "
              "- the downside capital must absorb.",
        nxt="Model the CFO's actual lever: a **dynamic duration glide path** that adds duration early in the plan year to "
            "protect the target and winds it down toward year-end. A full Monte-Carlo earnings simulation (paths, not just "
            "a bootstrap of the realised sample) would let that policy be optimised against plan-miss probability."))

    m.append("### B4. Lens 4 - Duration / ALM\n")
    rs = dur["rate_shock"]
    m.append(story(
        why="Insurers hold assets to back insurance **liabilities**; the first-order balance-sheet risk is the **duration "
            "gap** between the two, by currency. The brief's structure is explicit: run the P&L (matched) book roughly "
            "$-for-$ against liabilities so earnings are rate-neutral, while taking deliberate surplus duration in the "
            "**OCI** book - long economically, but without disturbing P&L.",
        means="A rate move changes the economic surplus and reported earnings by *different* amounts. The difference is "
              "precisely the OCI/surplus book's rate exposure - duration you are paid to hold for the long-term economics "
              "but which is kept out of the earnings line.",
        how="Asset dollar-durations (weight x duration) are split into P&L and OCI books per currency; liabilities are "
            "backed ~1:1 by the P&L book with durations from config. A +100bp parallel shock is then decomposed into the "
            "matched-book earnings impact (assets net of liabilities) and the full economic-surplus impact.",
        found=f"The total dollar-duration gap is **{dur['total_dollar_duration_gap']:+.2f}y**. A +100bp shock moves "
              f"**economic surplus {rs['economic_surplus_impact']:+.2%}** but **P&L earnings {rs['pnl_earnings_impact']:+.2%}** "
              f"- the gap (**{rs['economic_minus_pnl']:+.2%}**) is exactly the OCI/surplus rate exposure that bypasses "
              "earnings, confirming the matched-in-P&L / long-in-OCI design.",
        nxt="Move from single effective duration to **key-rate durations** (curve twists and steepeners, not just parallel "
            "shifts), add cross-currency basis, and drive liabilities from an actuarial cash-flow model rather than a "
            "stylised duration per currency."))
    m.append(img("09_duration_gap", "Figure 11. Asset vs liability duration by currency."))
    m.append(img("21_duration_contribution", "Figure 12. Each asset's contribution to portfolio duration."))

    m.append("### B5. Risk budgeting & diversification\n")
    evc = ra["earnings_vol_contribution"]
    m.append(story(
        why="Total volatility is one number; what a risk team manages is *where it comes from*. Two books with the same "
            "vol can be one diversified blend versus one concentrated bet - very different to govern.",
        means="Each asset's **risk contribution** is its share of portfolio volatility (the shares sum to 100%). The "
              "**diversification ratio** (weighted-average asset vol / portfolio vol) measures how much offset the "
              "correlations buy: 1.0 means none. Crucially, the **earnings-vol** budget differs from the total-risk "
              "budget - an asset can be small in one and large in the other.",
        how="Risk contributions use marginal contribution to risk, MCTR_i = w_i (Sigma w)_i / vol. The earnings-vol budget "
            "attributes the variance of annual P&L to assets via cov(asset P&L, total P&L) / var(total P&L), so only the "
            "sleeves whose mark-to-market actually flows through earnings score.",
        found=f"Diversification ratio **{ra['diversification_ratio']:.2f}**; average pairwise correlation "
              f"**{ra['avg_pairwise_correlation']:.2f}**. Earnings (P&L) volatility is concentrated in **{evc.index[0]}** "
              f"and **{evc.index[1]}** ({evc.iloc[0]:.0%} and {evc.iloc[1]:.0%} of P&L variance) - the long-duration P&L "
              "bonds, *not* the headline risk assets. Top total-risk contributors:",
        nxt="Risk contributions assume a stable covariance; stress them with **regime-dependent correlations** (which "
            "spike in a crisis) and add a **tail-risk contribution** (each asset's share of CVaR, not just variance)."))
    m.append(img("13_risk_contribution", "Figure 13. Risk contribution by asset and by capital category."))
    m.append(img("12_correlation_heatmap", "Figure 14. Asset return correlation matrix."))
    m.append(_df_to_md((ra["risk_by_asset"].head(5) * 100).round(1).rename("risk_%").to_frame()) + "\n")
    m.append(img("20_earnings_vol_contribution", "Figure 15. Which sleeves drive annual earnings (P&L) volatility."))

    if fa is not None:
        m.append("### B6. Through-time return drivers (factor analysis)\n")
        m.append(story(
            why="Risk lenses say *how much* the book can move; they do not say *why* it moves. Factor analysis is the "
                "through-time view the brief kept circling: it separates return that is **paid factor premia / carry** "
                "from return that is a **time-varying factor bet** - and shows whether the recent record is skill or a "
                "one-off tailwind.",
            means="A factor **beta** is the book's sensitivity to a driver (e.g. a rate beta near minus the duration). "
                  "The **attribution** turns each beta into the annualised return it contributed. **Rolling** betas show "
                  "the exposures drifting - a flat average can hide a bet that was put on and taken off over the cycle.",
            how="The book's monthly returns are regressed (OLS) on the macro/market factors - rates, credit spread, "
                "structured spread, equity, property, gold. Each factor's annual contribution = beta x mean monthly move "
                "x 12; the intercept is alpha/carry. Betas are then re-estimated on a rolling 36-month window.",
            found=f"The factors explain **{fa['r_squared']:.0%}** of monthly variation. The decomposition below shows "
                  "return dominated by **alpha/carry** plus a positive **rates** contribution - i.e. predictable income "
                  "plus the one-off secular fall in rates, not a repeatable directional skill. Attribution by factor:"))
        m.append(_df_to_md(fa["attribution"].round(4)) + "\n")
        m.append(img("24_factor_attribution", "Figure 16. Annualised return attributed to each macro/market factor (plus alpha/carry)."))
        m.append(img("25_rolling_factor_betas", "Figure 17. Rolling factor betas - exposures drift through time."))
        m.append(story(
            nxt="On real data, swap the generative factors for **observable** market series (a rates index change, IG "
                "OAS, an equity index) so the decomposition is genuine. Then test **factor timing / regime conditioning** "
                "and separate factor risk-premia from true alpha - the foundation for the 'AI-side', through-time "
                "optimisation the brief asked for."))

    m.append("### B7. Liquidity, credit quality & solvency\n")
    s = dg["surplus"]
    m.append(story(
        why="A high-returning book is no good if it cannot raise cash in a stress without fire-sales, or if a shock wipes "
            "out the solvency buffer. This lens checks the book is *fundable* and *solvent*, not just efficient.",
        means="**Liquidity tiers** show how much can be sold quickly; **sub-investment-grade %** and **effective number "
              "of assets** flag credit and concentration risk; **surplus** (assets minus liabilities) is the capital "
              "buffer and **coverage** is assets / liabilities - what a worst-case stress eats into.",
        how="Weights are bucketed by a liquidity score and by rating; concentration uses the Herfindahl index (effective "
            "N = 1 / sum of squared weights); surplus-at-risk applies the worst stress loss to the asset base and "
            "re-derives coverage.",
        found=f"**{dg['liquidity']['pct_illiquid']:.0%}** of the book is illiquid; **{dg['rating']['sub_investment_grade']:.0%}** "
              f"is sub-investment-grade; effective number of assets **{dg['concentration']['effective_n_assets']:.1f}**. "
              f"Surplus is **{s['surplus']:.0%}** of assets (coverage {s['coverage_ratio']:.2f}x); the worst stress erodes "
              f"surplus by **{s['surplus_erosion_pct']:.0%}** to coverage {s['coverage_ratio_stressed']:.2f}x.",
        nxt="Add a **liquidity-coverage** test (cash raisable in N days vs a stressed claims outflow), time-to-liquidate "
            "haircuts by tier, and counterparty / issuer concentration limits."))
    m.append(img("15_liquidity_and_rating", "Figure 18. Liquidity tiers and rating distribution."))

    m.append("### B8. Marginal efficiency & historical stress\n")
    m.append(story(
        why="When capital or risk is the binding constraint, the question is not 'which asset has the best Sharpe' but "
            "'which asset adds the most return for the next unit of *capital* (or risk) I spend'. And deterministic "
            "shocks should be sanity-checked against how the book *actually* behaved in real crises.",
        means="**Marginal efficiency** ranks assets by return per unit of marginal capital / marginal risk - the trade "
              "that improves the book at the margin. The **historical episodes** are a data-driven complement to the "
              "made-up scenarios: the realised drawdown through the GFC, COVID and the 2022 rate sell-off embedded in the "
              "data.",
        how="Marginal capital is the asset's prescribed LAGIC stress; marginal risk is (Sigma w)_i / vol. Historical "
            "behaviour just compounds the portfolio's realised returns over each episode window.",
        found="The episode table below shows the 2022 rate sell-off, not the GFC, is this book's worst drawdown - a "
              "direct consequence of its duration, and a useful cross-check on the deterministic 'Rates / inflation' "
              "scenario being the worst instantaneous shock.",
        nxt="Make the marginal trade **transaction-cost aware** (net of turnover) and condition the historical lens on "
            "the factor regime, so 'what hurt last time' is mapped to 'what we are exposed to now'."))
    m.append(img("16_marginal_efficiency", "Figure 19. Return vs marginal capital per asset (bubble = weight)."))
    m.append("Realised behaviour through the embedded historical episodes:\n")
    m.append(_df_to_md(dg["historical_stress"].round(4)) + "\n")

    m.append("### B9. Within-asset-class diversification (implementation alpha)\n")
    m.append(story(
        why="An insurer's top-level allocation (the 85/15 split, the currency mix) is largely *given* - set by ALM and "
            "policy, not a free optimisation. But inside each class the *mix of sub-sleeves* is a genuine lever: small, "
            "repeatable 'implementation' alpha that is orthogonal to the SAA. The brief named structured credit as the "
            "growth area to investigate here.",
        means="The test is **same-risk return**: holding each class's volatility at its benchmark level, can a better "
              "sub-sleeve mix earn more? A pickup that survives **out-of-sample and net of trading cost** is real edge; "
              "one that only shows up in-sample is an over-fit mirage.",
        how="Each class is decomposed into sub-sleeves (a shared-factor return model gives realistic, highly-correlated "
            "sub-returns). The enhanced mix maximises return subject to the benchmark-mix volatility. A walk-forward test "
            "then estimates the mix on a trailing window, applies it to the next year net of turnover cost, and "
            "accumulates the realised pickup. Structured credit is decomposed across CLOs by rating, ABS, RMBS and CMBS."))
    m.append(img("17_intra_asset_uplift", "Figure 20. Same-risk return pickup and diversification benefit by class."))
    m.append(_df_to_md(intra["summary"].round(3)) + "\n")
    m.append(img("18_intra_class_weights", "Figure 21. AUD Sovereign: the enhanced mix tilts to the belly of the curve "
                 "(5-10y), trimming the low-yield 2y and the high-vol 20y - more yield at the same duration risk."))
    m.append(story(
        found=f"In-sample the same-risk pickup is **+{intra['portfolio_return_uplift_bps']:.1f} bps** at the portfolio "
              f"level (SAA unchanged); **out-of-sample and net of cost** it is **{intra['portfolio_oos_uplift_bps']:+.1f} "
              "bps**, and it concentrates where sub-sleeve dispersion is *structural*: **Structured Credit** (CLOs by "
              "rating, ABS/RMBS/CMBS) is the standout, with **Listed Equities** (region/style) and **IG Credit** "
              "(quality/sector) also paying; **High Yield, the sovereign curve and Private Credit** are too noisy to rely "
              "on. The OOS test is exactly what separates real edge from an in-sample mirage.",
        nxt="Build structured credit out **granularly** - CLO tranches by rating and vintage, US vs EU, ABS/RMBS/CMBS "
            "sub-types on real index data - since it is both the strongest result here and the strategic growth area. "
            "Harvest only where dispersion is structural, size modestly, and rebalance slowly to keep turnover low."))

    sc = results.get("structured_credit")
    if sc is not None:
        m.append("#### B9a. Granular structured-credit deep-dive\n")
        m.append(story(
            why="Structured credit is the strategic growth area and the standout same-risk result, so it earns its own "
                "granular lens: CLOs AAA->BB (US/EU), ABS auto/card, RMBS prime/non-QM, CMBS conduit/SASB. It also splits "
                "across two capital buckets (senior ~3%, mezz ~11%), so the *capital* angle bites here specifically.",
            how="Each tranche carries a forward return/vol, a LAGIC capital category and a real index ticker hook. Risk is "
                "empirical (tranche returns share the book's securitised factor plus dispersion). Two mixes are solved: "
                "**same-risk** (max return at the benchmark volatility) and **capital-efficient** (max return per unit of "
                "LAGIC capital).",
            found=f"The same-risk mix adds **{sc['same_risk_uplift_bps']:+.0f} bps** at the benchmark volatility "
                  f"({sc['benchmark']['exp_return']:.2%} -> {sc['same_risk']['exp_return']:.2%}). The capital-efficient "
                  f"mix is more striking: by staying in **senior** tranches it lifts return on capital from "
                  f"{sc['benchmark']['return_on_capital']:.2f}x to **{sc['capital_efficient']['return_on_capital']:.2f}x** "
                  f"and cuts the capital charge from {sc['benchmark']['capital']:.2%} to {sc['capital_efficient']['capital']:.2%} "
                  "for almost the same return - the mezzanine reach is not paid for on a capital basis.",
            nxt="Wire the real CLO/ABS/RMBS/CMBS index series (the tickers are in config), add vintage and manager "
                "dimensions, and model the capital *cliff* between senior and mezz explicitly - it is the dominant "
                "consideration for an insurer growing this book."))
        m.append(img("29_structured_credit", "Figure 21b. Structured-credit tranche map (bubble = capital) and the "
                     "benchmark vs same-risk vs capital-efficient sub-allocation."))
        m.append(_df_to_md(sc["weights"].round(3)) + "\n")

    gp = results.get("glide_path")
    if gp is not None:
        gt = gp["table"]
        short_row = gt.loc[[i for i in gt.index if i.startswith("Short")][0]]
        static_row = gt.loc[[i for i in gt.index if i.startswith("Static")][0]]
        adapt_row = gt.loc[[i for i in gt.index if i.startswith("Adaptive")][0]]
        m.append("### B10. Dynamic duration glide path (through-time earnings protection)\n")
        m.append(story(
            why="Every lens so far is a *position* at a point in time. But the CFO's real lever is a *policy through "
                "time*: hold interest-rate duration early in the plan year to earn the carry needed to make the number, "
                "then wind it down as that carry is banked so you stop carrying rate risk on earnings already secured. No "
                "point-in-time optimiser can see this - it is the gap the brief singled out, and the most novel piece here.",
            means="Duration is a genuine trade-off: it **earns a term premium** (the carry you need to clear the plan) but "
                  "**bears rate risk**. A static duration over-holds risk all year; a glide winds it down on a fixed "
                  "schedule; an **adaptive** policy holds duration while *behind* the plan pace and cuts it once *ahead* - "
                  "banking the carry, then locking it in.",
            how="Stylised 12-month simulation: monthly earnings = base carry + term-premium x d_t - d_t x dr_t + non-rate "
                "noise, with base carry set below plan (so duration is needed) and rate paths block-bootstrapped from "
                "history (demeaned). The static level, glide start and adaptive trigger are each optimised to minimise the "
                "plan-miss probability under common random numbers; duration is capped at a realistic ALM limit.",
            found=f"From the same duration budget, **management style is what matters**. A short book cannot make the plan "
                  f"({short_row['prob_miss_plan']:.0%} miss). A static-high book lowers the miss to "
                  f"{static_row['prob_miss_plan']:.0%} but at high earnings volatility ({static_row['earnings_vol']:.2%}) "
                  f"and an ugly tail (EaR {static_row['earnings_at_risk_5pc']:.2%}). The **adaptive glide path is best: "
                  f"plan-miss {adapt_row['prob_miss_plan']:.0%} at the lowest earnings volatility "
                  f"({adapt_row['earnings_vol']:.2%})** - bank the carry early, de-risk once ahead.",
            nxt="Drive the policy off the *actual* P&L-book duration and liability profile (not a stylised term premium), "
                "let the optimiser choose the whole monthly schedule, and tie the trigger to the live earnings run-rate "
                "and a capital budget - a genuine dynamic asset-allocation overlay."))
        m.append(img("27_glide_path", "Figure 22. Left: duration held through the plan year by policy (the adaptive path "
                     "cuts once ahead of plan). Right: the resulting distribution of plan-year earnings."))
        m.append(_df_to_md(gt.round(4)) + "\n")

    rg = results.get("regimes")
    if rg is not None and len(rg["table"]):
        t = rg["table"]
        on, off = t.loc["Risk-on"], t.loc["Risk-off"]
        m.append("### B11. Regime-conditional risk (correlations are not constant)\n")
        m.append(story(
            why="Every other lens uses one full-sample covariance, but diversification is a fair-weather friend: in a "
                "risk-off regime correlations rise and the book is riskier than its long-run number - precisely when "
                "capital and surplus are tested. An insurer must size risk for the bad regime, not the average one.",
            means="Splitting history into **risk-on** and **risk-off** months and re-computing risk shows how much of the "
                  "headline diversification survives a crisis. The regime-conditional optimal book is what a "
                  "regime-aware investor would hold if they knew which state they were in.",
            how="A broad market proxy (equal-weight risk-asset return) defines the regime: trailing-3-month proxy return "
                "in the bottom tercile is risk-off, the rest risk-on. Covariance, average correlation, the baseline's "
                "volatility, the diversification ratio and a regime max-Sharpe are recomputed within each regime.",
            found=f"The baseline's volatility rises from **{on['baseline_vol']:.2%}** (risk-on) to "
                  f"**{off['baseline_vol']:.2%}** (risk-off) and its diversification ratio falls from "
                  f"**{on['diversification_ratio']:.2f}** to **{off['diversification_ratio']:.2f}** - the book is "
                  f"materially riskier in stress than its full-sample number suggests. The regime-optimal risk-off book "
                  "de-risks hard into senior structured credit, cash and high-grade sovereigns.",
            nxt="Replace the tercile rule with a formal **Markov-switching / HMM** classifier, feed the risk-off "
                "covariance into the capital and stress lenses (capital sized for the bad regime), and test a "
                "**regime-aware** dynamic allocation that de-risks on a regime signal."))
        m.append(img("30_regimes", "Figure 23. Risk-asset proxy with risk-off regimes shaded; and the rise in volatility "
                     "and fall in diversification when the regime turns."))
        m.append(_df_to_md(t.round(4)) + "\n")

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
    if results.get("factor_analysis"):
        fa = results["factor_analysis"]
        save_table(fa["attribution"].round(5), "factor_attribution.csv")
        save_table(fa["rolling_betas"].round(4), "factor_rolling_betas.csv")
    if results.get("pareto"):
        save_table(results["pareto"]["table"].round(5), "pareto_search.csv")
    if results.get("glide_path"):
        save_table(results["glide_path"]["table"].round(5), "glide_path.csv")
    if results.get("regimes") and len(results["regimes"]["table"]):
        save_table(results["regimes"]["table"].round(5), "regime_stats.csv")
    if results.get("reverse_stress") is not None:
        save_table(results["reverse_stress"].round(2), "reverse_stress.csv")
    if results.get("structured_credit"):
        sc = results["structured_credit"]
        save_table(sc["weights"].round(4), "structured_credit_weights.csv")
        save_table(sc["tranches"].round(4), "structured_credit_tranches.csv")
    if results.get("lagic_full"):
        lf = results["lagic_full"]
        mod = pd.Series(lf["modules"])
        mod["diversified_modules"] = lf["diversified_modules"]
        mod["concentration_addon"] = lf["concentration_addon"]
        mod["total_capital_requirement"] = lf["total_capital_requirement"]
        save_table(mod.round(5).to_frame("capital"), "lagic_full_modules.csv")

    # construction philosophies shown side by side (Lens 1)
    philosophies = [n for n in ["Equal-Weight", "Max-Sharpe", "Min-Variance",
                                "Risk-Parity", "Max-Diversification", "Black-Litterman",
                                "Robust", "ML-Forecast"] if n in portfolios]

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
    if results.get("pareto"):
        chart_pareto(results["pareto"])
    if results.get("glide_path"):
        chart_glide_path(results["glide_path"])
    if results.get("regimes") and len(results["regimes"]["table"]):
        chart_regimes(results["regimes"])
    if results.get("lagic_full"):
        chart_capital_modules(results["lagic_full"])
    if results.get("structured_credit"):
        chart_structured_credit(results["structured_credit"])
    if results.get("factor_analysis"):
        chart_factor_attribution(results["factor_analysis"]["attribution"])
        chart_rolling_factor_betas(results["factor_analysis"]["rolling_betas"])

    write_markdown(results)
    write_full_report(results)
