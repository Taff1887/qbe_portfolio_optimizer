"""Entry point: run the full multi-lens analysis and write outputs.

    python src/main.py

Generates dummy data (if missing), builds the baseline insurer portfolio, then
runs all six lenses (MVO, stress, capital, earnings, duration) and saves tables,
charts and a markdown summary to outputs/.
"""

from __future__ import annotations

import diagnostics
import duration_model
import earnings_model
import lagic_capital
import optimizer
import reporting
import risk_attribution
import stress_testing
from data_loader import load_market_data
from portfolio import baseline_portfolio
from utils import ensure_dirs, load_config, set_plot_style


def build_results(regenerate: bool = False) -> dict:
    """Run every lens and return one results bundle for reporting."""
    ensure_dirs()
    set_plot_style()
    config = load_config()
    market = load_market_data(regenerate=regenerate, config=config)

    # --- portfolios ---
    base = baseline_portfolio(market)
    risk20 = base.scale_risk_assets(config["portfolio"]["risk_asset_scaled"], name="Risk 20%")
    portfolios = {
        "Baseline": base,
        "Max-Sharpe": optimizer.max_sharpe(market, config),
        "Min-Variance": optimizer.min_variance(market, config),
        "Max-RoC": optimizer.max_return_on_capital(market, config),
        "Risk 20%": risk20,
    }

    stress = stress_testing.run_stress_tests(base, config)
    worst_total = float(stress["total_impact"].min())

    # --- the six core lenses + extra analyses ---
    return {
        "config": config,
        "market": market,
        "portfolios": portfolios,
        "comparison": reporting.comparison_table(portfolios, config),
        "frontier": optimizer.efficient_frontier(market, config),
        "capital_frontier": optimizer.capital_efficient_frontier(market, config),
        "stress": stress,
        "lagic": {name: lagic_capital.run_lagic(pf, config) for name, pf in portfolios.items()},
        "earnings": earnings_model.run_earnings(base, config),
        "duration": duration_model.run_duration(base, config),
        "duration_example": earnings_model.duration_earnings_example(config),
        "risk_attribution": risk_attribution.run_risk_attribution(base, config),
        "diagnostics": diagnostics.run_diagnostics(base, config, worst_total),
    }


def print_summary(results: dict) -> None:
    base = results["portfolios"]["Baseline"]
    stress = results["stress"]
    lagic = results["lagic"]["Baseline"]
    earn = results["earnings"]["summary"]
    dur = results["duration"]
    worst = stress["total_impact"].idxmin()

    print("\n" + "=" * 68)
    print("QBE-STYLE MULTI-LENS PORTFOLIO ANALYSIS")
    print("=" * 68)
    print(f"Baseline: {base.core_fi_share():.0%} core FI / {base.risk_asset_share():.0%} risk  |  "
          f"exp return {base.expected_return():.2%}  vol {base.volatility():.2%}  "
          f"carry {base.carry():.2%}  duration {base.duration():.1f}y")
    print(f"Lens 1  MVO        : Max-Sharpe exp return {results['portfolios']['Max-Sharpe'].expected_return():.2%} "
          f"at vol {results['portfolios']['Max-Sharpe'].volatility():.2%}")
    print(f"Lens 2  Stress     : worst = {worst}  total {stress.loc[worst,'total_impact']:.2%} "
          f"(P&L {stress.loc[worst,'pnl_impact']:.2%} / OCI {stress.loc[worst,'oci_impact']:.2%})")
    print(f"Lens 4  Duration   : total $-duration gap {dur['total_dollar_duration_gap']:+.2f}y  |  "
          f"+100bp -> economic {dur['rate_shock']['economic_surplus_impact']:+.2%} vs "
          f"P&L {dur['rate_shock']['pnl_earnings_impact']:+.2%}")
    print(f"Lens 5  Capital    : LAGIC charge {lagic['capital_charge']:.2%} of assets "
          f"(binding: {lagic['binding_basis']})  |  return on capital {lagic['return_on_capital']:.2f}x")
    print(f"Lens 6  Earnings   : annual earnings vol {earn['earnings_volatility']:.2%}  |  "
          f"P(miss {earn['plan_target']:.1%} plan) {earn['plan_miss_prob']:.0%}  |  "
          f"carry funds {earn['carry_share_of_return']:.0%} of return")
    print("=" * 68)
    print("Outputs written to outputs/ (tables/, charts/, summary_report.md)")


def main() -> None:
    results = build_results()
    reporting.generate_all(results)
    print_summary(results)


if __name__ == "__main__":
    main()
