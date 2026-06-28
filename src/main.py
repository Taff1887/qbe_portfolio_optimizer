"""Entry point: run the full multi-lens analysis and write outputs.

    python src/main.py

Generates dummy data (if missing), builds the baseline insurer portfolio, then
runs all six lenses (MVO, stress, capital, earnings, duration) and saves tables,
charts and a markdown summary to outputs/.
"""

from __future__ import annotations

import construction
import diagnostics
import duration_model
import earnings_model
import earnings_risk
import factor_analysis
import glide_path
import intra_asset
import lagic_capital
import multi_objective
import optimizer
import regimes
import reporting
import risk_attribution
import stress_testing
import structured_credit
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
    base_capital = optimizer.smooth_capital(market, config, base.weights.to_numpy())
    portfolios = {
        "Baseline": base,
        # construction philosophies (Lens 1) - scored on the same metrics
        "Equal-Weight": construction.equal_weight(market, config),
        "Max-Sharpe": optimizer.max_sharpe(market, config),
        "Min-Variance": optimizer.min_variance(market, config),
        "Risk-Parity": construction.risk_parity(market, config),
        "Max-Diversification": construction.max_diversification(market, config),
        # views / robust / ML-driven construction philosophies
        "Black-Litterman": construction.black_litterman(market, config),
        "Robust": construction.robust_optimizer(market, config),
        "ML-Forecast": construction.ml_forecast(market, config),
        # capital-aware optimisers
        "Max-RoC": optimizer.max_return_on_capital(market, config),
        # most return achievable at the SAME capital charge as the baseline
        "Capital-Budgeted": optimizer.max_return_capital_budget(market, config, base_capital),
        "Risk 20%": risk20,
    }

    # --- multi-objective: search for Pareto improvements over the current book ---
    min_evol = optimizer.min_earnings_volatility(market, config)
    pareto = multi_objective.pareto_search(market, config, base, seeds={
        "Max-Sharpe": portfolios["Max-Sharpe"],
        "Min-Variance": portfolios["Min-Variance"],
        "Min-EarningsVol": min_evol,
        "Max-RoC": portfolios["Max-RoC"],
        "Risk-Parity": portfolios["Risk-Parity"],
    }, grid=4)
    portfolios["Min-EarningsVol"] = min_evol
    if pareto["best_portfolio"] is not None:
        pareto["best_portfolio"].name = "Pareto-Balanced"
        portfolios["Pareto-Balanced"] = pareto["best_portfolio"]

    stress = stress_testing.run_stress_tests(base, config)
    worst_total = float(stress["total_impact"].min())
    stress_grid = stress_testing.stress_matrix(portfolios, config)
    worst_stress = stress_testing.worst_stress_by_portfolio(portfolios, config)
    dur = duration_model.run_duration(base, config)
    lagic_full = lagic_capital.lagic_full(base, config, dur["total_dollar_duration_gap"])

    # --- the six core lenses + extra analyses ---
    return {
        "config": config,
        "market": market,
        "portfolios": portfolios,
        "comparison": reporting.comparison_table(portfolios, config, benchmark=base, stress_losses=worst_stress),
        "frontier": optimizer.efficient_frontier(market, config),
        "capital_frontier": optimizer.capital_efficient_frontier(market, config),
        "stress": stress,
        "stress_grid": stress_grid,
        "worst_stress": worst_stress,
        "reverse_stress": stress_testing.reverse_stress(base, config, dur["total_dollar_duration_gap"]),
        "lagic": {name: lagic_capital.run_lagic(pf, config) for name, pf in portfolios.items()},
        "lagic_full": lagic_full,
        "earnings": earnings_model.run_earnings(base, config),
        "duration": dur,
        "duration_example": earnings_model.duration_earnings_example(config),
        "return_capital_budget": optimizer.return_capital_budget_frontier(market, config),
        "risk_attribution": risk_attribution.run_risk_attribution(base, config),
        "diagnostics": diagnostics.run_diagnostics(base, config, worst_total),
        "intra_asset": intra_asset.run_intra_asset(market, config),
        "earnings_risk": earnings_risk.run_earnings_risk(portfolios, config),
        "factor_analysis": factor_analysis.run_factor_analysis(base, config),
        "pareto": pareto,
        "glide_path": glide_path.run_glide_path(config),
        "structured_credit": structured_credit.run_structured_credit(market, config),
        "regimes": regimes.run_regimes(market, config, base),
    }


def print_summary(results: dict) -> None:
    base = results["portfolios"]["Baseline"]
    stress = results["stress"]
    lagic = results["lagic"]["Baseline"]
    earn = results["earnings"]["summary"]
    dur = results["duration"]
    worst = stress["total_impact"].idxmin()

    comp = results["comparison"]
    print("\n" + "=" * 68)
    print("QBE-STYLE PORTFOLIO OPTIMISATION RESEARCH LAB")
    print("=" * 68)
    print(f"Baseline: {base.core_fi_share():.0%} core FI / {base.risk_asset_share():.0%} risk  |  "
          f"exp return {base.expected_return():.2%}  vol {base.volatility():.2%}  "
          f"carry {base.carry():.2%}  duration {base.duration():.1f}y")
    print("Construction philosophies (exp return / vol / Sharpe):")
    for name in ["Equal-Weight", "Max-Sharpe", "Min-Variance", "Risk-Parity", "Max-Diversification"]:
        if name in comp.index:
            r = comp.loc[name]
            print(f"  {name:<20} {r['exp_return']:.2%} / {r['volatility']:.2%} / {r['sharpe']:.2f}")
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
    cap = results["portfolios"]["Capital-Budgeted"]
    print(f"Extra   Cap-budget : at the baseline's capital, max return rises to {cap.expected_return():.2%} "
          f"(vs {base.expected_return():.2%})")
    er = results["earnings_risk"]["table"].loc["Baseline"]
    print(f"Extra   Earn@risk  : EaR(5%) {er['earnings_at_risk_5pc']:.2%}  CTE95 {er['cte_95']:.2%}  "
          f"P(miss plan) {er['prob_miss_plan']:.0%}")
    ia = results["intra_asset"]
    print(f"Extra   Intra-class: same-risk pickup {ia['portfolio_return_uplift_bps']:+.1f}bps in-sample, "
          f"{ia['portfolio_oos_uplift_bps']:+.1f}bps out-of-sample net of cost (annual rebal, SAA unchanged)")
    print("=" * 68)
    print("Outputs written to outputs/ (tables/, charts/, summary_report.md)")


def main() -> None:
    results = build_results()
    reporting.generate_all(results)
    print_summary(results)


if __name__ == "__main__":
    main()
