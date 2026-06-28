"""Lens 5 - simplified APRA / LAGIC-style asset risk charge.

NOT legally exact, but structured the way the prudential capital framework works:
each capital category carries a prescribed stress (a stressed loss). We compute

1. standalone charge per asset / category  (weight x prescribed stress)
2. a diversified aggregate charge           (correlated combination of categories)
3. a stress-scenario panel and the WORST single-scenario charge (the binding one)
4. return on capital                        (expected return / capital charge)
5. marginal capital contribution by asset   (who consumes the binding capital)

The capital charge is expressed as a % of the portfolio (weights already sum to
1), so it reads directly as "cents of capital per dollar of assets".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import Portfolio


def _category_weights(portfolio: Portfolio) -> pd.Series:
    return portfolio.weights.groupby(portfolio.meta["capital_category"]).sum()


def category_charges(portfolio: Portfolio, config: dict) -> pd.Series:
    """Standalone capital charge for each capital category."""
    sf = config["lagic"]["stress_factors"]
    cw = _category_weights(portfolio)
    return pd.Series({c: cw.get(c, 0.0) * f for c, f in sf.items()})


def asset_charges(portfolio: Portfolio, config: dict) -> pd.Series:
    """Standalone capital charge attributable to each asset."""
    sf = config["lagic"]["stress_factors"]
    factor = portfolio.meta["capital_category"].map(sf).fillna(0.0)
    return (portfolio.weights * factor).sort_values(ascending=False)


def diversified_charge(cat_charges: pd.Series, config: dict) -> float:
    """Correlated aggregate of category charges: sqrt(c' R c)."""
    c = cat_charges.to_numpy()
    rho = config["lagic"]["cross_correlation"]
    k = len(c)
    R = np.full((k, k), rho)
    np.fill_diagonal(R, 1.0)
    return float(np.sqrt(max(c @ R @ c, 0.0)))


def scenario_charges(portfolio: Portfolio, config: dict) -> pd.Series:
    """Capital loss under each panel scenario, using the SAME correlated
    aggregation as `diversified_charge` (so the scenario panel and the aggregate
    charge are consistent and comparable)."""
    cat = category_charges(portfolio, config)
    rows = {}
    for sc in config["lagic"]["scenarios"]:
        mult = sc["multipliers"]
        scen_vec = pd.Series({c: cat.get(c, 0.0) * mult.get(c, 0.0) for c in cat.index})
        rows[sc["name"]] = diversified_charge(scen_vec, config)
    return pd.Series(rows)


def marginal_capital(portfolio: Portfolio, config: dict, scenario_name: str) -> pd.Series:
    """Per-asset contribution to the binding (worst) scenario charge."""
    sc = next(s for s in config["lagic"]["scenarios"] if s["name"] == scenario_name)
    mult, sf = sc["multipliers"], config["lagic"]["stress_factors"]
    eff = portfolio.meta["capital_category"].map(lambda c: sf.get(c, 0.0) * mult.get(c, 0.0))
    return (portfolio.weights * eff).sort_values(ascending=False)


def run_lagic(portfolio: Portfolio, config: dict) -> dict:
    """Full capital-lens bundle for a portfolio."""
    cat = category_charges(portfolio, config)
    scen = scenario_charges(portfolio, config)
    worst_name = scen.idxmax()
    worst_charge = float(scen.max())
    div = diversified_charge(cat, config)
    # LAGIC-style asset risk charge = the WORST single scenario in the prescribed
    # panel (the binding scenario). The full diversified aggregate ("all categories
    # stressed at once") is reported alongside as a conservative comparator.
    capital_charge = worst_charge
    binding_basis = worst_name
    exp_ret = portfolio.expected_return()
    return {
        "asset_charges": asset_charges(portfolio, config),
        "category_charges": cat.sort_values(ascending=False),
        "scenario_charges": scen.sort_values(ascending=False),
        "diversified_charge": div,
        "worst_scenario": worst_name,
        "binding_basis": binding_basis,
        "capital_charge": capital_charge,            # % of portfolio (worst of panel)
        "return_on_capital": exp_ret / capital_charge if capital_charge > 0 else np.nan,
        "marginal_capital": marginal_capital(portfolio, config, worst_name),
    }
