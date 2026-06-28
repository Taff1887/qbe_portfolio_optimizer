"""Multi-objective optimisation: search for Pareto improvements over the book.

An insurer does not have one objective - it has several that trade off: return,
volatility, **regulatory capital**, **earnings (P&L) volatility** and worst-case
**stress loss**. The interesting question is not "what is optimal" on one of them
but: *is there a portfolio that is at least as good as the current book on every
objective and strictly better on at least one?* That is a **Pareto improvement** -
"growing the pie" rather than re-slicing it.

Method (epsilon-constraint): we maximise expected return subject to hard caps on
capital and earnings volatility (and total volatility), sweeping those caps from
their minimum achievable level up to the baseline's level. Every solution is then
scored on all five objectives and tested for Pareto-dominance over the baseline.
The survivors are genuine free lunches relative to the current book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import optimizer
import stress_testing
from data_loader import MarketData
from portfolio import Portfolio

# Objective directions: +1 = higher is better, -1 = lower is better.
_DIRECTION = {
    "exp_return": +1,
    "volatility": -1,
    "capital": -1,
    "earnings_vol": -1,
    "worst_stress": +1,   # less negative is better
}


def objectives(pf: Portfolio, market: MarketData, config: dict) -> dict[str, float]:
    """The five competing objectives for one portfolio."""
    stress = stress_testing.run_stress_tests(pf, config)["total_impact"]
    return {
        "exp_return": pf.expected_return(),
        "volatility": pf.volatility(),
        "capital": optimizer.smooth_capital(market, config, pf.weights.to_numpy()),
        "earnings_vol": optimizer.earnings_volatility(market, pf.weights.to_numpy()),
        "worst_stress": float(stress.min()),
    }


def dominates(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """True if a is >= b on every objective and strictly better on at least one."""
    better_or_equal, strictly_better = True, False
    for k, d in _DIRECTION.items():
        delta = d * (a[k] - b[k])
        if delta < -tol:
            better_or_equal = False
            break
        if delta > tol:
            strictly_better = True
    return better_or_equal and strictly_better


def pareto_search(market: MarketData, config: dict, baseline: Portfolio,
                  seeds: dict[str, Portfolio] | None = None, grid: int = 4) -> dict:
    """Find portfolios that Pareto-dominate the baseline.

    `seeds` are already-built portfolios (the named optimisers) to include as
    candidates for free; `grid` controls the epsilon-constraint sweep density.
    """
    base_obj = objectives(baseline, market, config)

    # Floors: the best individually achievable capital and earnings vol set the
    # bottom of the sweep; the baseline level sets the top.
    cap_lo = optimizer.smooth_capital(
        market, config, optimizer.max_return_on_capital(market, config).weights.to_numpy())
    ev_lo = optimizer.earnings_volatility(
        market, optimizer.min_earnings_volatility(market, config).weights.to_numpy())
    cap_hi, ev_hi = base_obj["capital"], base_obj["earnings_vol"]
    vol_cap = base_obj["volatility"]

    candidates: dict[str, Portfolio] = dict(seeds or {})
    for cb in np.linspace(cap_lo, cap_hi, grid):
        for evb in np.linspace(ev_lo, ev_hi, grid):
            pf = optimizer.max_return_constrained(
                market, config, capital_budget=float(cb), earnings_vol_budget=float(evb),
                vol_budget=vol_cap, name=f"eps c={cb:.3f} e={evb:.3f}")
            candidates[pf.name] = pf

    rows, dominating = [], {}
    for name, pf in candidates.items():
        o = objectives(pf, market, config)
        is_dom = dominates(o, base_obj)
        rows.append({"portfolio": name, **o, "dominates_baseline": is_dom})
        if is_dom:
            dominating[name] = pf

    table = pd.DataFrame(rows).set_index("portfolio")
    # Rank dominating books by total improvement vs baseline (sum of signed,
    # scale-free gains across objectives) and surface the best balanced one.
    best = None
    if dominating:
        def score(o):
            return sum(_DIRECTION[k] * (o[k] - base_obj[k]) / (abs(base_obj[k]) + 1e-9)
                       for k in _DIRECTION)
        best = max(dominating, key=lambda n: score(objectives(dominating[n], market, config)))

    return {
        "baseline_objectives": base_obj,
        "table": table,
        "dominating": dominating,
        "n_dominating": len(dominating),
        "best_balanced": best,
        "best_portfolio": dominating.get(best) if best else None,
    }
