"""Extra lens - earnings-at-risk (forward plan-year P&L tail).

An insurer is judged on whether it makes its annual *earnings* plan. Point-in-time
volatility under-states this; we want the distribution of next-year P&L and its
downside tail.

We build that distribution by **block-bootstrapping the realised monthly P&L**
(from the earnings model) into many synthetic 12-month years. This resamples
*observed* returns (preserving short-run autocorrelation) rather than assuming a
parametric distribution - the honest way to get a tail. We then report:

- expected annual P&L and its volatility
- probability of missing the plan
- earnings-at-risk (5th percentile of annual P&L)
- CTE95 (mean of the worst 5% of years - the conditional tail)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from earnings_model import decompose_earnings
from portfolio import Portfolio
from utils import MONTHS_PER_YEAR


def annual_pnl_distribution(portfolio: Portfolio, n_boot: int = 5000, block: int = 3, seed: int = 42) -> np.ndarray:
    """Bootstrap distribution of plan-year (12-month) P&L from realised monthly P&L."""
    monthly = decompose_earnings(portfolio)["pnl"].dropna().to_numpy()
    n = len(monthly)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx: list[int] = []
        while len(idx) < MONTHS_PER_YEAR:                # draw 12 months in blocks (circular)
            s = int(rng.integers(0, n))
            idx.extend(((s + np.arange(block)) % n).tolist())
        out[b] = monthly[np.array(idx[:MONTHS_PER_YEAR])].sum()
    return out


def earnings_at_risk(portfolio: Portfolio, config: dict, n_boot: int = 5000) -> dict:
    dist = annual_pnl_distribution(portfolio, n_boot=n_boot)
    plan = config["portfolio"]["plan_return_target"]
    p5 = float(np.percentile(dist, 5))
    return {
        "mean_annual_pnl": float(dist.mean()),
        "pnl_volatility": float(dist.std(ddof=1)),
        "plan_target": plan,
        "prob_miss_plan": float((dist < plan).mean()),
        "earnings_at_risk_5pc": p5,                         # 5th percentile of annual P&L
        "cte_95": float(dist[dist <= p5].mean()),           # mean of worst 5% (conditional tail)
        "upside_95pc": float(np.percentile(dist, 95)),
    }


def run_earnings_risk(portfolios: dict[str, Portfolio], config: dict) -> dict:
    """Earnings-at-risk table across portfolios + the baseline distribution."""
    rows = {name: earnings_at_risk(pf, config) for name, pf in portfolios.items()}
    table = pd.DataFrame(rows).T
    base = "Baseline" if "Baseline" in portfolios else next(iter(portfolios))
    return {"table": table, "baseline_distribution": annual_pnl_distribution(portfolios[base], config.get("_n_boot", 5000))}
