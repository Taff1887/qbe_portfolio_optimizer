"""Return-series risk/performance metrics shared across the framework.

These are deliberately small, pure functions that take a monthly return series
(and, for tracking error, a benchmark series) and return a single number. They
complement the headline helpers in `utils.py` (annualise_return, annualise_vol,
sharpe_ratio, max_drawdown) with the downside- and tail-risk measures an
institutional comparison table needs: Sortino, historical VaR/CVaR and tracking
error. Keeping them here means every optimiser is scored on exactly the same
metric definitions.

Convention: VaR and CVaR are reported as *monthly* returns at the given
confidence level, so a 95% VaR of -0.03 reads as "a 1-in-20 month loses ~3%".
They are returned as signed returns (negative = loss) to stay consistent with
`max_drawdown`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import MONTHS_PER_YEAR


def downside_deviation(monthly: pd.Series, mar_annual: float = 0.0) -> float:
    """Annualised downside deviation below a minimum acceptable return (MAR)."""
    m = monthly.dropna()
    if m.empty:
        return float("nan")
    mar_m = mar_annual / MONTHS_PER_YEAR
    shortfall = np.minimum(m - mar_m, 0.0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(MONTHS_PER_YEAR))


def sortino_ratio(monthly: pd.Series, rf_annual: float = 0.0, mar_annual: float | None = None) -> float:
    """Annualised Sortino ratio: excess return over downside deviation.

    The MAR defaults to the risk-free rate, so Sortino and Sharpe share a
    reference point and differ only in penalising downside vs total volatility.
    """
    m = monthly.dropna()
    mar_annual = rf_annual if mar_annual is None else mar_annual
    dd = downside_deviation(m, mar_annual)
    if len(m) < 2 or dd == 0 or np.isnan(dd):
        return float("nan")
    excess_annual = (m.mean() * MONTHS_PER_YEAR) - rf_annual
    return float(excess_annual / dd)


def value_at_risk(monthly: pd.Series, level: float = 0.95) -> float:
    """Historical Value-at-Risk: the (1-level) quantile of monthly returns.

    Returned as a signed monthly return (negative = loss).
    """
    m = monthly.dropna()
    if m.empty:
        return float("nan")
    return float(np.percentile(m, (1.0 - level) * 100.0))


def conditional_value_at_risk(monthly: pd.Series, level: float = 0.95) -> float:
    """Historical Conditional VaR (expected shortfall): mean of returns at or
    below the VaR threshold - the average loss in the worst (1-level) tail."""
    m = monthly.dropna()
    if m.empty:
        return float("nan")
    var = value_at_risk(m, level)
    tail = m[m <= var]
    return float(tail.mean()) if not tail.empty else var


def tracking_error(monthly: pd.Series, benchmark_monthly: pd.Series) -> float:
    """Annualised tracking error: std of active (portfolio - benchmark) returns."""
    active = (monthly - benchmark_monthly).dropna()
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
