"""Extra lens - risk budgeting, diversification and marginal efficiency.

Volatility is a single number; *where it comes from* is what a risk team manages.
This module decomposes portfolio risk into each asset's contribution (and rolls
it up by currency / capital category), measures diversification, and ranks assets
by their efficiency at the margin (return per unit of marginal risk and capital).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import Portfolio


def risk_contributions(portfolio: Portfolio) -> pd.Series:
    """Each asset's share of total portfolio volatility (sums to 1).

    Marginal contribution to risk MCTR_i = w_i * (Sigma w)_i / vol; these sum to
    the portfolio volatility, so dividing by vol gives percentage shares.
    """
    w = portfolio.weights.to_numpy()
    cov = portfolio.covariance().to_numpy()
    vol = np.sqrt(max(w @ cov @ w, 1e-18))
    mctr = w * (cov @ w) / vol
    return pd.Series(mctr / vol, index=portfolio.weights.index)


def risk_contribution_by(portfolio: Portfolio, field: str) -> pd.Series:
    """Risk contribution grouped by a metadata field (currency, capital_category)."""
    rc = risk_contributions(portfolio)
    return rc.groupby(portfolio.meta[field]).sum().sort_values(ascending=False)


def diversification_ratio(portfolio: Portfolio) -> float:
    """(weighted average asset vol) / (portfolio vol). 1 = no diversification."""
    w = portfolio.weights.to_numpy()
    cov = portfolio.covariance().to_numpy()
    sig = np.sqrt(np.diag(cov))
    port_vol = np.sqrt(max(w @ cov @ w, 1e-18))
    return float((w @ sig) / port_vol)


def correlation_matrix(portfolio: Portfolio) -> pd.DataFrame:
    """Correlation matrix of the held assets (weight > 0)."""
    held = portfolio.weights[portfolio.weights > 1e-6].index
    return portfolio.market.returns[held].corr()


def avg_pairwise_correlation(portfolio: Portfolio) -> float:
    corr = correlation_matrix(portfolio).to_numpy()
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    off = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off))


def marginal_efficiency(portfolio: Portfolio, config: dict) -> pd.DataFrame:
    """Per-asset marginal return, risk and capital, and the efficiency ratios.

    Marginal risk = (Sigma w)_i / vol (the vol added per unit of weight).
    Marginal capital = the asset's prescribed LAGIC stress factor.
    """
    w = portfolio.weights.to_numpy()
    cov = portfolio.covariance().to_numpy()
    vol = np.sqrt(max(w @ cov @ w, 1e-18))
    marg_risk = (cov @ w) / vol
    sf = config["lagic"]["stress_factors"]
    marg_cap = portfolio.meta["capital_category"].map(sf).fillna(0.0).to_numpy()
    mu = portfolio.meta["exp_return"].to_numpy()

    df = pd.DataFrame({
        "weight": w,
        "exp_return": mu,
        "marginal_risk": marg_risk,
        "marginal_capital": marg_cap,
    }, index=portfolio.weights.index)
    df["return_per_risk"] = df["exp_return"] / df["marginal_risk"].replace(0, np.nan)
    df["return_per_capital"] = df["exp_return"] / df["marginal_capital"].replace(0, np.nan)
    return df.sort_values("return_per_capital", ascending=False)


def earnings_vol_contribution(portfolio: Portfolio) -> pd.Series:
    """Each asset's share of annual *earnings* (P&L) volatility (sums to 1).

    Builds each asset's monthly P&L contribution - w_i x (carry_i + price MtM_i
    if it is a P&L-bucket asset; carry only if OCI) - and attributes the variance
    of total P&L to assets via cov(contribution_i, total_P&L)/var(total_P&L).
    Carry is constant so contributes no variance; the result shows which sleeves
    (typically long-duration P&L bonds) drive earnings instability.
    """
    meta, w = portfolio.meta, portfolio.weights
    rets = portfolio.market.returns[w.index]
    carry_i = meta["yield"] / 12.0
    mtm_i = rets - carry_i
    is_pnl = (meta["accounting"] == "P&L").reindex(w.index)
    # per-asset monthly P&L contribution (carry on all; price MtM only on P&L bucket)
    contrib = (carry_i * w) + (mtm_i.loc[:, is_pnl] * w[is_pnl]).reindex(columns=w.index, fill_value=0.0)
    total = contrib.sum(axis=1)
    var_total = total.var(ddof=1)
    if var_total <= 0:
        return pd.Series(0.0, index=w.index)
    share = contrib.apply(lambda col: col.cov(total)) / var_total
    return share.sort_values(ascending=False)


def run_risk_attribution(portfolio: Portfolio, config: dict) -> dict:
    return {
        "risk_by_asset": risk_contributions(portfolio).sort_values(ascending=False),
        "risk_by_currency": risk_contribution_by(portfolio, "currency"),
        "risk_by_category": risk_contribution_by(portfolio, "capital_category"),
        "diversification_ratio": diversification_ratio(portfolio),
        "avg_pairwise_correlation": avg_pairwise_correlation(portfolio),
        "correlation_matrix": correlation_matrix(portfolio),
        "marginal_efficiency": marginal_efficiency(portfolio, config),
        "earnings_vol_contribution": earnings_vol_contribution(portfolio),
    }
