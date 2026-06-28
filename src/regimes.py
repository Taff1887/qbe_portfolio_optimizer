"""Lens - regime-conditional risk (correlations are not constant).

Every other lens here uses a single, full-sample covariance. But diversification
is a fair-weather friend: in a risk-off regime correlations rise, the
diversification ratio collapses, and the book is riskier than its long-run number
suggests - exactly when an insurer's capital and surplus are tested. This lens
splits history into **risk-on** and **risk-off** regimes and re-computes the risk
picture, plus the regime-conditional optimal portfolio.

Regime definition (transparent, no extra dependencies): a broad market proxy =
the equal-weight return of the risk-asset sleeves; months whose trailing 3-month
proxy return is in the bottom tercile are **risk-off**, the rest **risk-on**.
Replace with a formal Markov-switching / HMM classifier later; the conditional
analysis is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_loader import MarketData
from optimizer import _build, _inputs
from portfolio import Portfolio


def classify_regimes(market: MarketData, config: dict) -> pd.Series:
    """Label each month risk-on / risk-off from a broad risk-asset proxy."""
    risk = market.meta.index[market.meta["group"] == "risk"]
    cols = [c for c in risk if c in market.returns.columns] or list(market.returns.columns)
    proxy = market.returns[cols].mean(axis=1)
    trail = proxy.rolling(3, min_periods=1).mean()
    q = config.get("regimes", {}).get("risk_off_quantile", 0.33)
    thr = trail.quantile(q)
    return pd.Series(np.where(trail <= thr, "Risk-off", "Risk-on"), index=market.returns.index, name="regime")


def _avg_offdiag_corr(corr: np.ndarray) -> float:
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    return float(corr[~np.eye(n, dtype=bool)].mean())


def _max_sharpe_with_cov(market: MarketData, config: dict, S: np.ndarray) -> np.ndarray:
    """Constrained max-Sharpe using a supplied covariance (regime-specific)."""
    mu, _ = _inputs(market)
    m = mu.to_numpy()
    rf = config["portfolio"].get("risk_free", 0.0)
    cons, bounds, x0 = _build(market, config)

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ S @ w, 1e-12))
        return -((m @ w) - rf) / vol

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else x0


def run_regimes(market: MarketData, config: dict, base: Portfolio) -> dict:
    regime = classify_regimes(market, config)
    cols = list(base.weights.index)
    w = base.weights.to_numpy()
    sig_full = np.sqrt(np.diag(market.returns[cols].cov().to_numpy() * 12.0))

    rows, weights, corrs = {}, {}, {}
    for r in ["Risk-on", "Risk-off"]:
        sub = market.returns.loc[regime == r, cols]
        if len(sub) < 6:
            continue
        cov = sub.cov().to_numpy() * 12.0
        corr = sub.corr().to_numpy()
        port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        div_ratio = float((w @ sig_full) / port_vol) if port_vol > 0 else np.nan
        w_opt = _max_sharpe_with_cov(market, config, cov)
        rows[r] = {
            "months": int((regime == r).sum()),
            "avg_correlation": _avg_offdiag_corr(corr),
            "baseline_vol": port_vol,
            "diversification_ratio": div_ratio,
            "mean_monthly_return": float(sub.mean().mean()),
        }
        weights[r] = pd.Series(w_opt, index=cols)
        corrs[r] = pd.DataFrame(corr, index=cols, columns=cols)

    table = pd.DataFrame(rows).T
    risk = market.meta.index[market.meta["group"] == "risk"]
    proxy = market.returns[[c for c in risk if c in market.returns.columns]].mean(axis=1)
    return {
        "regime": regime,
        "proxy": proxy,
        "table": table,
        "optimal_weights": weights,
        "correlations": corrs,
    }
