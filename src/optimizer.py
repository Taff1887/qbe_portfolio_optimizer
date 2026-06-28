"""Lens 1 - constrained mean-variance optimisation.

Forward expected returns come from config (`exp_return`); the covariance comes
from the return history. Constraints reflect an insurer's reality:

- long only, each asset capped (default 25%)
- minimum core fixed-income share (ALM / liability backing)
- implied risk-asset cap (= 1 - min core FI)
- per-currency minimum/maximum buckets (currency mix largely given)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_loader import MarketData
from portfolio import Portfolio


def _inputs(market: MarketData) -> tuple[pd.Series, pd.DataFrame]:
    mu = market.meta["exp_return"].astype(float)
    cov = market.returns[mu.index].cov() * 12.0
    return mu, cov


def _build(market: MarketData, config: dict, risk_cap: float | None = None):
    meta = market.meta
    n = len(meta)
    core = (meta["group"] == "core_fi").astype(float).to_numpy()
    risk = (meta["group"] == "risk").astype(float).to_numpy()
    min_core = config["portfolio"]["min_core_fixed_income"]

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    cons.append({"type": "ineq", "fun": lambda w, c=core: c @ w - min_core})
    if risk_cap is not None:
        cons.append({"type": "ineq", "fun": lambda w, r=risk: risk_cap - r @ w})
    for ccy, b in config["currencies"].items():
        ind = (meta["currency"] == ccy).astype(float).to_numpy()
        if b.get("min") is not None:
            cons.append({"type": "ineq", "fun": lambda w, i=ind, lo=b["min"]: i @ w - lo})
        if b.get("max") is not None:
            cons.append({"type": "ineq", "fun": lambda w, i=ind, hi=b["max"]: hi - i @ w})

    # Risk assets are only permitted in the eligible ("big four") currencies; force
    # risk-asset weight to zero in every other currency (e.g. NZD, CAD).
    risk_ccys = config["optimiser"].get("risk_asset_currencies")
    if risk_ccys is not None:
        for ccy in config["currencies"]:
            if ccy not in risk_ccys:
                rind = ((meta["currency"] == ccy) & (meta["group"] == "risk")).astype(float).to_numpy()
                if rind.sum() > 0:
                    cons.append({"type": "eq", "fun": lambda w, i=rind: i @ w})

    maxw = config["optimiser"]["default_max_weight"]
    bounds = [(0.0, maxw)] * n
    x0 = market.baseline_weights.to_numpy()
    return cons, bounds, x0


def _solve(objective, market, config, risk_cap=None, extra_cons=None) -> pd.Series:
    cons, bounds, x0 = _build(market, config, risk_cap)
    if extra_cons:
        cons = cons + extra_cons
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0
    return pd.Series(w, index=market.meta.index)


def min_variance(market: MarketData, config: dict) -> Portfolio:
    _, cov = _inputs(market)
    S = cov.to_numpy()
    w = _solve(lambda w: w @ S @ w, market, config)
    return Portfolio(w, market, "Min-Variance")


def max_return(market: MarketData, config: dict) -> pd.Series:
    mu, _ = _inputs(market)
    m = mu.to_numpy()
    return _solve(lambda w: -(m @ w), market, config)


def max_sharpe(market: MarketData, config: dict, rf: float | None = None) -> Portfolio:
    if rf is None:
        rf = config["portfolio"].get("risk_free", 0.0)
    mu, cov = _inputs(market)
    m, S = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ S @ w, 1e-12))
        return -((m @ w) - rf) / vol

    w = _solve(neg_sharpe, market, config)
    return Portfolio(w, market, "Max-Sharpe")


def efficient_frontier(market: MarketData, config: dict, n_points: int | None = None) -> pd.DataFrame:
    """Trace the constrained efficient frontier (forward return vs volatility)."""
    mu, cov = _inputs(market)
    m, S = mu.to_numpy(), cov.to_numpy()
    n_points = n_points or config["optimiser"].get("frontier_points", 30)

    r_lo = float(mu @ min_variance(market, config).weights)
    r_hi = float(mu @ max_return(market, config))
    targets = np.linspace(r_lo, r_hi, n_points)

    rows = []
    for t in targets:
        extra = [{"type": "eq", "fun": lambda w, tt=t: m @ w - tt}]
        w = _solve(lambda w: w @ S @ w, market, config, extra_cons=extra)
        rows.append({"target_return": t,
                     "exp_return": float(m @ w),
                     "volatility": float(np.sqrt(max(w @ S @ w, 0.0)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------- capital-aware MVO
def _capital_components(market: MarketData, config: dict):
    """Build the pieces for a SMOOTH diversified-capital charge as a function of w.

    Category charge vector c(w) = (category weight) x (prescribed stress) is linear
    in w; the diversified charge sqrt(c' R c) is then smooth and SLSQP-friendly.
    """
    meta = market.meta
    sf = config["lagic"]["stress_factors"]
    cats = list(sf.keys())
    ind = np.zeros((len(meta), len(cats)))
    for i, cat in enumerate(meta["capital_category"]):
        if cat in cats:
            ind[i, cats.index(cat)] = 1.0
    stress = np.array([sf[c] for c in cats])
    rho = config["lagic"]["cross_correlation"]
    R = np.full((len(cats), len(cats)), rho)
    np.fill_diagonal(R, 1.0)
    return ind, stress, R


def _smooth_capital(w, ind, stress, R) -> float:
    c = (ind.T @ w) * stress
    return float(np.sqrt(max(c @ R @ c, 0.0)))


def max_return_on_capital(market: MarketData, config: dict) -> Portfolio:
    """Maximise expected return per unit of (diversified) capital charge."""
    mu, _ = _inputs(market)
    m = mu.to_numpy()
    ind, stress, R = _capital_components(market, config)

    def neg_roc(w):
        cap = _smooth_capital(w, ind, stress, R)
        return -(m @ w) / cap if cap > 1e-9 else 0.0

    w = _solve(neg_roc, market, config)
    return Portfolio(w, market, "Max-RoC")


def capital_efficient_frontier(market: MarketData, config: dict, n_points: int | None = None) -> pd.DataFrame:
    """Trace expected return vs minimum capital charge (the capital frontier)."""
    mu, _ = _inputs(market)
    m = mu.to_numpy()
    ind, stress, R = _capital_components(market, config)
    n_points = n_points or config["optimiser"].get("frontier_points", 30)

    r_lo = float(mu @ min_variance(market, config).weights)
    r_hi = float(mu @ max_return(market, config))
    rows = []
    for t in np.linspace(r_lo, r_hi, n_points):
        extra = [{"type": "eq", "fun": lambda w, tt=t: m @ w - tt}]
        w = _solve(lambda w: _smooth_capital(w, ind, stress, R), market, config, extra_cons=extra)
        rows.append({"target_return": t, "exp_return": float(m @ w),
                     "capital_charge": _smooth_capital(w, ind, stress, R)})
    return pd.DataFrame(rows)


def smooth_capital(market: MarketData, config: dict, weights) -> float:
    """The smooth (diversified) capital charge for a given weight vector."""
    ind, stress, R = _capital_components(market, config)
    return _smooth_capital(np.asarray(weights, dtype=float), ind, stress, R)


def max_return_capital_budget(market: MarketData, config: dict, capital_budget: float) -> Portfolio:
    """Maximise expected return subject to capital charge <= budget (+ all the
    usual constraints). This is how an insurer really optimises: capital, not
    volatility, is the binding scarce resource."""
    mu, _ = _inputs(market)
    m = mu.to_numpy()
    ind, stress, R = _capital_components(market, config)
    extra = [{"type": "ineq", "fun": lambda w: capital_budget - _smooth_capital(w, ind, stress, R)}]
    w = _solve(lambda w: -(m @ w), market, config, extra_cons=extra)
    return Portfolio(w, market, f"Capital-Budget {capital_budget:.1%}")


def return_capital_budget_frontier(market: MarketData, config: dict, n_points: int | None = None) -> pd.DataFrame:
    """Maximum expected return achievable at each level of capital budget."""
    mu, _ = _inputs(market)
    m = mu.to_numpy()
    ind, stress, R = _capital_components(market, config)
    n_points = n_points or config["optimiser"].get("frontier_points", 30)
    lo = _smooth_capital(min_variance(market, config).weights.to_numpy(), ind, stress, R)
    hi = _smooth_capital(max_return(market, config).to_numpy(), ind, stress, R)
    rows = []
    for b in np.linspace(lo, hi, n_points):
        extra = [{"type": "ineq", "fun": lambda w, bb=b: bb - _smooth_capital(w, ind, stress, R)}]
        w = _solve(lambda w: -(m @ w), market, config, extra_cons=extra)
        rows.append({"capital_budget": b, "exp_return": float(m @ w),
                     "capital_used": _smooth_capital(w, ind, stress, R)})
    return pd.DataFrame(rows)


def run_optimisation(market: MarketData, config: dict) -> dict:
    """Convenience bundle for main.py / reporting."""
    return {
        "frontier": efficient_frontier(market, config),
        "capital_frontier": capital_efficient_frontier(market, config),
        "return_capital_budget": return_capital_budget_frontier(market, config),
        "max_sharpe": max_sharpe(market, config),
        "min_variance": min_variance(market, config),
        "max_roc": max_return_on_capital(market, config),
    }
