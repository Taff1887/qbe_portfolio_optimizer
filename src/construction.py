"""Portfolio construction philosophies (Lens 1, extended).

The point of the lab is to *compare* construction philosophies, not to crown one
winner. `optimizer.py` already implements the return-driven and capital-driven
optimisers (max-Sharpe, min-variance, max-return-on-capital, capital-budgeted).
This module adds the classic risk-driven and naive constructions so every
philosophy is scored on the same metrics:

- **Equal weight**        - the naive 1/N benchmark (no optimisation, no mandate
                            constraints): the bar every optimiser must beat.
- **Risk parity**         - equal risk contribution from every asset.
- **Maximum diversification** - maximise the diversification ratio
                            (weighted-average asset vol / portfolio vol).

Risk parity and maximum diversification are solved under the *same* insurer
constraints as the mean-variance optimisers (long-only, per-asset cap, minimum
core fixed income, currency buckets) - reusing `optimizer._build`/`_solve` - so
the comparison is like-for-like within the insurer's mandate. Equal weight is the
deliberate exception: it is the unconstrained naive benchmark.

Two further philosophies are scaffolded as documented placeholders for the
roadmap (`black_litterman`, `robust_optimizer`); they raise NotImplementedError
until wired up, so they are importable and discoverable without producing
misleading numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_loader import MarketData
from optimizer import _build, _inputs, _solve
from portfolio import Portfolio


def _max_sharpe_given_mu(market: MarketData, config: dict, mu_vec, name: str) -> Portfolio:
    """Constrained max-Sharpe using a *supplied* expected-return vector (the shared
    engine for Black-Litterman, robust and ML, which differ only in how mu is formed)."""
    _, cov = _inputs(market)
    S = cov.to_numpy()
    m = np.asarray(mu_vec, dtype=float)
    rf = config["portfolio"].get("risk_free", 0.0)

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ S @ w, 1e-12))
        return -((m @ w) - rf) / vol

    w = _solve(neg_sharpe, market, config)
    return Portfolio(w, market, name)


def equal_weight(market: MarketData, config: dict) -> Portfolio:
    """Naive 1/N portfolio across the whole universe (no mandate constraints).

    This is the benchmark of last resort: famously hard to beat out-of-sample
    because it takes no estimation risk. Included so every optimiser can be
    measured against doing nothing clever at all.
    """
    n = len(market.meta)
    w = pd.Series(1.0 / n, index=market.meta.index)
    return Portfolio(w, market, "Equal-Weight")


def risk_parity(market: MarketData, config: dict) -> Portfolio:
    """Equal-risk-contribution portfolio (each asset adds the same risk).

    Minimises the dispersion of the assets' risk contributions
    RC_i = w_i (Sigma w)_i, subject to the insurer constraint set. At the optimum
    every asset contributes an equal share of portfolio volatility, so risk is
    spread by *risk* rather than by *capital*.
    """
    _, cov = _inputs(market)
    S = cov.to_numpy()
    n = S.shape[0]
    cons, bounds, _ = _build(market, config)

    def dispersion(w: np.ndarray) -> float:
        # Risk *fractions* p_i = w_i (Sigma w)_i / (w' Sigma w) sum to 1; equal risk
        # contribution means every p_i = 1/n. Working in fractions (O(1/n)) instead
        # of raw contributions (O(1e-3)) keeps the gradient well scaled for SLSQP.
        port_var = max(w @ S @ w, 1e-18)
        p = w * (S @ w) / port_var
        return float(np.sum((p - 1.0 / n) ** 2) * 100.0)

    # Inverse-volatility seed - the natural risk-parity starting point - rather than
    # the baseline, so the solver does not stall at a flat baseline gradient.
    inv_vol = 1.0 / np.sqrt(np.diag(S))
    x0 = inv_vol / inv_vol.sum()
    res = minimize(dispersion, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0
    return Portfolio(pd.Series(w, index=market.meta.index), market, "Risk-Parity")


def max_diversification(market: MarketData, config: dict) -> Portfolio:
    """Maximum-diversification portfolio (maximise the diversification ratio).

    The diversification ratio is (weighted-average stand-alone asset vol) /
    (portfolio vol); maximising it pushes weight toward assets that are cheap in
    correlation terms. Solved under the same constraints as the MVO optimisers.
    """
    _, cov = _inputs(market)
    S = cov.to_numpy()
    sig = np.sqrt(np.diag(S))

    def neg_div_ratio(w: np.ndarray) -> float:
        port_vol = np.sqrt(max(w @ S @ w, 1e-18))
        return -float((w @ sig) / port_vol)

    w = _solve(neg_div_ratio, market, config)
    return Portfolio(w, market, "Max-Diversification")


# ------------------------------------------------- views / robust / ML-driven
def black_litterman(market: MarketData, config: dict) -> Portfolio:
    """Black-Litterman: blend equilibrium returns with explicit views.

    Reverse-optimise **equilibrium** implied returns from the baseline (market)
    weights and covariance, Pi = delta * Sigma * w_mkt, then Bayesian-update them
    with any views in `config['black_litterman']['views']` before running the
    constrained max-Sharpe. With no views the result is the equilibrium portfolio
    (a useful anchor in its own right).
    """
    _, cov = _inputs(market)
    S = cov.to_numpy()
    names = list(market.meta.index)
    w_mkt = market.baseline_weights.reindex(names).fillna(0.0).to_numpy()
    rf = config["portfolio"].get("risk_free", 0.0)
    bl = config.get("black_litterman", {})
    tau = bl.get("tau", 0.05)

    mu_cma = market.meta["exp_return"].to_numpy()
    base_excess = float(mu_cma @ w_mkt) - rf
    base_var = float(w_mkt @ S @ w_mkt)
    delta = bl.get("risk_aversion") or max(base_excess / base_var, 1.0)   # implied risk aversion
    pi = delta * S @ w_mkt                                                # equilibrium excess returns

    views = bl.get("views", [])
    if views:
        P, Q, conf = [], [], []
        for v in views:
            row = np.zeros(len(names))
            for nm, c in v["assets"].items():
                if nm in names:
                    row[names.index(nm)] = c
            P.append(row); Q.append(v["q"]); conf.append(v.get("confidence", 0.5))
        P, Q = np.array(P), np.array(Q)
        tauS = tau * S
        # Omega from view confidence: higher confidence -> tighter (smaller) variance.
        omega = np.diag([(1.0 / max(c, 1e-3) - 1.0) * float(P[i] @ tauS @ P[i]) + 1e-10
                         for i, c in enumerate(conf)])
        A = np.linalg.inv(tauS) + P.T @ np.linalg.inv(omega) @ P
        b = np.linalg.inv(tauS) @ pi + P.T @ np.linalg.inv(omega) @ Q
        mu_excess = np.linalg.solve(A, b)
    else:
        mu_excess = pi

    return _max_sharpe_given_mu(market, config, mu_excess + rf, "Black-Litterman")


def robust_optimizer(market: MarketData, config: dict) -> Portfolio:
    """Robust optimisation by **resampling** (Michaud-style).

    Expected returns are uncertain. We draw many plausible mean vectors from the
    sampling distribution of the estimate, mu ~ N(mu_hat, Sigma / n_years), solve a
    constrained max-Sharpe for each, and **average the weights**. The averaged book
    is stable to estimation error - it does not bet the farm on one noisy forecast.
    """
    mu, cov = _inputs(market)
    m, S = mu.to_numpy(), cov.to_numpy()
    rc = config.get("robust", {})
    n_resample = rc.get("n_resample", 30)
    n_years = max(len(market.returns) / 12.0, 1.0)
    seed = config["meta"].get("random_seed", 7)
    rng = np.random.default_rng(seed)

    acc = np.zeros(len(m))
    for _ in range(int(n_resample)):
        mu_s = rng.multivariate_normal(m, S / n_years)
        acc += _max_sharpe_given_mu(market, config, mu_s, "robust-sample").weights.to_numpy()
    w = acc / acc.sum() if acc.sum() > 0 else market.baseline_weights.to_numpy()
    return Portfolio(pd.Series(w, index=market.meta.index), market, "Robust")


def ml_forecast(market: MarketData, config: dict) -> Portfolio:
    """ML-driven expected returns (transparent, illustrative).

    A pooled ridge regression predicts next-month return from four standard,
    leakage-safe signals - 12-1 momentum, 1-month reversal, carry (yield) and
    trailing volatility. The latest-month forecast is annualised and blended with
    the config capital-market assumptions (to stabilise a noisy single forecast),
    then fed to the constrained max-Sharpe. Stands in for richer ML forecasters.
    """
    meta = market.meta
    R = market.returns[meta.index]
    yld = (meta["yield"] / 12.0)
    mom = R.rolling(12).mean().shift(1)     # 12-1 momentum (shifted: no lookahead)
    rev = R.shift(1)                        # last month (reversal)
    vol = R.rolling(12).std().shift(1)

    X, y = [], []
    for t in range(13, len(R) - 1):
        for a in meta.index:
            row = [mom.iloc[t][a], rev.iloc[t][a], float(yld[a]), vol.iloc[t][a]]
            if not np.any(np.isnan(row)):
                X.append(row); y.append(R.iloc[t + 1][a])
    mlc = config.get("ml", {})
    mu_cma = meta["exp_return"].to_numpy()
    if len(X) < 50:                         # too little history - fall back to CMA
        return _max_sharpe_given_mu(market, config, mu_cma, "ML-Forecast")

    X, y = np.array(X), np.array(y)
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    Xz = (X - Xm) / Xs
    lam = mlc.get("ridge_lambda", 10.0)
    beta = np.linalg.solve(Xz.T @ Xz + lam * np.eye(Xz.shape[1]), Xz.T @ (y - y.mean()))

    preds = []
    for a in meta.index:
        row = np.array([mom.iloc[-1][a], rev.iloc[-1][a], float(yld[a]), vol.iloc[-1][a]])
        if np.any(np.isnan(row)):
            preds.append(meta.loc[a, "exp_return"])
        else:
            preds.append((y.mean() + ((row - Xm) / Xs) @ beta) * 12.0)   # annualise
    mu_ml = np.array(preds)
    blend = mlc.get("blend", 0.5)
    mu_final = blend * mu_ml + (1.0 - blend) * mu_cma
    return _max_sharpe_given_mu(market, config, mu_final, "ML-Forecast")
