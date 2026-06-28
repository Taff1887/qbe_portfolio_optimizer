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


# --------------------------------------------------------------- placeholders
def black_litterman(market: MarketData, config: dict) -> Portfolio:
    """Black-Litterman allocation - ROADMAP PLACEHOLDER.

    Intended design: recover equilibrium (reverse-optimised) expected returns from
    the baseline/market weights and the covariance, then Bayesian-blend them with
    explicit investor views (P, Q, Omega from config) before running the
    constrained MVO in `optimizer.py`. Not yet implemented.
    """
    raise NotImplementedError(
        "black_litterman is a roadmap placeholder - see docstring for the intended design."
    )


def robust_optimizer(market: MarketData, config: dict) -> Portfolio:
    """Robust (uncertainty-aware) optimisation - ROADMAP PLACEHOLDER.

    Intended design: optimise the worst case over an uncertainty set around the
    expected-return estimates (e.g. an ellipsoidal/box set, or resampled-frontier
    averaging) so the solution is stable to estimation error. Not yet implemented.
    """
    raise NotImplementedError(
        "robust_optimizer is a roadmap placeholder - see docstring for the intended design."
    )
