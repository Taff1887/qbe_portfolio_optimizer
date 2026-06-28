"""Optimiser feasibility: every construction satisfies the insurer constraints."""

import numpy as np
import pytest

import construction
import optimizer

TOL = 1e-4


def _check_feasible(pf, market, config):
    w = pf.weights
    assert np.isclose(w.sum(), 1.0, atol=TOL)
    assert (w >= -TOL).all()
    assert w.max() <= config["optimiser"]["default_max_weight"] + 1e-3
    core = w[market.meta["group"] == "core_fi"].sum()
    assert core >= config["portfolio"]["min_core_fixed_income"] - TOL
    # risk assets only in eligible currencies
    elig = config["optimiser"].get("risk_asset_currencies")
    if elig:
        bad = market.meta[(market.meta["group"] == "risk") & (~market.meta["currency"].isin(elig))].index
        assert float(w[bad].sum()) <= TOL


@pytest.mark.parametrize("name", ["min_variance", "max_sharpe", "max_return_on_capital"])
def test_optimizer_feasible(name, market, config):
    _check_feasible(getattr(optimizer, name)(market, config), market, config)


@pytest.mark.parametrize("name", ["risk_parity", "max_diversification", "black_litterman",
                                  "robust_optimizer", "ml_forecast"])
def test_construction_feasible(name, market, config):
    _check_feasible(getattr(construction, name)(market, config), market, config)


def test_capital_budget_respected(market, config):
    budget = 0.03
    pf = optimizer.max_return_capital_budget(market, config, budget)
    assert optimizer.smooth_capital(market, config, pf.weights.to_numpy()) <= budget + 1e-3


def test_earnings_vol_constraint_respected(market, config):
    ev_budget = 0.02
    pf = optimizer.max_return_constrained(market, config, earnings_vol_budget=ev_budget)
    assert optimizer.earnings_volatility(market, pf.weights.to_numpy()) <= ev_budget + 1e-3


def test_min_variance_is_lowest_vol(market, config):
    mv = optimizer.min_variance(market, config).volatility()
    ms = optimizer.max_sharpe(market, config).volatility()
    assert mv <= ms + 1e-6
