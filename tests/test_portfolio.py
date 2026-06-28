"""Portfolio invariants and summary."""

import numpy as np
import pandas as pd

from portfolio import Portfolio


def test_weights_normalised(base):
    assert np.isclose(base.weights.sum(), 1.0)
    assert (base.weights >= -1e-12).all()


def test_unnormalised_weights_get_normalised(market):
    raw = pd.Series(2.0, index=market.meta.index)      # all 2.0 -> should renormalise
    pf = Portfolio(raw, market)
    assert np.isclose(pf.weights.sum(), 1.0)


def test_summary_has_all_keys(base):
    s = base.summary()
    for k in ["expected_return", "volatility", "sharpe", "sortino",
              "var_95", "cvar_95", "duration", "liquidity"]:
        assert k in s and np.isfinite(s[k])


def test_core_plus_risk_is_one(base):
    assert np.isclose(base.core_fi_share() + base.risk_asset_share(), 1.0)


def test_turnover_from_self_is_zero(base):
    assert np.isclose(base.turnover_from(base), 0.0)


def test_volatility_matches_quadratic_form(base):
    w = base.weights.to_numpy()
    cov = base.covariance().to_numpy()
    assert np.isclose(base.volatility(), np.sqrt(w @ cov @ w))
