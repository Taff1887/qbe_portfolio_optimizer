"""Metric functions on known inputs."""

import numpy as np
import pandas as pd

import metrics
from utils import annualise_vol, sharpe_ratio


def test_var_cvar_ordering():
    r = pd.Series(np.linspace(-0.10, 0.10, 101))
    var = metrics.value_at_risk(r, 0.95)
    cvar = metrics.conditional_value_at_risk(r, 0.95)
    assert cvar <= var <= 0.0                      # CVaR is the deeper tail
    assert np.isclose(var, np.percentile(r, 5))


def test_tracking_error_zero_against_self():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 120))
    assert metrics.tracking_error(r, r) == 0.0


def test_tracking_error_matches_scaled_std():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 0.02, 240))
    b = pd.Series(rng.normal(0, 0.02, 240))
    te = metrics.tracking_error(a, b)
    assert np.isclose(te, (a - b).std(ddof=1) * np.sqrt(12))


def test_sortino_only_penalises_downside():
    # All-positive returns -> no downside -> Sortino is nan (no downside deviation).
    up = pd.Series([0.01] * 24)
    assert np.isnan(metrics.sortino_ratio(up))
    # Mixed series: Sortino should exceed Sharpe (downside vol < total vol).
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.005, 0.02, 240))
    assert metrics.sortino_ratio(r) > sharpe_ratio(r)


def test_downside_deviation_nonneg():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.03, 120))
    assert metrics.downside_deviation(r) >= 0.0
