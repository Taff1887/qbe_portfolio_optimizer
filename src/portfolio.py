"""The Portfolio object - a set of weights plus every cross-cutting view of it.

Design choice: the *forward* expected return uses the config `exp_return`
assumptions (not the historical mean, which is inflated by the modelled secular
rate decline), while volatility / correlations come from the return history.
This mirrors real practice - forward capital-market assumptions for return,
empirical data for risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import MarketData
from metrics import (
    conditional_value_at_risk,
    sortino_ratio,
    tracking_error,
    value_at_risk,
)
from utils import (
    MONTHS_PER_YEAR,
    annualise_return,
    annualise_vol,
    max_drawdown,
    sharpe_ratio,
)


class Portfolio:
    """Holds weights over the asset universe and exposes finance views of them."""

    def __init__(self, weights: pd.Series, market: MarketData, name: str = "Portfolio"):
        self.market = market
        self.meta = market.meta
        self.name = name
        # Align to the universe, fill missing with 0, renormalise to sum 1.
        w = pd.Series(weights, dtype=float).reindex(self.meta.index).fillna(0.0)
        total = w.sum()
        self.weights = w / total if total != 0 else w

    # --------------------------------------------------------------- returns
    @property
    def return_series(self) -> pd.Series:
        """Monthly portfolio return history."""
        cols = self.weights.index
        return (self.market.returns[cols] * self.weights).sum(axis=1)

    def covariance(self, annualise: bool = True) -> pd.DataFrame:
        cov = self.market.returns[self.weights.index].cov()
        return cov * MONTHS_PER_YEAR if annualise else cov

    # -------------------------------------------------------- forward views
    def expected_return(self) -> float:
        """Forward expected return (config assumptions, weighted)."""
        return float((self.weights * self.meta["exp_return"]).sum())

    def carry(self) -> float:
        """Weighted running yield (annual carry income)."""
        return float((self.weights * self.meta["yield"]).sum())

    def volatility(self) -> float:
        """Annualised volatility from the historical covariance, sqrt(w' Sigma w).

        This equals the realised in-sample annualised vol by construction; we use
        the covariance form so it is consistent with the optimiser's risk model.
        (Forward CMAs are used only for expected *return*, not for risk.)
        """
        w = self.weights.to_numpy()
        cov = self.covariance().to_numpy()
        return float(np.sqrt(max(w @ cov @ w, 0.0)))

    def duration(self) -> float:
        return float((self.weights * self.meta["duration"]).sum())

    def spread_duration(self) -> float:
        return float((self.weights * self.meta["spread_duration"]).sum())

    def liquidity_score(self) -> float:
        return float((self.weights * self.meta["liquidity"]).sum())

    # ----------------------------------------------------------- exposures
    def exposure_by(self, field: str) -> pd.Series:
        """Total weight grouped by a metadata field (currency, group, ...)."""
        return self.weights.groupby(self.meta[field]).sum().sort_values(ascending=False)

    def core_fi_share(self) -> float:
        return float(self.weights[self.meta["group"] == "core_fi"].sum())

    def risk_asset_share(self) -> float:
        return float(self.weights[self.meta["group"] == "risk"].sum())

    # --------------------------------------------------- realised metrics
    def sortino(self) -> float:
        rf = self.market.config["portfolio"].get("risk_free", 0.0)
        return sortino_ratio(self.return_series, rf_annual=rf)

    def value_at_risk(self, level: float = 0.95) -> float:
        """Historical monthly VaR (signed return; negative = loss)."""
        return value_at_risk(self.return_series, level)

    def conditional_value_at_risk(self, level: float = 0.95) -> float:
        """Historical monthly CVaR / expected shortfall (signed return)."""
        return conditional_value_at_risk(self.return_series, level)

    def tracking_error(self, benchmark: "Portfolio | pd.Series") -> float:
        """Annualised tracking error against a benchmark portfolio or return series."""
        bench = benchmark.return_series if isinstance(benchmark, Portfolio) else benchmark
        return tracking_error(self.return_series, bench)

    def turnover_from(self, benchmark: "Portfolio | pd.Series") -> float:
        """One-way turnover (sum of absolute weight changes / 2) to move FROM a
        benchmark portfolio's weights to this portfolio's - i.e. the trade needed
        to implement this construction starting from the benchmark book."""
        bench_w = benchmark.weights if isinstance(benchmark, Portfolio) else pd.Series(benchmark)
        diff = (self.weights - bench_w.reindex(self.weights.index).fillna(0.0)).abs()
        return float(diff.sum() / 2.0)

    def realised_metrics(self) -> dict[str, float]:
        r = self.return_series
        rf = self.market.config["portfolio"].get("risk_free", 0.0)
        return {
            "ann_return": annualise_return(r),
            "ann_vol": annualise_vol(r),
            "sharpe": sharpe_ratio(r, rf_annual=rf),
            "sortino": self.sortino(),
            "max_drawdown": max_drawdown(r),
            "var_95": self.value_at_risk(0.95),
            "cvar_95": self.conditional_value_at_risk(0.95),
        }

    def summary(self) -> dict[str, float]:
        """One-line bundle used by reporting and comparison tables."""
        rm = self.realised_metrics()
        return {
            "expected_return": self.expected_return(),
            "volatility": self.volatility(),
            "carry": self.carry(),
            "duration": self.duration(),
            "spread_duration": self.spread_duration(),
            "core_fi": self.core_fi_share(),
            "risk_assets": self.risk_asset_share(),
            "liquidity": self.liquidity_score(),
            **rm,
        }

    # ----------------------------------------------------- transformations
    def scale_risk_assets(self, target: float, name: str | None = None) -> "Portfolio":
        """Rescale risk assets to `target` share (core FI to 1-target), pro rata
        within each group so the *mix* inside core FI and risk is preserved.

        This is a scenario *transform* of an existing portfolio, not an optimised
        solution, so it does not re-impose the optimiser's currency / per-asset
        constraints - it simply scales the two sleeves.
        """
        is_risk = self.meta["group"] == "risk"
        w = self.weights.copy()
        risk, core = w[is_risk], w[~is_risk]
        if risk.sum() > 0:
            w[is_risk] = risk / risk.sum() * target
        if core.sum() > 0:
            w[~is_risk] = core / core.sum() * (1.0 - target)
        return Portfolio(w, self.market, name or f"{self.name} (risk={target:.0%})")


def baseline_portfolio(market: MarketData) -> Portfolio:
    """The insurer-style baseline: 85% core FI / 15% risk, currency mix as given."""
    return Portfolio(market.baseline_weights, market, name="Baseline")
