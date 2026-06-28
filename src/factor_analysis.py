"""Lens - factor analysis: the through-time drivers of return.

Point-in-time risk (volatility, VaR) says *how much* a book can move; it does not
say *why* it moves. This lens decomposes returns into the common macro/market
factors that drive them - rates, credit spread, structured-credit spread, equity,
property and gold - by regressing returns on those factors:

    r_t = alpha + sum_k beta_k * factor_k,t + e_t

It then reports each factor's beta, its share of explained variance, and the
annualised return *attributed* to each factor (beta_k x mean factor move), plus
**rolling** factor betas so you can see exposures drift through time (e.g. the
rate beta winding down). This is the "through time vs point in time" view: the
same average return can come from very different, time-varying factor bets.

On dummy data the factors are the generative factors saved to
`data/processed/factors.csv`. On real data, replace that file with observable
market factor series (e.g. a rates index change, an IG OAS change, an equity
index return) and the regression is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dummy_data import write_dummy_data
from portfolio import Portfolio
from utils import DATA_PROCESSED, MONTHS_PER_YEAR

# Human-readable labels for the generative factor columns.
FACTOR_LABELS = {
    "d_rate": "Rates",
    "d_spread": "Credit spread",
    "d_struct": "Structured spread",
    "equity": "Equity",
    "property": "Property",
    "gold": "Gold",
}


def load_factors(config: dict) -> pd.DataFrame | None:
    """Load the macro/market factor series.

    For synthetic data these are the generative factors. For real data, observable
    factor series are not yet wired in, so this returns None (the lens is skipped)
    rather than fabricating factors - on real data, supply a factors.csv of
    observable market series (rates index change, IG OAS, equity index, ...).
    """
    path = DATA_PROCESSED / "factors.csv"
    if not path.exists():
        if config.get("data", {}).get("source") == "synthetic":
            write_dummy_data(config)
        else:
            return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS with intercept. Returns (coeffs incl. intercept first, R-squared)."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


def factor_regression(portfolio: Portfolio, factors: pd.DataFrame) -> dict:
    """Regress portfolio returns on the factors; return betas, R^2 and attribution."""
    r = portfolio.return_series
    df = pd.concat([r.rename("ret"), factors], axis=1).dropna()
    cols = [c for c in factors.columns if c in df.columns]
    y = df["ret"].to_numpy()
    X = df[cols].to_numpy()
    coef, r2 = _ols(y, X)
    alpha_m, betas = coef[0], coef[1:]

    # Annualised return attributed to each factor = beta_k x mean monthly factor x 12.
    mean_f = df[cols].mean().to_numpy()
    contrib_ann = betas * mean_f * MONTHS_PER_YEAR
    attribution = pd.DataFrame({
        "factor": [FACTOR_LABELS.get(c, c) for c in cols],
        "beta": betas,
        "ann_contribution": contrib_ann,
    }).set_index("factor")
    attribution.loc["Alpha / carry"] = [np.nan, alpha_m * MONTHS_PER_YEAR]

    betas_s = pd.Series(betas, index=[FACTOR_LABELS.get(c, c) for c in cols])
    return {
        "betas": betas_s,
        "alpha_ann": float(alpha_m * MONTHS_PER_YEAR),
        "r_squared": r2,
        "attribution": attribution,
        "cols": cols,
    }


def rolling_betas(portfolio: Portfolio, factors: pd.DataFrame, cols: list[str],
                  window: int = 36) -> pd.DataFrame:
    """Rolling-window factor betas, to show exposures drifting through time."""
    r = portfolio.return_series
    df = pd.concat([r.rename("ret"), factors], axis=1).dropna()
    y_all = df["ret"].to_numpy()
    X_all = df[cols].to_numpy()
    idx = df.index
    rows, dates = [], []
    for t in range(window, len(df) + 1):
        coef, _ = _ols(y_all[t - window:t], X_all[t - window:t])
        rows.append(coef[1:])
        dates.append(idx[t - 1])
    out = pd.DataFrame(rows, index=pd.Index(dates, name="date"),
                       columns=[FACTOR_LABELS.get(c, c) for c in cols])
    return out


def run_factor_analysis(portfolio: Portfolio, config: dict) -> dict | None:
    factors = load_factors(config)
    if factors is None:
        return None
    reg = factor_regression(portfolio, factors)
    roll = rolling_betas(portfolio, factors, reg["cols"],
                         window=config["optimiser"].get("factor_window", 36))
    return {
        "betas": reg["betas"],
        "alpha_ann": reg["alpha_ann"],
        "r_squared": reg["r_squared"],
        "attribution": reg["attribution"],
        "rolling_betas": roll,
    }
