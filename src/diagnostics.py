"""Extra lens - portfolio diagnostics an insurer cares about beyond return/risk.

- liquidity profile: can the book raise cash in a stress without fire sales?
- concentration: how concentrated is the book (Herfindahl) by asset and currency?
- credit quality: the rating distribution and sub-investment-grade share.
- surplus / solvency: surplus = assets - liabilities; surplus-at-risk under the
  worst stress; coverage ratio.
- historical stress: the portfolio's ACTUAL behaviour in the GFC / COVID episodes
  embedded in the data, as a data-driven complement to deterministic shocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import Portfolio
from utils import drawdown_series

_SUB_IG = ("BB", "B", "CCC")


def liquidity_profile(portfolio: Portfolio) -> dict:
    """Weight by liquidity tier plus headline liquidity stats."""
    liq = portfolio.meta["liquidity"]
    tier = pd.cut(liq, bins=[0, 3, 6, 10], labels=["Illiquid (1-3)", "Semi (4-6)", "Liquid (7-10)"])
    by_tier = portfolio.weights.groupby(tier, observed=False).sum()
    return {
        "by_tier": by_tier,
        "weighted_score": float((portfolio.weights * liq).sum()),
        "pct_illiquid": float(portfolio.weights[liq <= 3].sum()),
        "pct_liquid": float(portfolio.weights[liq >= 7].sum()),
    }


def concentration(portfolio: Portfolio) -> dict:
    """Herfindahl indices and the largest holdings."""
    w = portfolio.weights
    ccy = portfolio.exposure_by("currency")
    return {
        "hhi_asset": float((w ** 2).sum()),
        "hhi_currency": float((ccy ** 2).sum()),
        "effective_n_assets": float(1.0 / (w ** 2).sum()),
        "top5_assets": w.nlargest(5),
    }


def rating_distribution(portfolio: Portfolio) -> dict:
    """Weight by rating bucket and the sub-investment-grade share."""
    by_rating = portfolio.weights.groupby(portfolio.meta["rating"]).sum().sort_values(ascending=False)
    sub_ig = float(portfolio.weights[portfolio.meta["rating"].isin(_SUB_IG)].sum())
    return {"by_rating": by_rating, "sub_investment_grade": sub_ig}


def surplus_analysis(portfolio: Portfolio, config: dict, worst_stress_total: float) -> dict:
    """Surplus and coverage, and surplus-at-risk under the worst stress.

    Assets are normalised to 1. Liabilities = `liability_ratio`. Surplus is the
    difference (the economic capital buffer). A worst-case asset move of
    `worst_stress_total` erodes surplus directly.
    """
    liab = config["portfolio"].get("liability_ratio", 0.82)
    assets = 1.0
    surplus = assets - liab
    assets_stressed = assets * (1.0 + worst_stress_total)
    surplus_stressed = assets_stressed - liab
    return {
        "assets": assets,
        "liabilities": liab,
        "surplus": surplus,
        "coverage_ratio": assets / liab,
        "worst_stress_total": worst_stress_total,
        "surplus_at_risk": surplus_stressed - surplus,            # change in surplus (negative)
        "surplus_erosion_pct": (surplus_stressed - surplus) / surplus,
        "coverage_ratio_stressed": assets_stressed / liab,
    }


def historical_stress(portfolio: Portfolio) -> pd.DataFrame:
    """The portfolio's realised return through embedded historical episodes."""
    r = portfolio.return_series
    dd = drawdown_series(r)
    episodes = {
        "Worst single month": (r.idxmin(), float(r.min())),
        "GFC (Sep08-Feb09)": (None, float((1 + r.loc["2008-09":"2009-02"]).prod() - 1)),
        "COVID (Feb-Mar20)": (None, float((1 + r.loc["2020-02":"2020-03"]).prod() - 1)),
        "Rate sell-off 2022": (None, float((1 + r.loc["2022-01":"2022-12"]).prod() - 1)),
        "Max drawdown": (dd.idxmin(), float(dd.min())),
    }
    return pd.DataFrame(
        [{"episode": k, "date": (str(v[0].date()) if v[0] is not None else "-"), "return": v[1]}
         for k, v in episodes.items()]
    ).set_index("episode")


def run_diagnostics(portfolio: Portfolio, config: dict, worst_stress_total: float) -> dict:
    return {
        "liquidity": liquidity_profile(portfolio),
        "concentration": concentration(portfolio),
        "rating": rating_distribution(portfolio),
        "surplus": surplus_analysis(portfolio, config, worst_stress_total),
        "historical_stress": historical_stress(portfolio),
    }
