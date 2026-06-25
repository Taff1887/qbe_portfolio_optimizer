"""Lens 4 - duration / ALM-style risk management.

Insurers hold assets to back insurance liabilities. The key risk is the DURATION
GAP between assets and liabilities by currency. We separate:

- the P&L (matched) book, which backs liabilities and should be near duration-
  matched, so asset and liability rate moves offset in earnings;
- the OCI (surplus) book, whose duration is an outright economic rate exposure
  that flows through reserves, not earnings.

Liabilities are assumed to be backed ~1:1 by the P&L asset book (their notional in
each currency = the P&L asset weight there), with durations from config. This lets
us show the crucial point: under a rate shock the *economic* surplus impact and
the *earnings* (P&L) impact differ by exactly the surplus (OCI) book's rate exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import Portfolio

DEFAULT_RATE_SHOCK = 0.01   # +100bp headline ALM shock


def duration_by_currency(portfolio: Portfolio, config: dict) -> pd.DataFrame:
    """Asset vs liability duration and the gap, per currency."""
    meta, w = portfolio.meta, portfolio.weights
    is_pnl = meta["accounting"] == "P&L"
    rows = []
    for ccy, b in config["currencies"].items():
        in_ccy = meta["currency"] == ccy
        cw = float(w[in_ccy].sum())
        pnl_in = in_ccy & is_pnl
        pnl_w = float(w[pnl_in].sum())
        a_dur = float((w[in_ccy] * meta.loc[in_ccy, "duration"]).sum() / cw) if cw > 0 else 0.0
        p_dur = float((w[pnl_in] * meta.loc[pnl_in, "duration"]).sum() / pnl_w) if pnl_w > 0 else 0.0
        l_dur = b["liability_duration"]
        rows.append({
            "currency": ccy,
            "currency_weight": cw,
            "pnl_weight": pnl_w,
            "asset_duration": a_dur,
            "pnl_asset_duration": p_dur,
            "liability_duration": l_dur,
            "duration_gap": a_dur - l_dur,
            "pnl_duration_gap": p_dur - l_dur,
        })
    return pd.DataFrame(rows).set_index("currency")


def rate_shock_impact(portfolio: Portfolio, config: dict, shock: float = DEFAULT_RATE_SHOCK) -> dict:
    """Decompose a parallel rate shock into earnings vs economic impact."""
    meta, w = portfolio.meta, portfolio.weights
    is_pnl = meta["accounting"] == "P&L"

    # Dollar-durations (weight x duration), already as a share of the portfolio.
    pnl_dollar_dur = float((w[is_pnl] * meta.loc[is_pnl, "duration"]).sum())
    oci_dollar_dur = float((w[~is_pnl] * meta.loc[~is_pnl, "duration"]).sum())
    asset_dollar_dur = pnl_dollar_dur + oci_dollar_dur

    # Liability dollar-duration: notional = P&L asset weight per currency.
    by_ccy = duration_by_currency(portfolio, config)
    liab_dollar_dur = float((by_ccy["pnl_weight"] * by_ccy["liability_duration"]).sum())

    asset_mtm = -asset_dollar_dur * shock
    pnl_asset_mtm = -pnl_dollar_dur * shock
    oci_asset_mtm = -oci_dollar_dur * shock
    liability_offset = +liab_dollar_dur * shock          # liabilities fall in value -> gain

    pnl_earnings_impact = pnl_asset_mtm + liability_offset   # matched book in P&L
    economic_surplus_impact = asset_mtm + liability_offset   # full surplus move
    return {
        "shock": shock,
        "asset_mtm": asset_mtm,
        "pnl_earnings_impact": pnl_earnings_impact,
        "oci_impact": oci_asset_mtm,
        "liability_offset": liability_offset,
        "economic_surplus_impact": economic_surplus_impact,
        "economic_minus_pnl": economic_surplus_impact - pnl_earnings_impact,  # = OCI exposure
        "pnl_dollar_duration": pnl_dollar_dur,
        "oci_dollar_duration": oci_dollar_dur,
        "liability_dollar_duration": liab_dollar_dur,
    }


def run_duration(portfolio: Portfolio, config: dict) -> dict:
    by_ccy = duration_by_currency(portfolio, config)
    rs = rate_shock_impact(portfolio, config)
    # Headline gap = total asset dollar-duration minus liability dollar-duration,
    # using the SAME P&L-backed liability notional as the rate-shock figures, so
    # economic_surplus_impact == -total_dollar_duration_gap * shock by construction.
    total_gap = portfolio.duration() - rs["liability_dollar_duration"]
    matched_gap = float((by_ccy["pnl_weight"] * by_ccy["pnl_duration_gap"]).sum())
    return {
        "by_currency": by_ccy,
        "portfolio_asset_duration": portfolio.duration(),
        "total_dollar_duration_gap": total_gap,
        "matched_book_dollar_gap": matched_gap,
        "rate_shock": rs,
    }
