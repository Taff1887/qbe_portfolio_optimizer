"""Lens 2 - instantaneous deterministic stress testing.

Each scenario applies simultaneous shocks. The mark-to-market impact on each
asset is a first-order (duration/beta) approximation:

    impact = -rate_duration   * d_rate
             -spread_duration * (d_spread [+ d_structured if structured])
             +equity_beta     * equity_move
             +property_beta   * property_move

Portfolio impact is the weighted sum. We also split it into the P&L book
(FVTPL - hits earnings now) and the OCI book (FVOCI - hits reserves, not
earnings), which is the accounting distinction insurers care about.
"""

from __future__ import annotations

import pandas as pd

from portfolio import Portfolio

_STRUCTURED = ("structured_senior", "structured_mezz")


def asset_shock_impact(meta: pd.DataFrame, scenario: dict) -> pd.Series:
    """Per-asset mark-to-market return under one scenario."""
    d_rate = scenario["rate_bps"] / 10_000.0
    d_spread = scenario["spread_bps"] / 10_000.0
    d_struct = scenario["structured_bps"] / 10_000.0
    is_struct = meta["capital_category"].isin(_STRUCTURED).astype(float)
    spread_total = d_spread + is_struct * d_struct

    return (
        -meta["duration"] * d_rate
        - meta["spread_duration"] * spread_total
        + meta["equity_beta"] * scenario["equity_pct"]
        + meta["property_beta"] * scenario["property_pct"]
    )


def run_stress_tests(portfolio: Portfolio, config: dict) -> pd.DataFrame:
    """Total / P&L / OCI impact for every configured scenario."""
    meta = portfolio.meta
    is_pnl = meta["accounting"] == "P&L"
    rows = []
    for sc in config["stress_scenarios"]:
        contrib = portfolio.weights * asset_shock_impact(meta, sc)
        rows.append({
            "scenario": sc["name"],
            "total_impact": float(contrib.sum()),
            "pnl_impact": float(contrib[is_pnl].sum()),
            "oci_impact": float(contrib[~is_pnl].sum()),
        })
    return pd.DataFrame(rows).set_index("scenario")


def stress_attribution(portfolio: Portfolio, config: dict, scenario_name: str) -> pd.Series:
    """Per-asset contribution to one named scenario (sorted, worst first)."""
    sc = next(s for s in config["stress_scenarios"] if s["name"] == scenario_name)
    contrib = portfolio.weights * asset_shock_impact(portfolio.meta, sc)
    return contrib.sort_values()


def worst_scenario(stress_table: pd.DataFrame) -> tuple[str, float]:
    """Name and total impact of the worst (most negative) scenario."""
    s = stress_table["total_impact"]
    return s.idxmin(), float(s.min())
