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
# Sub-investment-grade credit sleeves that feel a high-yield-specific widening.
_HY_SLEEVES = ("high_yield", "structured_mezz", "private_credit")


def asset_shock_impact(meta: pd.DataFrame, scenario: dict) -> pd.Series:
    """Per-asset mark-to-market return under one scenario.

    Every scenario key is read with a default of 0, so new scenarios can be added
    in config.yaml (e.g. a high-yield-only `hy_bps` shock) without code changes.
    """
    d_rate = scenario.get("rate_bps", 0) / 10_000.0
    d_spread = scenario.get("spread_bps", 0) / 10_000.0
    d_struct = scenario.get("structured_bps", 0) / 10_000.0
    d_hy = scenario.get("hy_bps", 0) / 10_000.0
    is_struct = meta["capital_category"].isin(_STRUCTURED).astype(float)
    is_hy = meta["capital_category"].isin(_HY_SLEEVES).astype(float)
    spread_total = d_spread + is_struct * d_struct + is_hy * d_hy

    return (
        -meta["duration"] * d_rate
        - meta["spread_duration"] * spread_total
        + meta["equity_beta"] * scenario.get("equity_pct", 0.0)
        + meta["property_beta"] * scenario.get("property_pct", 0.0)
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


def reverse_stress(portfolio: Portfolio, config: dict, net_duration_gap_years: float) -> pd.DataFrame:
    """Reverse stress test: instead of 'given a shock, what is the loss', solve
    'what single-factor move *breaches* a given loss limit'.

    For each risk factor the portfolio loss is linear in the shock, so the breach
    move = limit / sensitivity. Rate moves use the *net* (asset-minus-liability)
    duration for the economic-surplus limit and the gross asset duration for the
    P&L-earnings limit. Answers 'how big a move wipes out surplus / breaks plan?'.
    """
    meta, w = portfolio.meta, portfolio.weights
    surplus = 1.0 - config["portfolio"].get("liability_ratio", 0.82)
    earn_lim = config.get("reverse_stress", {}).get("earnings_limit", 0.02)
    rate_gross = portfolio.duration()
    rate_net = abs(net_duration_gap_years)
    spread = portfolio.spread_duration()
    eqty = float((w * meta["equity_beta"]).sum())
    prop = float((w * meta["property_beta"]).sum())

    def row(limit_loss, rdur):
        return {
            "limit_loss": limit_loss,
            "rates_bps": limit_loss / rdur * 1e4 if rdur > 1e-9 else float("inf"),
            "credit_spread_bps": limit_loss / spread * 1e4 if spread > 1e-9 else float("inf"),
            "equity_pct": -(limit_loss / eqty) if eqty > 1e-9 else float("-inf"),
            "property_pct": -(limit_loss / prop) if prop > 1e-9 else float("-inf"),
        }

    return pd.DataFrame({
        f"Erode all surplus ({surplus:.0%})": row(surplus, rate_net),
        f"P&L loss of {earn_lim:.0%}": row(earn_lim, rate_gross),
    }).T


def stress_matrix(portfolios: dict[str, Portfolio], config: dict) -> pd.DataFrame:
    """Total instantaneous impact of every scenario on every portfolio.

    Rows = scenarios, columns = portfolios. This is the "every optimiser under
    every stress" grid the comparison needs.
    """
    return pd.DataFrame({
        name: run_stress_tests(pf, config)["total_impact"]
        for name, pf in portfolios.items()
    })


def worst_stress_by_portfolio(portfolios: dict[str, Portfolio], config: dict) -> pd.Series:
    """Worst-case (most negative) total stress impact for each portfolio."""
    return stress_matrix(portfolios, config).min(axis=0).rename("worst_stress_loss")
