"""Lens 3/6 - through-time earnings, carry and accounting-impact model.

Point-in-time volatility is not enough for an insurer: the business is held to an
annual *earnings* plan, and how returns split between carry (predictable) and
mark-to-market (volatile), and between P&L (hits earnings) and OCI (hits reserves),
matters as much as the headline return.

Each month we decompose the return:
    total = carry + mark_to_market
    P&L impact = carry + MtM(P&L-bucket assets)        # flows through earnings
    OCI impact =        MtM(OCI-bucket assets)         # flows through reserves
(by construction P&L + OCI = total). Note (IFRS 9): all running carry - on BOTH
buckets - is recognised in P&L; only the *price* mark-to-market of FVOCI assets
sits in OCI, which is exactly what this split does.

We then aggregate to annual earnings, measure earnings volatility and the chance
of missing the annual plan, and include a worked duration example showing that
*more duration can hurt today's mark-to-market but lift and stabilise full-year
carry / earnings* - and that under ALM the asset move is offset by liabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import Portfolio
from utils import MONTHS_PER_YEAR


def decompose_earnings(portfolio: Portfolio) -> pd.DataFrame:
    """Monthly decomposition into total / carry / MtM / P&L / OCI."""
    meta = portfolio.meta
    w = portfolio.weights
    rets = portfolio.market.returns[w.index]
    carry_i = meta["yield"] / MONTHS_PER_YEAR          # constant monthly accrual
    mtm_i = rets - carry_i                             # price component per asset
    is_pnl = (meta["accounting"] == "P&L").reindex(w.index)

    total = (rets * w).sum(axis=1)
    carry = float((w * carry_i).sum())                 # same every month
    pnl = carry + (mtm_i.loc[:, is_pnl] * w[is_pnl]).sum(axis=1)
    oci = (mtm_i.loc[:, ~is_pnl] * w[~is_pnl]).sum(axis=1)

    return pd.DataFrame({
        "total": total,
        "carry": carry,
        "mtm": total - carry,
        "pnl": pnl,
        "oci": oci,
    })


def annual_earnings(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to calendar-year earnings (total compounds; flows sum)."""
    g = monthly.groupby(monthly.index.year)
    return pd.DataFrame({
        "total": g["total"].apply(lambda s: (1 + s).prod() - 1),
        "carry": g["carry"].sum(),
        "mtm": g["mtm"].sum(),
        "pnl": g["pnl"].sum(),
        "oci": g["oci"].sum(),
    })


def earnings_summary(portfolio: Portfolio, config: dict) -> dict[str, float]:
    monthly = decompose_earnings(portfolio)
    annual = annual_earnings(monthly)
    plan = config["portfolio"]["plan_return_target"]
    return {
        "mean_annual_total": float(annual["total"].mean()),
        "mean_annual_pnl": float(annual["pnl"].mean()),
        "earnings_volatility": float(annual["pnl"].std(ddof=1)),   # std of annual P&L
        "total_return_volatility": float(annual["total"].std(ddof=1)),
        "plan_target": plan,
        # plan is an *earnings* plan, so measure misses on the P&L line (consistent
        # with earnings_volatility); the total-return basis is also reported.
        "plan_miss_prob": float((annual["pnl"] < plan).mean()),
        "plan_miss_prob_total": float((annual["total"] < plan).mean()),
        "worst_year_pnl": float(annual["pnl"].min()),
        "carry_share_of_return": float(annual["carry"].sum() / (annual["carry"].sum() + annual["mtm"].sum())),
    }


def run_earnings(portfolio: Portfolio, config: dict) -> dict:
    monthly = decompose_earnings(portfolio)
    return {
        "monthly": monthly,
        "annual": annual_earnings(monthly),
        "summary": earnings_summary(portfolio, config),
    }


def duration_earnings_example(config: dict | None = None) -> dict:
    """Worked example: why matching duration to (long) liabilities gives higher,
    steadier earnings, even though the longer asset book has a bigger *standalone*
    mark-to-market swing.

    Setup: a long-tail insurer with 7y liabilities. Compare a short, *unmatched*
    asset book (3y, 4.0% carry) with a long, *matched* book (7y, 4.8% carry),
    under both a +100bp and a -100bp rate move. Year-1 accounting earnings =
    asset MtM + offsetting liability remeasurement + carry.

    The unmatched book's earnings swing wildly with the rate direction (it is
    really an unhedged rate bet); the matched book earns its (higher) carry in
    either scenario - higher AND far more stable earnings.
    """
    liab_dur = 7.0
    books = [
        {"name": "Short book (3y, unmatched)", "dur": 3.0, "carry": 0.040},
        {"name": "Long matched book (7y)", "dur": 7.0, "carry": 0.048},
    ]
    rate_scenarios = {"Rates +100bp": 0.01, "Rates -100bp": -0.01}

    rows = []
    for bk in books:
        for sname, d_rate in rate_scenarios.items():
            asset_mtm = -bk["dur"] * d_rate
            liab_offset = +liab_dur * d_rate       # surplus gains when rates rise
            economic = asset_mtm + liab_offset      # MtM net of the liability move
            earnings = economic + bk["carry"]       # year-1 P&L = economic + carry
            rows.append({"book": bk["name"], "scenario": sname,
                         "asset_mtm": asset_mtm, "economic_mtm": economic, "earnings": earnings})
    table = pd.DataFrame(rows)
    # pivot() sorts the index alphabetically, so select rows by NAME (not iloc).
    pivot = table.pivot(index="book", columns="scenario", values="earnings").reindex([b["name"] for b in books])
    short_name, long_name = books[0]["name"], books[1]["name"]

    rng = (pivot.max(axis=1) - pivot.min(axis=1))   # earnings range = rate-driven instability
    summary = {
        "liability_duration": liab_dur,
        "short_earnings_range": float(rng.loc[short_name]),
        "long_earnings_range": float(rng.loc[long_name]),
        "short_mean_earnings": float(pivot.loc[short_name].mean()),
        "long_mean_earnings": float(pivot.loc[long_name].mean()),
    }
    return {"table": table, "pivot": pivot, "summary": summary, "liability_duration": liab_dur}
