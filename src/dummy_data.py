"""Generate realistic monthly dummy returns for the asset universe.

We use a small, transparent FACTOR MODEL so the returns have sensible
cross-correlations and real-looking crisis behaviour (GFC, COVID, the 2022 rate
sell-off) rather than independent noise:

    asset_return = carry + rate_leg + spread_leg + equity_leg + property_leg + idiosyncratic

where each leg uses the asset's own duration / spread-duration / equity beta /
property beta from config.yaml. Idiosyncratic vol is calibrated so each asset's
total volatility matches its `ann_vol` assumption.

Replace the output (data/processed/returns.csv) with a real return history and
the rest of the framework works unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import DATA_PROCESSED, MONTHS_PER_YEAR, load_config


def _crisis_mask(dates: pd.DatetimeIndex, periods: list[tuple[str, str]]) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for s, e in periods:
        mask |= (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
    return mask


def _build_factors(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Construct the common macro/market factors (all in monthly decimal units)."""
    n = len(dates)
    yr = dates.year.to_numpy()

    # --- Government rate change (decimal, e.g. +0.0010 = +10bp) -------------
    # Secular decline 2000-2021, sharp sell-off in 2022, mild normalisation after.
    drift = np.where(yr <= 2021, -0.00035, np.where(yr == 2022, 0.0030, 0.0006))
    d_rate = drift + rng.normal(0, 0.0012, n)

    # --- Credit spread change and an extra structured-credit leg -----------
    d_spread = rng.normal(0, 0.0005, n)
    d_struct = rng.normal(0, 0.0007, n)

    # --- Equity and property market factors --------------------------------
    equity = rng.normal(0.0075, 0.038, n)          # ~9%/yr drift, ~13%/yr vol
    # Property is smoothed (appraisal-based): AR(1) around a steady income trend.
    prop = np.zeros(n)
    shock = rng.normal(0.0045, 0.012, n)
    for i in range(1, n):
        prop[i] = 0.7 * prop[i - 1] + 0.3 * shock[i]

    # --- Gold: a safe-haven commodity leg ----------------------------------
    # Mild trend, high stand-alone vol, and a positive bid in crises (the
    # diversifying flight-to-quality behaviour gold is held for).
    gold = rng.normal(0.0028, 0.040, n)            # ~3.4%/yr drift, ~14%/yr vol

    # --- Inject crisis episodes (flight to quality + spread blow-out) -------
    gfc = _crisis_mask(dates, [("2008-09-01", "2009-02-28")])
    covid = _crisis_mask(dates, [("2020-02-01", "2020-03-31")])
    rebound = _crisis_mask(dates, [("2009-03-01", "2009-06-30"), ("2020-04-01", "2020-06-30")])

    d_rate[gfc] -= 0.0035                 # rates fall in a crisis
    d_rate[covid] -= 0.0045
    d_spread[gfc] += 0.0055               # spreads blow out
    d_spread[covid] += 0.0090
    d_struct[gfc] += 0.0130               # structured credit hit hardest
    d_struct[covid] += 0.0150
    equity[gfc] -= 0.120
    equity[covid] -= 0.140
    equity[rebound] += 0.075              # sharp recoveries
    prop[gfc] -= 0.020
    prop[covid] -= 0.025
    gold[gfc] += 0.030                    # safe-haven bid when equities fall
    gold[covid] += 0.020

    return pd.DataFrame(
        {"d_rate": d_rate, "d_spread": d_spread, "d_struct": d_struct,
         "equity": equity, "property": prop, "gold": gold},
        index=dates,
    )


def generate_returns(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (monthly_returns, factors). Columns of returns are asset names."""
    config = config or load_config()
    seed = config["meta"].get("random_seed", 7)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(config["meta"]["start_date"], config["meta"]["end_date"], freq="ME")
    factors = _build_factors(dates, rng)

    out = {}
    for a in config["assets"]:
        # Structured-credit sleeves also feel the extra structured spread leg.
        is_struct = a["capital_category"] in ("structured_senior", "structured_mezz")
        spread_change = factors["d_spread"] + (factors["d_struct"] if is_struct else 0.0)
        is_gold = a["capital_category"] == "gold"

        systematic = (
            -a["duration"] * factors["d_rate"]
            - a["spread_duration"] * spread_change
            + a["equity_beta"] * factors["equity"]
            + a["property_beta"] * factors["property"]
            + (factors["gold"] if is_gold else 0.0)
        )
        carry = a["yield"] / MONTHS_PER_YEAR
        target_vol = a["ann_vol"] / np.sqrt(MONTHS_PER_YEAR)

        # Price return = systematic factor exposure + idiosyncratic noise, then
        # rescaled to EXACTLY the target monthly vol (so realised vol matches the
        # ann_vol assumption even where the systematic part alone would overshoot).
        price = systematic.to_numpy() + rng.normal(0.0, 0.5 * target_vol, len(dates))
        cur_vol = price.std()
        if cur_vol > 1e-12:
            price = price * (target_vol / cur_vol)

        out[a["name"]] = carry + price

    returns = pd.DataFrame(out, index=dates)
    returns.index.name = "date"
    return returns, factors


def write_dummy_data(config: dict | None = None) -> pd.DataFrame:
    """Generate and persist the dummy return history to data/processed."""
    config = config or load_config()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    returns, factors = generate_returns(config)
    returns.to_csv(DATA_PROCESSED / "returns.csv")
    factors.to_csv(DATA_PROCESSED / "factors.csv")
    return returns


if __name__ == "__main__":
    df = write_dummy_data()
    print(f"Wrote {df.shape[0]} months x {df.shape[1]} assets to data/processed/returns.csv")
    print((df.mean() * 12).round(3).rename("ann_return").to_string())
