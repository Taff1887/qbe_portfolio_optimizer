"""Load configuration, asset metadata and the monthly return history.

`metadata` is built straight from config.yaml (the single source of truth for
static assumptions). `returns` is read from data/processed/returns.csv, generated
on the fly by the dummy-data module if the file does not yet exist. To use real
data, drop a real returns.csv in place and the rest is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dummy_data import write_dummy_data
from utils import DATA_PROCESSED, load_config

META_COLUMNS = [
    "group", "currency", "rating", "accounting", "capital_category", "liquidity",
    "yield", "duration", "spread_duration", "equity_beta", "property_beta",
    "exp_return", "ann_vol", "baseline_weight",
]


def load_metadata(config: dict) -> pd.DataFrame:
    """Asset static data as a DataFrame indexed by asset name."""
    meta = pd.DataFrame(config["assets"]).set_index("name")
    return meta[META_COLUMNS]


def load_returns(config: dict, regenerate: bool = False) -> pd.DataFrame:
    """Monthly returns (index=date, columns=asset names)."""
    path = DATA_PROCESSED / "returns.csv"
    if regenerate or not path.exists():
        write_dummy_data(config)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


@dataclass
class MarketData:
    """Bundle of everything the analysis modules need."""

    config: dict
    meta: pd.DataFrame                 # static asset data (indexed by name)
    returns: pd.DataFrame             # monthly returns (date x asset)
    baseline_weights: pd.Series       # baseline portfolio weights (sum to 1)

    @property
    def assets(self) -> list[str]:
        return list(self.meta.index)


def load_market_data(regenerate: bool = False, config: dict | None = None) -> MarketData:
    """Top-level loader used by main.py and every lens."""
    config = config or load_config()
    meta = load_metadata(config)
    returns = load_returns(config, regenerate=regenerate)
    # Align return columns to the metadata order.
    returns = returns[[c for c in meta.index if c in returns.columns]]
    baseline = meta["baseline_weight"].astype(float)
    baseline = baseline / baseline.sum()       # normalise defensively
    return MarketData(config=config, meta=meta, returns=returns, baseline_weights=baseline)
