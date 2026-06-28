"""Load configuration, asset metadata and the monthly return history.

`metadata` is built straight from config.yaml (the single source of truth for
static assumptions). `returns` is read from data/processed/returns.csv, generated
on the fly by the dummy-data module if the file does not yet exist. To use real
data, drop a real returns.csv in place and the rest is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from dummy_data import write_dummy_data
from utils import DATA_PROCESSED, load_config

META_COLUMNS = [
    "group", "currency", "rating", "accounting", "capital_category", "liquidity",
    "yield", "duration", "spread_duration", "equity_beta", "property_beta",
    "exp_return", "ann_vol", "baseline_weight",
]

_PROV_PATH = DATA_PROCESSED / "provenance.json"


def load_metadata(config: dict) -> pd.DataFrame:
    """Asset static data as a DataFrame indexed by asset name."""
    meta = pd.DataFrame(config["assets"]).set_index("name")
    return meta[META_COLUMNS]


def _write_provenance(prov: dict) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    _PROV_PATH.write_text(json.dumps(prov, indent=2), encoding="utf-8")


def _read_provenance() -> dict:
    if _PROV_PATH.exists():
        try:
            return json.loads(_PROV_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_returns(config: dict, regenerate: bool = False) -> tuple[pd.DataFrame, dict]:
    """Monthly returns plus a provenance dict.

    `data.source: real` fetches from the configured provider (FMP/Yahoo). If that
    fails and `allow_synthetic_fallback` is set, it falls back to the synthetic
    factor model and labels the provenance accordingly so the report can say so.
    """
    path = DATA_PROCESSED / "returns.csv"
    dcfg = config.get("data", {})
    source = dcfg.get("source", "synthetic")
    prov = _read_provenance()

    needs_build = regenerate or not path.exists()
    if source == "real":
        if needs_build or prov.get("source", "").startswith("synthetic"):
            try:
                import data_sources
                data_sources.write_real_data(config)
                prov = {"source": "real", "provider": config.get("market_data", {}).get("provider")}
            except Exception as e:
                if dcfg.get("allow_synthetic_fallback", True):
                    print(f"[data] real fetch unavailable ({type(e).__name__}: {str(e)[:120]}); "
                          "falling back to SYNTHETIC factor data.")
                    write_dummy_data(config)
                    prov = {"source": "synthetic-fallback", "reason": f"{type(e).__name__}: {str(e)[:160]}"}
                else:
                    raise
            _write_provenance(prov)
    else:
        if needs_build:
            write_dummy_data(config)
        prov = {"source": "synthetic"}
        _write_provenance(prov)

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df, prov


@dataclass
class MarketData:
    """Bundle of everything the analysis modules need."""

    config: dict
    meta: pd.DataFrame                 # static asset data (indexed by name)
    returns: pd.DataFrame             # monthly returns (date x asset)
    baseline_weights: pd.Series       # baseline portfolio weights (sum to 1)
    provenance: dict = field(default_factory=dict)   # where the returns came from

    @property
    def assets(self) -> list[str]:
        return list(self.meta.index)


def load_market_data(regenerate: bool = False, config: dict | None = None) -> MarketData:
    """Top-level loader used by main.py and every lens."""
    config = config or load_config()
    meta = load_metadata(config)
    returns, provenance = load_returns(config, regenerate=regenerate)
    # Keep metadata to the assets we actually have returns for (real data may be
    # missing some sleeves), preserving config order.
    have = [c for c in meta.index if c in returns.columns]
    meta = meta.loc[have]
    returns = returns[have]
    baseline = meta["baseline_weight"].astype(float)
    baseline = baseline / baseline.sum()       # normalise defensively
    return MarketData(config=config, meta=meta, returns=returns,
                      baseline_weights=baseline, provenance=provenance)
