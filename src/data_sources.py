"""Real market-data ingestion - FMP or Yahoo Finance.

The framework is data-source agnostic: every analysis module reads
`data/processed/returns.csv` (monthly total returns, columns = the asset names in
`config.yaml`). This module *produces* that file from real market data, so moving
from synthetic to real data is a one-config change.

Two providers are supported, selected by `market_data.provider` in config:

- **fmp**    - Financial Modeling Prep. Needs an API key in the `FMP_API_KEY`
               environment variable (or `market_data.fmp_key_env`). Uses the
               dividend/split-adjusted close, so price returns approximate total
               returns.
- **yahoo**  - Yahoo Finance via `yfinance` (monthly adjusted close).

Each universe asset maps to a liquid, investable proxy (ETF or index) under
`market_data.tickers`. Where no clean long-history proxy exists (some structured-
credit and private-asset sleeves), the mapping is the best available listed proxy
and is flagged in `market_data.proxy_notes`; replace with a real index series when
a Bloomberg/ICE feed is available.

Network note: outbound access to market-data hosts must be permitted by the
environment. In a locked-down sandbox these hosts may be blocked; run locally or
in an environment whose network policy allows them.

    python src/data_sources.py --provider fmp        # refresh returns.csv from FMP
    python src/data_sources.py --provider yahoo
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from utils import DATA_PROCESSED, load_config


# --------------------------------------------------------------- transforms
def prices_to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end total returns from a (date x ticker) adjusted-price frame.

    Pure function (no I/O) so it is unit-testable offline: resample to month end,
    take the last observation, then percentage change.
    """
    monthly = prices.sort_index().resample("ME").last()
    return monthly.pct_change().dropna(how="all")


def align_common_window(returns: pd.DataFrame, how: str = "common") -> pd.DataFrame:
    """Align asset histories. `common` truncates to the window where every asset
    has data (cleanest for covariance); `pairwise` keeps all rows (NaNs remain,
    downstream covariance is pairwise-complete)."""
    returns = returns.dropna(axis=1, how="all")
    if how == "common":
        return returns.dropna(how="any")
    return returns


# ------------------------------------------------------------------ FMP
def _fmp_series(ticker: str, key: str, start: str) -> pd.Series:
    import requests
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {"from": start, "apikey": key, "serietype": "line"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hist = r.json().get("historical", [])
    if not hist:
        raise ValueError(f"FMP returned no history for {ticker}")
    df = pd.DataFrame(hist)
    col = "adjClose" if "adjClose" in df.columns else "close"
    s = pd.Series(df[col].to_numpy(), index=pd.to_datetime(df["date"]), name=ticker)
    return s.sort_index()


def fetch_fmp(tickers: dict[str, str], start: str, key: str) -> pd.DataFrame:
    prices = {}
    for asset, tk in tickers.items():
        if not tk:
            continue
        try:
            prices[asset] = _fmp_series(tk, key, start)
        except Exception as e:  # keep going; report missing at the end
            print(f"  ! {asset} ({tk}): {type(e).__name__} {str(e)[:80]}")
    if not prices:
        raise RuntimeError("FMP returned no usable series (check key / tickers / network).")
    return prices_to_monthly_returns(pd.DataFrame(prices))


# ------------------------------------------------------------------ Yahoo
def fetch_yahoo(tickers: dict[str, str], start: str) -> pd.DataFrame:
    import yfinance as yf
    tk_map = {tk: a for a, tk in tickers.items() if tk}
    data = yf.download(list(tk_map), start=start, interval="1mo",
                       auto_adjust=True, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.rename(columns=tk_map)
    return close.pct_change().dropna(how="all")


# ------------------------------------------------------------- orchestration
def fetch_returns(config: dict) -> tuple[pd.DataFrame, dict]:
    """Fetch real monthly returns per `market_data` config. Returns (returns, provenance)."""
    md = config.get("market_data", {})
    provider = md.get("provider", "fmp").lower()
    start = md.get("start", config["meta"]["start_date"])
    tickers = md.get("tickers", {})
    align = md.get("align", "common")

    if provider == "fmp":
        key = os.environ.get(md.get("fmp_key_env", "FMP_API_KEY"))
        if not key:
            raise RuntimeError("FMP_API_KEY not set in the environment.")
        raw = fetch_fmp(tickers, start, key)
    elif provider == "yahoo":
        raw = fetch_yahoo(tickers, start)
    else:
        raise ValueError(f"Unknown market_data.provider: {provider!r}")

    # Keep only universe assets we have, in config order.
    universe = [a["name"] for a in config["assets"]]
    cols = [c for c in universe if c in raw.columns]
    returns = align_common_window(raw[cols], how=align)
    returns.index.name = "date"
    provenance = {
        "source": f"real:{provider}",
        "n_assets": returns.shape[1],
        "n_months": returns.shape[0],
        "start": str(returns.index.min().date()) if len(returns) else None,
        "end": str(returns.index.max().date()) if len(returns) else None,
        "missing_assets": [a for a in universe if a not in returns.columns],
    }
    return returns, provenance


def write_real_data(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    returns, prov = fetch_returns(config)
    returns.to_csv(DATA_PROCESSED / "returns.csv")
    print(f"Wrote {prov['n_months']} months x {prov['n_assets']} assets "
          f"({prov['start']} -> {prov['end']}) from {prov['source']}")
    if prov["missing_assets"]:
        print("  missing (no proxy / no data):", ", ".join(prov["missing_assets"]))
    return returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Refresh returns.csv from real market data.")
    ap.add_argument("--provider", choices=["fmp", "yahoo"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    if args.provider:
        cfg.setdefault("market_data", {})["provider"] = args.provider
    write_real_data(cfg)
