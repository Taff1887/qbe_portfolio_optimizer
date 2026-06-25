"""Shared helpers: paths, config loading, plotting style and small metric funcs.

Everything that touches the filesystem or matplotlib lives here so the analysis
modules stay focused on finance logic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Repo root = one level up from this file's folder (src/).
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES = ROOT / "outputs" / "tables"
CONFIG_PATH = ROOT / "config.yaml"

MONTHS_PER_YEAR = 12

# Colour-blind-friendly palette reused across charts.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#F0E442", "#999999", "#8C564B", "#117733"]


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load the YAML configuration file into a dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    """Create every output/data directory the pipeline writes to."""
    for d in (DATA_RAW, DATA_PROCESSED, OUT_CHARTS, OUT_TABLES):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ plotting
def set_plot_style() -> None:
    """Apply a clean, consistent house style."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_chart(fig: plt.Figure, name: str) -> Path:
    """Save a figure to outputs/charts and close it."""
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    path = OUT_CHARTS / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    """Save a DataFrame to outputs/tables as CSV."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    path = OUT_TABLES / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=index)
    return path


# ------------------------------------------------------------------- metrics
def annualise_return(monthly: pd.Series) -> float:
    """Geometric annualised return from a monthly return series."""
    m = monthly.dropna()
    if m.empty:
        return float("nan")
    return float((1.0 + m).prod() ** (MONTHS_PER_YEAR / len(m)) - 1.0)


def annualise_vol(monthly: pd.Series) -> float:
    """Annualised volatility from a monthly return series."""
    return float(monthly.dropna().std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def drawdown_series(monthly: pd.Series) -> pd.Series:
    """Drawdown path (0 at a new high-water mark)."""
    wealth = (1.0 + monthly.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(monthly: pd.Series) -> float:
    """Worst peak-to-trough drawdown."""
    dd = drawdown_series(monthly)
    return float(dd.min()) if not dd.empty else float("nan")


def sharpe_ratio(monthly: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio (excess of a flat annual risk-free)."""
    m = monthly.dropna()
    if len(m) < 2 or m.std(ddof=1) == 0:
        return float("nan")
    excess = m - rf_annual / MONTHS_PER_YEAR
    return float(excess.mean() / m.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
