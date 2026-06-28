"""Real-data transforms (pure, offline)."""

import numpy as np
import pandas as pd

import data_sources


def test_prices_to_monthly_returns():
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    px = pd.DataFrame({"A": 100 * (1 + np.linspace(0, 0.09, 90))}, index=idx)
    r = data_sources.prices_to_monthly_returns(px)
    assert len(r) == 2                               # 3 months of daily -> 2 returns
    assert (r["A"] > 0).all()


def test_align_common_vs_pairwise():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    r = pd.DataFrame({"A": [0.01, 0.02, 0.0, 0.01],
                      "B": [np.nan, 0.01, 0.0, 0.02]}, index=idx)
    assert len(data_sources.align_common_window(r, "common")) == 3
    assert len(data_sources.align_common_window(r, "pairwise")) == 4


def test_align_drops_all_nan_columns():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    r = pd.DataFrame({"A": [0.01, 0.0, 0.01], "Z": [np.nan, np.nan, np.nan]}, index=idx)
    out = data_sources.align_common_window(r, "pairwise")
    assert "Z" not in out.columns
