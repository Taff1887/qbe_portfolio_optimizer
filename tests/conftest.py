"""Shared pytest fixtures. Builds the market once on synthetic data (deterministic,
no network) and exposes config / market / baseline to every test."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_market_data  # noqa: E402
from portfolio import baseline_portfolio  # noqa: E402
from utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    cfg = load_config()
    cfg.setdefault("data", {})["source"] = "synthetic"   # deterministic, offline
    cfg.setdefault("robust", {})["n_resample"] = 6        # keep robust test fast
    return cfg


@pytest.fixture(scope="session")
def market(config):
    return load_market_data(regenerate=True, config=config)


@pytest.fixture(scope="session")
def base(market):
    return baseline_portfolio(market)
