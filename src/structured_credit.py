"""Lens - granular structured-credit deep-dive (strategic growth area).

The within-class lens flagged structured credit as the largest, out-of-sample-
robust same-risk opportunity. This module zooms in: it decomposes structured
credit into granular tranches (CLOs AAA->BB, US/EU; ABS auto/card; RMBS prime/
non-QM; CMBS conduit/SASB) and answers two questions an insurer actually faces:

1. **Same-risk mix** - holding volatility at the benchmark level, what tranche mix
   earns the most? (pure selection alpha within the sleeve)
2. **Capital-efficient mix** - structured credit splits across two LAGIC buckets
   (senior ~3%, mezz ~11%); what mix earns the most return per unit of *capital*?

Sub-tranche returns are built from the book's structured-credit holdings as a
shared factor plus tranche-specific dispersion (highly but not perfectly
correlated), so risk is empirical and return uses the forward assumptions. Each
tranche has a real ETF/index ticker hook (`structured_credit.tranches[*].ticker`)
for when a Bloomberg/ICE feed is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_loader import MarketData
from utils import MONTHS_PER_YEAR


def _parent_series(market: MarketData, spec: dict) -> pd.Series:
    """Shared structured-credit factor = baseline-weighted blend of the book's
    securitised holdings (falls back to an equal blend / first available)."""
    pa = [a for a in spec.get("parent_assets", []) if a in market.returns.columns]
    if not pa:
        return market.returns.iloc[:, 0]
    w = market.baseline_weights[pa]
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(pa), index=pa)
    return (market.returns[pa] * w).sum(axis=1)


def _sub_returns(spec: dict, parent: pd.Series, seed: int) -> pd.DataFrame:
    rho = spec.get("intra_correlation", 0.72)
    rng = np.random.default_rng(seed)
    z = ((parent - parent.mean()) / (parent.std(ddof=0) + 1e-12)).to_numpy()
    out = {}
    for t in spec["tranches"]:
        tv = t["ann_vol"] / np.sqrt(MONTHS_PER_YEAR)
        idio = rng.normal(0.0, 1.0, len(z))
        out[t["name"]] = t["exp_return"] / MONTHS_PER_YEAR + tv * (np.sqrt(rho) * z + np.sqrt(1 - rho) * idio)
    return pd.DataFrame(out, index=parent.index)


def _solve(objective, n: int, max_w: float, extra=None) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if extra:
        cons += extra
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_w)] * n, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def run_structured_credit(market: MarketData, config: dict) -> dict:
    spec = config.get("structured_credit")
    if not spec:
        return None
    tranches = spec["tranches"]
    names = [t["name"] for t in tranches]
    mu = np.array([t["exp_return"] for t in tranches])
    sf = config["lagic"]["stress_factors"]
    cap = np.array([sf.get(t["capital_category"], 0.0) for t in tranches])   # capital per tranche
    max_w = config["optimiser"].get("intra_max_weight", 0.40)
    rf = config["portfolio"].get("risk_free", 0.0)

    sub = _sub_returns(spec, _parent_series(market, spec), config["meta"].get("random_seed", 7) + 99)
    cov = (sub.cov() * MONTHS_PER_YEAR).to_numpy()

    bench = np.array([t["benchmark_weight"] for t in tranches], float)
    bench = bench / bench.sum()

    def metrics(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        return {"exp_return": ret, "volatility": vol, "capital": float(w @ cap),
                "sharpe": (ret - rf) / vol if vol > 0 else np.nan,
                "return_on_capital": ret / float(w @ cap) if w @ cap > 0 else np.nan}

    mb = metrics(bench)
    # Same-risk: max return at benchmark vol. Capital-efficient: max return / capital.
    same_risk = _solve(lambda w: -(w @ mu), len(mu), max_w,
                       [{"type": "ineq", "fun": lambda w: mb["volatility"] ** 2 - w @ cov @ w}])
    cap_eff = _solve(lambda w: -(w @ mu) / max(w @ cap, 1e-9), len(mu), max_w)
    max_sharpe = _solve(lambda w: -((w @ mu) - rf) / np.sqrt(max(w @ cov @ w, 1e-12)), len(mu), max_w)

    weights = pd.DataFrame({"benchmark": bench, "same_risk": same_risk,
                            "capital_efficient": cap_eff, "max_sharpe": max_sharpe}, index=names)

    # Return vs volatility frontier across the tranche set.
    r_lo, r_hi = float(mu.min()), float(mu.max())
    frontier = []
    for target in np.linspace(mb["exp_return"], r_hi, 12):
        w = _solve(lambda w: w @ cov @ w, len(mu), max_w,
                   [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
        frontier.append({"exp_return": float(w @ mu), "volatility": float(np.sqrt(max(w @ cov @ w, 0.0))),
                         "capital": float(w @ cap)})

    tranche_tbl = pd.DataFrame({
        "exp_return": mu, "ann_vol": [t["ann_vol"] for t in tranches],
        "capital": cap, "benchmark_weight": bench,
        "rating": [t["rating"] for t in tranches], "region": [t["region"] for t in tranches],
    }, index=names)

    return {
        "tranches": tranche_tbl,
        "weights": weights,
        "benchmark": mb,
        "same_risk": metrics(same_risk),
        "capital_efficient": metrics(cap_eff),
        "max_sharpe": metrics(max_sharpe),
        "frontier": pd.DataFrame(frontier),
        "same_risk_uplift_bps": (metrics(same_risk)["exp_return"] - mb["exp_return"]) * 1e4,
        "capital_roc_gain": metrics(cap_eff)["return_on_capital"] - mb["return_on_capital"],
    }
