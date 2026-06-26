"""Extra lens - diversification and selection WITHIN an asset class.

The top-level strategic allocation (85/15, currency mix) is largely fixed for an
insurer. But inside each asset class there is room to (a) diversify across sub-
sleeves to cut idiosyncratic risk and (b) tilt toward better risk-adjusted sub-
sleeves - small, repeatable "implementation" pickups that do not change the SAA.

For each decomposed class we compare three intra-class mixes:
- **Concentrated**: 100% in the single largest sub-sleeve (no diversification).
- **Benchmark**: the default `benchmark_weight` mix (diversified but untilted).
- **Optimised**: a constrained max-Sharpe mix across the sub-sleeves.

We then aggregate the per-class return pickup to a portfolio-level uplift, weighting
each class by its share of the book - holding the SAA unchanged.

Sub-sleeve returns are generated from the parent sleeve's return as a shared
factor plus sub-specific dispersion (so they are highly but not perfectly
correlated). Forward `exp_return` assumptions drive the optimisation; the
generated covariance drives risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_loader import MarketData
from utils import MONTHS_PER_YEAR


def generate_sub_returns(class_name: str, spec: dict, parent: pd.Series, seed: int) -> pd.DataFrame:
    """Monthly returns for the sub-sleeves of one class, tied to the parent."""
    rho = spec["intra_correlation"]
    rng = np.random.default_rng(seed)
    z = ((parent - parent.mean()) / parent.std(ddof=0)).to_numpy()   # standardised parent
    out = {}
    for s in spec["components"]:
        tv = s["ann_vol"] / np.sqrt(MONTHS_PER_YEAR)
        idio = rng.normal(0.0, 1.0, len(z))
        out[s["name"]] = s["exp_return"] / MONTHS_PER_YEAR + tv * (np.sqrt(rho) * z + np.sqrt(1 - rho) * idio)
    return pd.DataFrame(out, index=parent.index)


def _solve(objective, n: int, max_w: float, extra_cons=None) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if extra_cons:
        cons += extra_cons
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0.0, max_w)] * n,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def _max_return_at_vol(mu, cov, vol_cap, max_w) -> np.ndarray:
    """Maximise expected return subject to volatility <= vol_cap (same-risk uplift)."""
    cons = [{"type": "ineq", "fun": lambda w, vc=vol_cap: vc ** 2 - w @ cov @ w}]
    return _solve(lambda w: -(w @ mu), len(mu), max_w, cons)


def _max_sharpe(mu, cov, rf, max_w) -> np.ndarray:
    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-12))
        return -((w @ mu) - rf) / vol
    return _solve(neg_sharpe, len(mu), max_w)


def analyse_class(class_name: str, spec: dict, parent: pd.Series, config: dict, seed: int) -> dict:
    rf = config["portfolio"].get("risk_free", 0.0)
    max_w = config["optimiser"].get("intra_max_weight", 0.40)
    subs = spec["components"]
    names = [s["name"] for s in subs]

    sub_ret = generate_sub_returns(class_name, spec, parent, seed)
    mu = np.array([s["exp_return"] for s in subs])
    cov = (sub_ret.cov() * MONTHS_PER_YEAR).to_numpy()

    bench = np.array([s["benchmark_weight"] for s in subs], float)
    bench = bench / bench.sum()
    conc = np.zeros(len(subs)); conc[int(np.argmax(bench))] = 1.0          # single largest sub-sleeve

    def metrics(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        return {"exp_return": ret, "volatility": vol, "sharpe": (ret - rf) / vol if vol > 0 else np.nan}

    mb = metrics(bench)
    # Headline: most return achievable WITHOUT taking more risk than the benchmark mix.
    enhanced = _max_return_at_vol(mu, cov, mb["volatility"], max_w)
    best_sharpe = _max_sharpe(mu, cov, rf, max_w)
    mc, me, ms = metrics(conc), metrics(enhanced), metrics(best_sharpe)

    # Diversification benefit = weighted-average standalone vol minus the diversified
    # mix vol (always >= 0 because intra-class correlation < 1).
    wavg_vol = float(bench @ np.sqrt(np.diag(cov)))
    return {
        "names": names,
        "weights": pd.DataFrame({"benchmark": bench, "enhanced": enhanced, "max_sharpe": best_sharpe}, index=names),
        "concentrated": mc,
        "benchmark": mb,
        "enhanced": me,                # same risk as benchmark, more return
        "max_sharpe": ms,              # best risk-adjusted (secondary view)
        "weighted_avg_vol": wavg_vol,
        "diversification_vol_saved": wavg_vol - mb["volatility"],           # >= 0
        "incremental_return": me["exp_return"] - mb["exp_return"],          # SAME-RISK return pickup
        "incremental_sharpe": ms["sharpe"] - mb["sharpe"],
    }


def rolling_oos_pickup(class_name: str, spec: dict, parent: pd.Series, config: dict, seed: int,
                       window: int = 36) -> dict:
    """Walk-forward test: estimate the enhanced mix on a trailing window, apply it
    to the NEXT month, and accumulate the realised pickup over the benchmark mix,
    net of turnover cost. Tests whether the same-risk alpha survives out-of-sample.
    """
    max_w = config["optimiser"].get("intra_max_weight", 0.40)
    cost_bps = config["portfolio"].get("trading_cost_bps", 5)
    mu = np.array([s["exp_return"] for s in spec["components"]])
    bench = np.array([s["benchmark_weight"] for s in spec["components"]], float)
    bench = bench / bench.sum()
    R = generate_sub_returns(class_name, spec, parent, seed).to_numpy()
    n = len(R)
    enh_r, bch_r, turns, w = [], [], [], None
    for t in range(window, n - 1):
        # Re-optimise ANNUALLY (the realistic cadence for an intra-class tilt) and
        # hold between - monthly re-optimisation just churns turnover.
        to = 0.0
        if (t - window) % MONTHS_PER_YEAR == 0 or w is None:
            cov = np.cov(R[t - window:t], rowvar=False) * MONTHS_PER_YEAR
            bvol = np.sqrt(max(bench @ cov @ bench, 1e-12))
            new_w = _max_return_at_vol(mu, cov, bvol, max_w)
            to = 0.0 if w is None else float(np.abs(new_w - w).sum())
            w = new_w
        rnext = R[t + 1]
        enh_r.append(float(w @ rnext) - to * cost_bps / 1e4)
        bch_r.append(float(bench @ rnext))
        turns.append(to)
    enh, bch = np.array(enh_r), np.array(bch_r)
    return {
        "oos_pickup_net_bps": float((enh.mean() - bch.mean()) * MONTHS_PER_YEAR * 1e4),
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "n_months": len(enh),
    }


def run_intra_asset(market: MarketData, config: dict) -> dict:
    """Per-class within-asset-class analysis + portfolio-level uplift."""
    subs_cfg = config.get("sub_sleeves", {})
    seed0 = config["meta"].get("random_seed", 7)
    baseline = market.baseline_weights

    per_class = {}
    rows = []
    for i, (cls, spec) in enumerate(subs_cfg.items()):
        if cls not in market.returns.columns:
            continue
        seed = seed0 + 17 * (i + 1)
        res = analyse_class(cls, spec, market.returns[cls], config, seed)
        oos = rolling_oos_pickup(cls, spec, market.returns[cls], config, seed)
        per_class[cls] = res
        cw = float(baseline.get(cls, 0.0))
        rows.append({
            "asset_class": cls,
            "class_weight": cw,
            "incremental_return_bps": res["incremental_return"] * 1e4,         # in-sample same-risk pickup
            "oos_pickup_net_bps": oos["oos_pickup_net_bps"],                   # out-of-sample, net of cost
            "div_vol_saved_bps": res["diversification_vol_saved"] * 1e4,
            "oos_turnover": oos["avg_turnover"],
            "portfolio_uplift_bps": cw * res["incremental_return"] * 1e4,
            "portfolio_oos_uplift_bps": cw * oos["oos_pickup_net_bps"],
        })
    summary = pd.DataFrame(rows).set_index("asset_class")
    return {
        "per_class": per_class,
        "summary": summary,
        "portfolio_return_uplift_bps": float(summary["portfolio_uplift_bps"].sum()),
        "portfolio_oos_uplift_bps": float(summary["portfolio_oos_uplift_bps"].sum()),
    }
