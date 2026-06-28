"""Lens - dynamic duration glide path (through-time earnings protection).

This models the lever the brief described: an insurer is held to an **annual
earnings plan**, and the CFO manages *interest-rate duration through the year* to
protect that number - taking duration early to immunise the remaining-year
earnings against rate moves, then winding it down toward year-end as the outcome
is banked and there is less left to protect.

Stylised model (deliberately transparent, like the rest of the lab). Over a
12-month plan year the remaining-year earnings have a rate exposure that *shrinks
as the year runs off*: a natural hedge target

    h_t = L * (months remaining at t) / 12

where `L` is the rate sensitivity of the plan (≈ a liability duration). Holding
asset duration `d_t` offsets it, so the month-t earnings rate impact is
`-(d_t - h_t) * dr_t`. Monthly earnings are carry minus that impact:

    e_t = carry_m - (d_t - h_t) * dr_t,    plan-year earnings E = sum_t e_t

We compare duration policies over many simulated rate paths (block-bootstrapped
from history, so the real rate dynamics drive the result):

- **Short**  (d_t = 0)            - fully exposed to rates all year.
- **Static** (d_t = D, optimised) - constant duration; over-hedges late in the year.
- **Glide**  (d_t = D0 * remaining/12, optimised) - declining, tracks the hedge target.

The glide path tracks `h_t` and so minimises the variance of plan-year earnings -
and therefore the probability of missing the plan - for the same expected earnings.
That is the value the static MVO/duration lenses cannot see, because it is a
*through-time* policy, not a point-in-time position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import DATA_PROCESSED, MONTHS_PER_YEAR


def _rate_changes(config: dict) -> np.ndarray:
    """Monthly government-rate changes to drive the simulation.

    Uses the saved factor series (`d_rate`) when present (real dynamics incl. the
    crises and the 2022 sell-off); otherwise a calibrated normal as a fallback.
    """
    path = DATA_PROCESSED / "factors.csv"
    if path.exists():
        f = pd.read_csv(path, index_col=0)
        if "d_rate" in f.columns:
            return f["d_rate"].dropna().to_numpy()
    rng = np.random.default_rng(config["meta"].get("random_seed", 7))
    return rng.normal(0.0, 0.0012, 600)


def _bootstrap_paths(changes: np.ndarray, n_paths: int, block: int, seed: int) -> np.ndarray:
    """Block-bootstrap n_paths x 12 monthly rate-change paths (preserves runs)."""
    rng = np.random.default_rng(seed)
    n = len(changes)
    out = np.empty((n_paths, MONTHS_PER_YEAR))
    for p in range(n_paths):
        idx: list[int] = []
        while len(idx) < MONTHS_PER_YEAR:
            s = int(rng.integers(0, n))
            idx.extend(((s + np.arange(block)) % n).tolist())
        out[p] = changes[np.array(idx[:MONTHS_PER_YEAR])]
    return out


def _hedge_target(L: float) -> np.ndarray:
    """Declining hedge target h_t = L * remaining/12 over the 12 months."""
    remaining = (MONTHS_PER_YEAR - np.arange(MONTHS_PER_YEAR)) / MONTHS_PER_YEAR
    return L * remaining


def _earnings(d_path: np.ndarray, rate_paths: np.ndarray, noise: np.ndarray,
              carry_m: float, L: float) -> np.ndarray:
    """Plan-year earnings for a duration schedule across all simulated rate paths.

    Earnings = carry - sum_t (d_t - h_t) dr_t + idiosyncratic (non-rate) noise.
    The noise (same draws for every policy - common random numbers) is the
    spread/credit/equity earnings variation a rate hedge cannot remove, so the
    best policy minimises *rate* earnings variance but never reaches zero.
    """
    h = _hedge_target(L)
    rate_impact = ((d_path - h)[None, :] * rate_paths).sum(axis=1)
    return carry_m * MONTHS_PER_YEAR - rate_impact + noise.sum(axis=1)


def _metrics(E: np.ndarray, plan: float) -> dict:
    return {
        "mean_earnings": float(E.mean()),
        "earnings_vol": float(E.std(ddof=1)),
        "prob_miss_plan": float((E < plan).mean()),
        "earnings_at_risk_5pc": float(np.percentile(E, 5)),
    }


def run_glide_path(config: dict, n_paths: int = 4000) -> dict:
    """Compare short / optimised-static / optimised-glide duration policies."""
    pcfg = config["portfolio"]
    gcfg = config.get("glide_path", {})
    plan = pcfg["plan_return_target"]
    # The book carries a small buffer above plan (running yield > plan target), so
    # lower earnings *variance* translates directly into a lower plan-miss chance.
    carry_ann = gcfg.get("carry_annual", plan + 0.003)
    carry_m = carry_ann / MONTHS_PER_YEAR
    L = gcfg.get("plan_rate_sensitivity", config["currencies"]["AUD"]["liability_duration"])
    seed = config["meta"].get("random_seed", 7)

    changes = _rate_changes(config)
    if gcfg.get("demean_rates", True):
        changes = changes - changes.mean()               # duration as a pure risk lever, not a rate bet
    paths = _bootstrap_paths(changes, n_paths, block=3, seed=seed)
    # Idiosyncratic (non-rate) monthly earnings noise, shared across policies.
    idio_ann = gcfg.get("idio_earnings_vol", 0.012)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0.0, idio_ann / np.sqrt(MONTHS_PER_YEAR), (n_paths, MONTHS_PER_YEAR))

    rem = (MONTHS_PER_YEAR - np.arange(MONTHS_PER_YEAR)) / MONTHS_PER_YEAR
    grid = np.linspace(0.0, 2.0 * L, 41)
    static = min(grid, key=lambda D: _metrics(_earnings(np.full(MONTHS_PER_YEAR, D), paths, noise, carry_m, L), plan)["prob_miss_plan"])
    d0 = min(grid, key=lambda D0: _metrics(_earnings(D0 * rem, paths, noise, carry_m, L), plan)["prob_miss_plan"])

    policies = {
        "Short (no duration)": np.zeros(MONTHS_PER_YEAR),
        f"Static (d={static:.1f}y)": np.full(MONTHS_PER_YEAR, static),
        f"Glide (D0={d0:.1f}y -> 0)": d0 * rem,
    }
    rows, dists, schedules = {}, {}, {}
    for name, d_path in policies.items():
        E = _earnings(d_path, paths, noise, carry_m, L)
        rows[name] = _metrics(E, plan)
        dists[name] = E
        schedules[name] = d_path

    table = pd.DataFrame(rows).T
    return {
        "table": table,
        "distributions": dists,
        "schedules": schedules,
        "plan": plan,
        "L": L,
        "static_opt": static,
        "glide_opt_d0": d0,
        "months": np.arange(1, MONTHS_PER_YEAR + 1),
    }
