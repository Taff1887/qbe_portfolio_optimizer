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


def _earnings(d_path: np.ndarray, rate_paths: np.ndarray, noise: np.ndarray,
              base_carry_m: float, tp_m: float) -> np.ndarray:
    """Plan-year earnings for a (possibly time-varying) duration schedule.

    Each month duration `d_t` both *earns a term premium* (tp_m per year of
    duration - the carry you need to clear the plan) and *bears rate risk*
    (-d_t * dr_t). Earnings = base carry + term-premium carry - rate MtM + noise.
    The trade-off is real: you need duration to make the number, but it is risk.
    """
    carry = base_carry_m * MONTHS_PER_YEAR + tp_m * d_path.sum()
    rate_impact = (d_path[None, :] * rate_paths).sum(axis=1)
    return carry - rate_impact + noise.sum(axis=1)


def _earnings_adaptive(d0: float, rate_paths: np.ndarray, noise: np.ndarray,
                       base_carry_m: float, tp_m: float, plan: float) -> tuple[np.ndarray, np.ndarray]:
    """Path-dependent policy: hold duration d0 while cumulative earnings are behind
    the plan pace, then cut to zero once ahead (bank the carry, then de-risk).

    Returns (plan-year earnings, average duration schedule across paths).
    """
    n = rate_paths.shape[0]
    cum = np.zeros(n)
    held = np.zeros((n, MONTHS_PER_YEAR))
    pace = plan / MONTHS_PER_YEAR
    for t in range(MONTHS_PER_YEAR):
        ahead = cum >= pace * t                       # banked enough so far?
        d_t = np.where(ahead, 0.0, d0)                # cut once ahead, else hold
        held[:, t] = d_t
        e_t = base_carry_m + tp_m * d_t - d_t * rate_paths[:, t] + noise[:, t]
        cum = cum + e_t
    return cum, held.mean(axis=0)


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
    # Base carry is set BELOW plan, so the book *needs* duration's term premium to
    # make the number - but duration is also rate risk. That tension is the lever.
    base_carry_ann = gcfg.get("base_carry_annual", plan - 0.004)
    base_carry_m = base_carry_ann / MONTHS_PER_YEAR
    tp_ann = gcfg.get("term_premium_per_year", 0.0025)   # extra carry per year of duration
    max_dur = gcfg.get("max_duration", 8.0)              # realistic ALM cap on surplus duration
    tp_m = tp_ann / MONTHS_PER_YEAR
    L = gcfg.get("plan_rate_sensitivity", config["currencies"]["AUD"]["liability_duration"])
    seed = config["meta"].get("random_seed", 7)

    changes = _rate_changes(config)
    if gcfg.get("demean_rates", True):
        changes = changes - changes.mean()               # duration as a pure risk lever, not a rate bet
    paths = _bootstrap_paths(changes, n_paths, block=3, seed=seed)
    idio_ann = gcfg.get("idio_earnings_vol", 0.010)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0.0, idio_ann / np.sqrt(MONTHS_PER_YEAR), (n_paths, MONTHS_PER_YEAR))

    rem = (MONTHS_PER_YEAR - np.arange(MONTHS_PER_YEAR)) / MONTHS_PER_YEAR
    grid = np.linspace(0.0, max_dur, 41)
    static = min(grid, key=lambda D: _metrics(_earnings(np.full(MONTHS_PER_YEAR, D), paths, noise, base_carry_m, tp_m), plan)["prob_miss_plan"])
    glide0 = min(grid, key=lambda D0: _metrics(_earnings(D0 * rem, paths, noise, base_carry_m, tp_m), plan)["prob_miss_plan"])
    adapt0 = min(grid, key=lambda D0: _metrics(_earnings_adaptive(D0, paths, noise, base_carry_m, tp_m, plan)[0], plan)["prob_miss_plan"])

    adapt_E, adapt_sched = _earnings_adaptive(adapt0, paths, noise, base_carry_m, tp_m, plan)
    policies = {
        "Short (no duration)": (np.zeros(MONTHS_PER_YEAR), _earnings(np.zeros(MONTHS_PER_YEAR), paths, noise, base_carry_m, tp_m)),
        f"Static (d={static:.1f}y)": (np.full(MONTHS_PER_YEAR, static), _earnings(np.full(MONTHS_PER_YEAR, static), paths, noise, base_carry_m, tp_m)),
        f"Glide (D0={glide0:.1f}y -> 0)": (glide0 * rem, _earnings(glide0 * rem, paths, noise, base_carry_m, tp_m)),
        f"Adaptive (hold {adapt0:.1f}y, cut when ahead)": (adapt_sched, adapt_E),
    }
    rows, dists, schedules = {}, {}, {}
    for name, (d_path, E) in policies.items():
        rows[name] = _metrics(E, plan)
        dists[name] = E
        schedules[name] = d_path

    return {
        "table": pd.DataFrame(rows).T,
        "distributions": dists,
        "schedules": schedules,
        "plan": plan,
        "L": L,
        "static_opt": static,
        "glide_opt_d0": glide0,
        "adaptive_opt_d0": adapt0,
        "months": np.arange(1, MONTHS_PER_YEAR + 1),
    }
