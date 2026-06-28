"""Lens-level invariants: stress, reverse stress, capital, Pareto, glide, regimes."""

import numpy as np

import duration_model
import glide_path
import lagic_capital
import multi_objective
import stress_testing


def test_stress_total_is_pnl_plus_oci(base, config):
    s = stress_testing.run_stress_tests(base, config)
    assert np.allclose(s["total_impact"], s["pnl_impact"] + s["oci_impact"], atol=1e-9)


def test_reverse_stress_breach_recovers_limit(base, config):
    gap = duration_model.run_duration(base, config)["total_dollar_duration_gap"]
    rev = stress_testing.reverse_stress(base, config, gap)
    # A +rates_bps move on the gross book should cost ~ the P&L limit.
    row = rev.loc[[i for i in rev.index if i.startswith("P&L")][0]]
    implied = row["rates_bps"] / 1e4 * base.duration()
    assert np.isclose(implied, row["limit_loss"], rtol=1e-6)


def test_lagic_charges_nonneg_and_roc(base, config):
    lg = lagic_capital.run_lagic(base, config)
    assert lg["capital_charge"] >= 0
    assert (lg["asset_charges"] >= -1e-12).all()
    assert lg["return_on_capital"] > 0


def test_lagic_full_total_at_least_diversified(base, config):
    gap = duration_model.run_duration(base, config)["total_dollar_duration_gap"]
    lf = lagic_capital.lagic_full(base, config, gap)
    assert lf["total_capital_requirement"] >= lf["diversified_modules"] - 1e-12
    assert lf["concentration_addon"] >= 0


def test_concentration_exempts_sovereign(base, config):
    # The baseline holds >10% sovereign buckets; those must not be charged.
    addon = lagic_capital.concentration_addon(base, config)
    # Build the same excess WITHOUT the exemption and confirm it would be larger.
    w = base.weights
    lim = config["lagic"]["concentration_single_limit"]
    raw_excess = (w - lim).clip(lower=0).sum() * config["lagic"]["concentration_factor"]
    assert addon <= raw_excess + 1e-12
    assert raw_excess > addon                       # exemption actually bites


def test_dominates_logic():
    base = {"exp_return": 0.05, "volatility": 0.03, "capital": 0.03,
            "earnings_vol": 0.02, "worst_stress": -0.10}
    better = {**base, "volatility": 0.02}           # strictly lower vol, rest equal
    worse = {**base, "exp_return": 0.04}            # lower return
    assert multi_objective.dominates(better, base)
    assert not multi_objective.dominates(worse, base)
    assert not multi_objective.dominates(base, base)  # not strictly better


def test_pareto_best_dominates_baseline(market, config, base):
    res = multi_objective.pareto_search(market, config, base, seeds={
        "Max-Sharpe": __import__("optimizer").max_sharpe(market, config)}, grid=3)
    if res["best_portfolio"] is not None:
        bo = res["baseline_objectives"]
        o = multi_objective.objectives(res["best_portfolio"], market, config)
        assert multi_objective.dominates(o, bo)


def test_glide_adaptive_beats_short(config):
    gp = glide_path.run_glide_path(config, n_paths=2000)
    t = gp["table"]
    short = t.loc[[i for i in t.index if i.startswith("Short")][0], "prob_miss_plan"]
    adapt = t.loc[[i for i in t.index if i.startswith("Adaptive")][0], "prob_miss_plan"]
    assert adapt <= short                            # adaptive never worse on plan-miss
