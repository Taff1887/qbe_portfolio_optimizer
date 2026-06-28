# Portfolio Optimisation Research Lab — Executive Summary

*Two-page brief. Dummy data (Jan 2000–2024), built to mirror a QBE-style insurer book. Full detail + charts in `outputs/report.md`; every figure reconciles to a CSV in `outputs/tables/`.*

---

## What this is

A research **platform**, not a single optimiser. It builds an insurer's portfolio several different ways (construction *philosophies*), scores them all on one identical metric set, then runs every book through the lenses an insurer is actually governed by: **regulatory capital, earnings stability, accounting (P&L vs OCI) and asset–liability duration**. The aim is a portfolio that is *acceptable across every lens* — and, where possible, a change that improves several at once ("grow the pie, don't re-slice it").

## Best findings

**1. There is a clean "grow the pie" move — Max-Sharpe dominates the baseline on almost every axis.** For essentially the *same* expected return (5.08% vs 5.09%) it runs at **half the volatility** (1.7% vs 3.3%), **~30% less capital** (2.1% vs 2.5%), **a third of the drawdown** (−4.5% vs −16.1%) and a far smaller worst-case stress (−6.2% vs −11.3%). This is not a return/risk trade-off — it is a strict improvement, the single most actionable result.

**2. Capital — not volatility — is the binding lens, and it re-ranks everything.** Ranked by **return on capital**, the order flips entirely: Max-RoC **4.2×** and Min-Variance **3.6×** versus the Baseline's **2.0×**. Holding the capital budget fixed, the capital-budgeted optimiser lifts return from 5.09% to **5.88%**. For a regulated insurer this is the efficiency measure that matters, and the lab now computes it (a LAGIC-style worst-of-8-scenario asset risk charge) — the gap that originally motivated the build.

**3. The track record is carry plus a one-off rates windfall — not directional skill.** Factor analysis decomposes the return into **~5.4% carry/alpha + 2.6% from the secular fall in rates + 0.9% equity**. The rates contribution is a tailwind that cannot repeat. This is the strongest argument for the multi-lens discipline and for forward humility: do not extrapolate the historical headline.

**4. Structured credit is the best *implementation* opportunity — and it survives out-of-sample.** Within-class analysis finds a same-risk pickup of **+44.7 bps in-sample and +35 bps out-of-sample, net of cost** in structured credit (CLOs by rating, ABS/RMBS/CMBS) — by far the largest, and one of only three sleeves (with Listed Equities and IG Credit) whose edge is robust out-of-sample. This lines up exactly with the stated strategic growth area.

**5. The earnings-plan risk that point-in-time models miss is real but manageable.** Through-time: a **16% chance of missing the 4.5% plan**, earnings-at-risk (5%) of **1.4%**, and a worst-1-in-20 outcome (CTE95) of **−0.5%**. Predictable carry funds ~59% of the return, which is what keeps the plan-miss probability that low.

**6. The ALM structure works as intended.** A +100bp shock moves **economic surplus +0.29% but P&L earnings +0.48%** — the difference is precisely the OCI/surplus book's rate exposure. Confirms the "matched in P&L, deliberately long in OCI" design: duration is taken for the long-term economics without disturbing the earnings line.

## The contest at a glance

| Portfolio | Return | Vol | Sharpe | Max DD | Capital | RoC | Worst stress |
|---|---|---|---|---|---|---|---|
| **Baseline** | 5.09% | 3.28% | 1.75 | −16.1% | 2.54% | 2.01× | −11.3% |
| **Max-Sharpe** | 5.08% | 1.70% | 2.15 | −4.5% | 2.11% | 2.41× | −6.2% |
| **Min-Variance** | 4.72% | 1.57% | 1.97 | −5.1% | 1.30% | 3.65× | −6.3% |
| **Risk-Parity** | 4.93% | 2.23% | 1.92 | −8.2% | 2.01% | 2.46× | −7.7% |
| **Max-RoC** | 4.39% | 2.91% | 1.57 | −15.0% | 1.04% | 4.24× | −8.4% |
| **Capital-Budgeted** | 5.88% | 3.47% | 1.59 | −15.3% | 3.55% | 1.66× | −12.1% |

*No single column picks the winner — that is the point. The choice depends on which constraint is binding for the business that year.*

## What it means

- The **current baseline is leaving efficiency on the table**: the same return is available at materially lower risk and capital. The honest caveat is implementation cost — Max-Sharpe is ~67% turnover from today's book, so it is a direction to phase toward, not an overnight trade.
- **Capital efficiency and earnings stability should be explicit objectives**, not by-products of a vol-based optimisation. When they are, the recommended book changes.
- The biggest *repeatable* edge is **selection within structured credit**, not a top-level reallocation — which fits an insurer whose strategic allocation is largely fixed.

## Recommended next steps (priority order)

1. **Make capital and earnings-stability hard objectives** in the optimiser (not just the capital-aware variants), and search for Pareto improvements over the current book.
2. **Build structured credit out granularly** — CLO tranches by rating/vintage, US vs EU, ABS/RMBS/CMBS — on real index data. Highest-conviction, strategically aligned.
3. **Model the dynamic duration "glide path"** — add duration early in the plan year to protect the earnings target, wind it down toward year-end (the CFO lever from the brief). This is the novel, high-value piece nothing yet captures.
4. **Deepen the capital model** toward full LAGIC (rate stress net of liabilities, insurance/operational risk, prescribed correlations, concentration add-ons).
5. **Wire in the scaffolded optimisers** — Black-Litterman, robust, then ML return forecasts — behind the same comparison ("don't get stuck in one way of thinking").
6. **Swap in real data** — Bloomberg/ICE feeds for returns and observable factor series; the framework is built for a one-CSV + config change.

## Caveats

All figures are on **transparent dummy data** (a factor model with embedded GFC/COVID/2022 episodes), so they are illustrative, not forecasts — realised returns inherit a one-off rate-decline tailwind. The LAGIC module is a simplified, educational asset-risk charge, not the legal standard. The value here is the **framework and the relative comparisons**; the absolute numbers refresh the moment real data is dropped in.
