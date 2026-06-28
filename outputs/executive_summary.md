# Portfolio Optimisation Research Lab — Executive Summary

*~2-page brief. The platform prefers real market data (FMP/Yahoo); where the run environment blocks those hosts it falls back to a transparent factor model and flags the report accordingly. Full detail + charts in `outputs/report.md`; every figure reconciles to a CSV in `outputs/tables/`.*

---

## What this is

A research **platform**, not a single optimiser. It builds an insurer's portfolio several different ways (construction *philosophies* — equal weight, mean-variance, min-variance, risk parity, max diversification, **Black-Litterman, robust, ML**, and capital/earnings-aware optimisers), scores them all on one identical metric set, then runs every book through the lenses an insurer is actually governed by: **regulatory capital, earnings stability, accounting (P&L vs OCI), asset–liability duration, and through-time policy**. The aim is a portfolio that is *acceptable across every lens* — and, where possible, a change that improves several at once ("grow the pie").

## Best findings

**1. There are genuine "free lunches" over the current book.** A multi-objective **Pareto search** found 8 portfolios that beat the baseline on *every* objective simultaneously. The best-balanced one earns the same-or-more return (5.1%) at **~⅔ the volatility, well under half the capital, lower earnings volatility and a smaller worst-case stress**. The current book is not on the efficient surface.

**2. Capital — not volatility — is the binding lens, and it re-ranks everything.** By return on capital, Max-RoC (4.2×) and Min-Variance (3.6×) dominate the baseline (2.0×). Holding the capital budget fixed, the capital-budgeted optimiser lifts return from 5.1% to **5.9%**. The lab now computes a LAGIC-style charge (worst of an 8-scenario panel) **and** a fuller requirement (rate-net-of-liabilities + insurance + operational + concentration), where **insurance risk dominates** and the matched book's rate capital is tiny (~0.6%) — the structural payoff of ALM.

**3. The dynamic duration "glide path" is the high-value novel piece.** Modelling the CFO lever — duration earns the carry needed to make the plan but bears rate risk — an **adaptive, path-dependent** policy (hold duration while behind the plan, cut once ahead) **lowers the plan-miss probability to ~28% vs ~35–38% for static or fixed-glide management, at the lowest earnings volatility**, from the same duration budget. No point-in-time optimiser can see this; it is a through-time *policy*, and it is the most novel piece.

**4. Structured credit is the strongest, capital-aware opportunity.** The granular deep-dive (CLOs AAA→BB US/EU, ABS/RMBS/CMBS) finds a same-risk pickup of **~+40 bps**, but the capital angle is sharper: staying in **senior** tranches lifts return on capital from **1.2× to 2.1×** for almost the same return — the mezzanine reach is not paid for on a capital basis. This is the strategic growth area and the result is robust out-of-sample.

**5. The track record is carry plus a one-off rates windfall — not skill.** Factor analysis decomposes the return into **~5.4% carry/alpha + 2.6% from the secular fall in rates + 0.9% equity**. The rates piece cannot repeat — the strongest argument for the multi-lens discipline and for forward humility.

**6. The earnings-plan risk point-in-time models miss is real but manageable.** ~16% chance of missing the 4.5% plan; earnings-at-risk (5%) ~1.4%; carry funds ~59% of return — which is what keeps the miss probability that low. The glide path (finding 3) is the lever to push it lower.

**7. Risk realism — two checks every static optimiser misses.** A **regime lens** shows the book is **~40% more volatile in risk-off regimes** (and its diversification ratio falls from ~2.2 to ~1.6) — size risk for the bad regime, not the average. A **reverse stress test** shows no single factor can erode the 18% surplus (equity would need >100%), but the **P&L is most rate-sensitive** — a ~48bp move alone costs 2% of earnings, which is exactly why the glide path matters.

## The contest at a glance (illustrative, current data basis)

| Portfolio | Return | Vol | Sharpe | Capital | RoC | Worst stress |
|---|---|---|---|---|---|---|
| **Baseline** | 5.1% | 3.3% | 1.75 | 2.5% | 2.0× | −11.3% |
| **Max-Sharpe** | 5.1% | 1.7% | 2.15 | 2.1% | 2.4× | −6.2% |
| **Min-Variance** | 4.7% | 1.6% | 1.97 | 1.3% | 3.6× | −6.3% |
| **Robust** | 5.1% | 1.7% | 2.14 | 2.0% | — | −6.2% |
| **ML-Forecast** | 5.2% | 1.8% | 2.14 | 2.6% | — | −6.7% |
| **Pareto-Balanced** | 5.1% | 2.1% | — | 1.7% | — | −7.8% |

*No single column picks the winner — that is the point. The choice depends on which constraint is binding for the business that period.*

## Recommended next steps (priority order)

1. **Get real data flowing** — supply an FMP key or run where Yahoo/FMP are reachable (the adapter and ticker map are built); then re-baseline on the actual QBE disclosed allocation.
2. **Operationalise the glide path** — make it path-dependent (cut duration once the plan is banked) and drive it off the real P&L-book duration and liability profile. Highest novel value.
3. **Build structured credit granularly on real index series** — the tickers are mapped; model the senior/mezz capital cliff explicitly.
4. **Make capital + earnings-stability standing constraints** in every optimiser and routinely search for the cheapest *implementable* Pareto improvement over the live book.
5. **Deepen the capital model** to the prescribed standard (real liability cash-flows for the rate module, counterparty/reinsurance risk).
6. **Extend the philosophy set** — regime-switching and richer ML forecasters behind the same comparison.

## Quality / engineering

The platform is modular (one file per lens/optimiser), documented, and covered by a **33-test `pytest` suite** (optimiser feasibility, capital/earnings constraints, stress identities, Pareto/glide logic, data transforms). New philosophies and lenses plug into the same comparison and report.

## Caveats

Where real data is unavailable the figures are on a **transparent synthetic factor model** (with embedded GFC/COVID/2022 episodes) — illustrative, not forecasts; the report banner says which basis was used. The LAGIC and glide-path models are simplified and educational. The value is the **framework and the relative comparisons**; absolute numbers refresh the moment real data is connected.
