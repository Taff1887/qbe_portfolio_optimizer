# QBE-style Portfolio Optimisation Research Lab

A practical, **multi-lens** institutional portfolio-construction **research platform** inspired by an insurer's (QBE-style) investment book. It is built to do two things at once:

1. **Compare construction philosophies** — equal weight, mean-variance, minimum variance, risk parity, maximum diversification and capital-aware optimisers — each independent, each scored on *exactly the same metrics*. The goal is comparison, not crowning one "perfect" portfolio.
2. **Evaluate every portfolio through the lenses an insurer actually manages to** — return, risk, **regulatory capital**, **accounting (P&L vs OCI)**, **through-time earnings**, and **asset–liability (duration) management** — at the same time.

It is fully **local-file driven** — no APIs, no Bloomberg. Realistic monthly dummy data (from January 2000) is generated from a transparent factor model so the whole thing runs out of the box. Real data drops straight in later. The architecture is modular and extensible: each optimiser is its own function, and new ones (Black-Litterman, robust, ML forecasts) plug into the same comparison.

```bash
pip install -r requirements.txt
python src/main.py
```

---

## Why classic mean-variance optimisation is not enough

Mean-variance optimisation (MVO) answers one question: *for a given volatility, what mix maximises expected return?* For an insurer that is necessary but badly incomplete, because the same return/volatility point can imply:

- a very different **regulatory capital** charge (LAGIC asset risk charge), and therefore a very different **return on capital**;
- a very different **earnings path** — how much of the return is predictable **carry** versus volatile **mark-to-market**, and how often the annual **plan** is missed;
- a very different **accounting footprint** — whether moves hit the **P&L** (FVTPL, liability-matching book) or **OCI** (FVOCI, surplus book);
- a very different **duration gap** to insurance liabilities, which drives both economic surplus and earnings volatility.

Two portfolios with identical Sharpe ratios can have double the capital charge, or swing annual earnings by twice as much. This framework puts all of those lenses side by side.

## Construction philosophies (compared on identical metrics)

The lab builds several candidate portfolios from independent construction philosophies and scores them on one common metric set (expected return, volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, duration, capital charge, worst-case stress loss, liquidity, diversification ratio and turnover):

| Philosophy | Module / function | Idea |
|---|---|---|
| **Equal weight (1/N)** | `construction.equal_weight` | The naive, unconstrained benchmark every optimiser must beat. |
| **Mean-variance (max-Sharpe)** | `optimizer.max_sharpe` | Most return per unit of volatility, within insurer constraints. |
| **Minimum variance** | `optimizer.min_variance` | Lowest achievable volatility. |
| **Risk parity** | `construction.risk_parity` | Every asset contributes an equal share of portfolio risk. |
| **Maximum diversification** | `construction.max_diversification` | Maximise the diversification ratio (weighted-avg asset vol / portfolio vol). |
| **Max return-on-capital** | `optimizer.max_return_on_capital` | Most return per unit of regulatory capital. |
| **Capital-budgeted** | `optimizer.max_return_capital_budget` | Most return at a *fixed* capital charge. |
| **Black-Litterman / Robust** | `construction.black_litterman`, `construction.robust_optimizer` | Roadmap **placeholders** — documented scaffolds, not yet wired in. |

Risk parity and maximum diversification are solved under the **same** insurer constraints as mean-variance (min core fixed income, currency buckets, per-asset caps), so the comparison is like-for-like; equal weight is the deliberately unconstrained naive benchmark. Every philosophy is also run through **every stress scenario** (the stress grid) and capital/earnings/duration lenses below.

## The six lenses

| # | Lens | Module | What it answers |
|---|------|--------|-----------------|
| 1 | **Mean-variance optimisation** (+ construction philosophies) | `optimizer.py`, `construction.py` | Constrained efficient frontier, max-Sharpe and min-variance portfolios, plus equal-weight / risk-parity / max-diversification, subject to insurer constraints (min core FI, currency buckets, per-asset caps). |
| 2 | **Instantaneous stress testing** | `stress_testing.py` | Deterministic shocks (rates ±100bp, credit +100/+150/+200bp, equity −20/−30%, property −15%, high-yield widening, structured, recession, inflation) via duration / spread duration / beta, split into P&L and OCI impact; a stress grid runs every scenario against every portfolio. |
| 3 / 6 | **Through-time earnings & carry** | `earnings_model.py` | Decomposes returns into carry vs mark-to-market and P&L vs OCI; annual earnings volatility and probability of missing the plan; a worked duration-vs-earnings-stability example. |
| 4 | **Duration / ALM** | `duration_model.py` | Asset vs liability duration and the gap by currency; separates the matched (P&L) book from the surplus (OCI) book; rate-shock impact split into earnings vs economic. |
| 5 | **LAGIC-style capital** | `lagic_capital.py` | Simplified APRA/LAGIC asset risk charge: prescribed category stresses, a diversified aggregate, a worst-scenario panel, return on capital and marginal capital by asset. |
| — | **Comparison & metrics** | `reporting.py`, `metrics.py` | Every portfolio compared on identical metrics — return, volatility, Sharpe, **Sortino**, drawdown, **VaR**, **CVaR**, duration, capital charge, **stress loss**, liquidity, **diversification ratio**, **turnover** and **tracking error**. |

**Additional analyses** (because a multi-lens framework should keep adding lenses): a **capital-efficient frontier**, a **max-return-on-capital** portfolio, and a **capital-budgeted optimiser** (max return at a fixed capital charge - capital, not volatility, is an insurer's binding constraint); **risk budgeting** (each asset's and currency's share of portfolio risk, diversification ratio, correlation matrix, marginal efficiency) plus **earnings-volatility attribution** (which sleeves drive P&L instability - distinct from total risk); **earnings-at-risk** (block-bootstrap of the plan-year P&L tail: EaR, CTE95, P(miss plan)); and **diagnostics** (liquidity tiers, concentration / effective number of assets, rating distribution, surplus & coverage with surplus-at-risk, and realised behaviour through the GFC/COVID/2022 episodes). The within-class lens is also stress-tested **out-of-sample, net of costs** - honestly showing the implementation alpha is robust only where sub-sleeve dispersion is structural. Finally, a **within-asset-class** lens (`intra_asset.py`): the top-level SAA is fixed, so it decomposes each class into sub-sleeves (e.g. the AUD sovereign curve, IG by rating/sector, equity by region/style) and finds the mix that earns **more return at the same risk** - small, repeatable *implementation* alpha (a few bps per class) orthogonal to the policy allocation. A single comprehensive **`outputs/report.md`** pulls every lens together with embedded charts.

## Portfolio context (insurer assumptions)

- **~85% core fixed income** (sovereigns, IG/corporate credit, senior structured, cash) used mainly for **ALM / liability matching** — this is the **P&L (FVTPL)** book.
- **~15% risk assets** (high yield, mezz structured, private credit, infrastructure, listed equity, unlisted property, **gold**) — mostly the **OCI (FVOCI) / surplus** book. The user can scale this to **20%** for scenario analysis (existing risk assets rescale pro rata). Gold is modelled as a no-carry, safe-haven diversifier (a positive bid in the GFC/COVID episodes).
- **Currency is largely given**: buckets are **AUD, USD, GBP, EUR, NZD, CAD**; risk is concentrated in the larger currencies. Currency min/max bounds are honoured by the optimiser.
- **P&L vs OCI** distinction is explicit everywhere: mark-to-market on the P&L book hits earnings now; on the OCI book it sits in reserves.

## Repository structure

```
qbe_portfolio_optimizer/
  README.md
  requirements.txt
  config.yaml            # single source of truth: assets, durations, capital factors, scenarios
  data/
    raw/                 # (place real source files here)
    processed/           # returns.csv (generated by dummy_data; replace with real)
  outputs/
    charts/              # PNG charts
    tables/              # CSV tables
    summary_report.md    # generated one-page summary
    report.md            # generated full report (Part A philosophies, Part B lenses)
  src/
    main.py              # run everything
    utils.py             # paths, config, plotting, metric helpers
    metrics.py           # Sortino, VaR, CVaR, tracking error, downside deviation
    dummy_data.py        # factor-model return generator (incl. gold safe-haven leg)
    data_loader.py       # load config + metadata + returns
    portfolio.py         # Portfolio object (weights -> every view + tail metrics)
    optimizer.py         # lens 1: constrained MVO (+ capital-efficient frontier, max-RoC)
    construction.py      # lens 1: equal-weight, risk-parity, max-diversification (+ BL/robust placeholders)
    stress_testing.py    # lens 2: instantaneous shocks + stress grid (every portfolio x scenario)
    earnings_model.py    # lens 3/6: through-time carry / P&L / OCI
    duration_model.py    # lens 4: ALM / duration gap
    lagic_capital.py     # lens 5: LAGIC-style capital
    risk_attribution.py  # extra: risk budgeting, diversification, marginal efficiency
    diagnostics.py       # extra: liquidity, concentration, rating, surplus, historical stress
    intra_asset.py       # extra: within-asset-class diversification & same-risk return pickup
    earnings_risk.py     # extra: earnings-at-risk (block-bootstrap plan-year P&L tail)
    reporting.py         # tables, charts, markdown (incl. full report.md)
```

## How to run

```bash
pip install -r requirements.txt   # numpy, pandas, scipy, matplotlib, pyyaml
python src/main.py
```

This will: generate dummy data (if `data/processed/returns.csv` is missing), build the baseline portfolio, run all six lenses, write tables/charts to `outputs/`, and print a concise summary. To regenerate dummy data, delete `data/processed/returns.csv` (or call `dummy_data.write_dummy_data()`).

## What each output means

**Tables (`outputs/tables/`)**
- `portfolio_comparison.csv` — every construction philosophy (equal-weight, max-Sharpe, min-variance, risk-parity, max-diversification, max-RoC, capital-budgeted, baseline, risk-20%) across the full identical metric set: return, vol, Sharpe, Sortino, drawdown, VaR, CVaR, duration, capital charge, stress loss, liquidity, diversification ratio, turnover, tracking error, carry and return on capital. *The core comparison.*
- `efficient_frontier.csv` — the constrained efficient frontier (return vs volatility).
- `stress_scenarios.csv` — total / P&L / OCI impact for each shock (baseline).
- `stress_grid.csv` — every stress scenario against every portfolio (total impact).
- `lagic_asset_charges.csv`, `lagic_scenario_charges.csv` — capital charge by asset and by stress scenario.
- `earnings_annual.csv` — annual total / carry / MtM / P&L / OCI.
- `duration_by_currency.csv`, `duration_rate_shock.csv` — duration gaps and the earnings-vs-economic split of a rate shock.
- `duration_earnings_example.csv` — the matched-vs-unmatched earnings-stability worked example.
- `baseline_allocation.csv` — the baseline weights.

**Charts (`outputs/charts/`)** — efficient frontier (with every philosophy), philosophy risk-adjusted-return comparison, allocation comparison, stress impacts (P&L vs OCI), stress grid heatmap, capital charge by category, capital-efficient frontier, return-vs-volatility, return-vs-capital, rolling annual return, earnings vs plan, earnings-at-risk distribution, duration gap by currency, duration contribution by asset, drawdown, correlation heatmap, risk contribution, marginal efficiency, liquidity/rating, within-class uplift, and the duration/earnings-stability example.

**`outputs/report.md`** — the full report: Part A compares construction philosophies; Part B applies the evaluation lenses. **`outputs/summary_report.md`** — a one-page narrative read.

## Replacing the dummy data with real data

The framework is built so this is a config + one-CSV change:

1. **Returns** — replace `data/processed/returns.csv` with a real monthly total-return history (rows = dates, columns = the asset names in `config.yaml`). Sources later: Bloomberg / ICE indices, fund returns, or returns implied from the annual report.
2. **Static assumptions** — edit `config.yaml`: each asset's `yield`, `duration`, `spread_duration`, `rating`, `currency`, `accounting`, `capital_category`, `liquidity`, and `baseline_weight`. These are exactly the fields you would lift from index factsheets, the investment supplement of the annual report, or a risk system.
3. **Liability durations** and **capital factors** — set `currencies[*].liability_duration` from the actuarial liability profile and `lagic.stress_factors` from the current prudential standard.

No code changes are required — every module reads from config and `returns.csv`.

## Modelling choices & simplifications (read me)

- **Dummy data** is a factor model (rates / credit / equity / property factors + idiosyncratic) with embedded GFC, COVID and 2022 episodes; vols are calibrated to each asset's `ann_vol`.
- **Expected returns for MVO** use the forward `exp_return` assumptions in config, **not** the historical mean (which is inflated by the modelled secular rate decline) — mirroring real practice (forward CMAs for return, empirical data for risk).
- **LAGIC** here is a *simplified, illustrative* asset-risk-charge model: prescribed category stresses + a correlated aggregate + a worst-scenario panel. It is **not** the legal standard and omits insurance-risk, operational-risk and the full aggregation/diversification rules.
- **Stress impacts** are first-order (duration/beta) approximations — no convexity.
- **Liabilities** are stylised: assumed backed ~1:1 by the P&L asset book, with durations from config, to make the ALM and accounting points concrete.

## Future enhancements

- Real data adapters (Bloomberg / ICE / annual-report parsers) behind `data_loader.py`.
- A fuller LAGIC build: insurance + operational risk, prescribed correlation matrix, asset concentration and counterparty grades.
- Liability cash-flow model (key-rate durations, not just effective duration) and surplus-at-risk.
- Stochastic earnings simulation (Monte Carlo paths) and Conditional Tail Expectation on annual earnings.
- **Black-Litterman** and **robust optimisation** — currently documented placeholders in `construction.py`; the design notes are in the docstrings, ready to be wired into the same comparison.
- Further philosophies behind the same interface: factor models, ML expected-return forecasts, regime switching, Bayesian / multi-objective optimisation.
- New-money / turnover and transaction-cost-aware rebalancing.

---

*This is a teaching/prototype framework on dummy data. It is not investment advice and the LAGIC module is illustrative, not a regulatory calculation.*
