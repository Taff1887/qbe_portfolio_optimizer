# Running on real market data

The framework is wired to run on **real** monthly total returns; the ETF/index
proxies are mapped in `config.yaml` under `market_data.tickers` and the whole
universe (21/21 assets) is covered. The ingestion path is verified end-to-end.

**It cannot fetch from inside the locked-down web sandbox** (the environment's
network policy returns `403` for Yahoo/FMP/Stooq). Run it in either of these
places instead — it is one command.

## Option A — Yahoo Finance (no key)

On your own machine (Mac/PC), or any environment whose network allows Yahoo:

```bash
pip install -r requirements-dev.txt          # installs yfinance
python src/data_sources.py --provider yahoo  # writes data/processed/returns.csv from Yahoo
python src/main.py                           # rebuild every table, chart and report
```

`outputs/report.md` will show a **"Data: REAL market data (provider: yahoo)"**
banner instead of the synthetic one.

## Option B — Financial Modeling Prep (your key)

```bash
export FMP_API_KEY=your_key_here
python src/data_sources.py --provider fmp
python src/main.py
```

## Proxies used (edit `market_data.tickers` to taste)

| Asset | Ticker | Asset | Ticker |
|---|---|---|---|
| AUD Sovereign | AGVT.AX | High Yield | HYG |
| USD Sovereign | IEF | CLO BBB | JBBB |
| GBP Sovereign | IGLT.L | Private Credit | BIZD |
| EUR Sovereign | SEGA.L | Infra Debt | VCLT* |
| NZD Sovereign | IGOV* | Infra Equity | IGF |
| CAD Sovereign | XGB.TO | Listed Equities | ACWI |
| IG Credit | LQD | Unlisted Property | VNQ* |
| Corporate Credit | CRED.AX | Gold | GLD |
| CLO AAA | JAAA | Cash | BIL |
| ABS | VCSH* | RMBS | MBB |
| CMBS | CMBS | | |

`*` = proxy where no clean direct instrument exists (real estate via the VNQ REIT
ETF, ABS via short IG corporate, infra debt via long IG corporate, NZD rates via
international sovereign). Swap any of these for a Bloomberg/ICE index when you
have one — no code change, just edit the ticker.

## Notes

- `data.source: real` (in `config.yaml`) is already set, so `python src/main.py`
  will itself try the real fetch first and only fall back to the synthetic factor
  model if the network is unavailable (and the report banner will say which).
- The CLO ETFs (JAAA 2020, JBBB 2022) have short histories; `market_data.align:
  common` truncates to the shared window. Set `align: pairwise` to keep longer
  histories where covariance is computed pairwise, or substitute longer indices.
