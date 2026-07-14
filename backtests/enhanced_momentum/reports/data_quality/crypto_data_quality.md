# Data Quality Report: CRYPTO

Generated: 2026-07-14 04:33
Source: Binance Public Data monthly 1d kline archives
Universe: Archived historical USDT pairs, top 50 by lagged volume

**Known caveats:**

- Historical universe is discovered from Binance Public Data archive; archive files may be revised by Binance
- Listing bias: new coins enter universe mid-sample
- 24/7 trading — no natural close; Binance uses 00:00 UTC candle close
- Extreme returns common (>20% daily) — not necessarily data errors
- Market proxy (BTC) is also in the universe — creates mild circularity

## 1. Market Coverage

| Metric | Value |
|--------|-------|
| First date | 2018-01-01 |
| Last date | 2025-06-30 |
| Trading days | 2,738 |
| Calendar years | 8 |
| Total assets (ever) | 537 |
| Avg trading days/year | 342 |

## 2. Assets Available Per Year

| Year | Mean | Min | Max |
|------|------|-----|-----|
| 2018 | 13 | 0 | 18 |
| 2019 | 26 | 17 | 37 |
| 2020 | 43 | 28 | 50 |
| 2021 | 50 | 49 | 50 |
| 2022 | 50 | 48 | 50 |
| 2023 | 50 | 49 | 50 |
| 2024 | 50 | 49 | 50 |
| 2025 | 50 | 49 | 50 |

## 3. Asset Lifetimes

### Calendar lifetime

| Statistic | Calendar days | Years |
|-----------|---------------|-------|
| Median lifetime | 1223 | 3.4 |
| Mean lifetime | 1122 | 3.1 |
| Min lifetime | 0 | 0.0 |
| Max lifetime | 2737 | 7.5 |

### Valid price observations

| Statistic | Observations |
|-----------|--------------|
| Median valid observations | 1224 |
| Mean valid observations | 1123 |
| Min valid observations | 1 |
| Max valid observations | 2738 |

Assets with <126 valid observations: 43 (8.0%)

Assets with <252 valid observations: 81 (15.1%)

## 4. Listings / Historical Exits by Year

An asset is classified as a historical exit when its last price observation is more than 30 calendar days before the panel end. This is an inactivity heuristic, not necessarily a formal exchange delisting event.

| Year | New listings | Historical exits |
|------|--------------|------------------|
| 2018 | 23 | 3 |
| 2019 | 67 | 3 |
| 2020 | 108 | 8 |
| 2021 | 123 | 4 |
| 2022 | 44 | 29 |
| 2023 | 57 | 18 |
| 2024 | 62 | 41 |
| 2025 | 53 | 34 |

**Historical exits before panel end: 140**
**Assets still observed near panel end: 397**

Historical coverage check: 26.1% of assets have no observations within the final 30 calendar days of the panel. The presence of inactive historical assets indicates that the universe is not restricted to end-of-sample survivors.

## 5. Missingness (within active lifetime)

| Statistic | Value |
|-----------|-------|
| Assets with zero gaps | 530 (98.7%) |
| Median missing % | 0.00% |
| Mean missing % | 0.01% |
| Worst asset missing % | 1.23% |

Worst 5 assets by gap share:

| Asset | Active days | Missing | % |
|-------|-------------|---------|---|
| BCCUSDT | 324 | 4 | 1.2% |
| BTCSTUSDT | 685 | 3 | 0.4% |
| BNXUSDT | 1231 | 5 | 0.4% |
| COCOSUSDT | 1378 | 3 | 0.2% |
| QUICKUSDT | 1418 | 3 | 0.2% |

## 6. Daily Return Distribution (pooled)

Total daily return observations: 602,648

| Quantile | Return |
|----------|--------|
| 0.1% | -31.41% |
| 1.0% | -15.78% |
| 5.0% | -9.10% |
| 25.0% | -2.97% |
| 50.0% | -0.06% |
| 75.0% | +2.69% |
| 95.0% | +9.45% |
| 99.0% | +20.36% |
| 99.9% | +52.69% |

Annualization convention: 365 days/year

Median asset annualized vol: 130.4%

## 7. Extreme Daily Returns (|r| > 50%)

Total: 783 events (0.130% of observations)

| Year | Extreme events |
|------|----------------|
| 2018 | 5 |
| 2019 | 4 |
| 2020 | 81 |
| 2021 | 232 |
| 2022 | 145 |
| 2023 | 109 |
| 2024 | 132 |
| 2025 | 75 |

⚠ marks years with more than max(3× median, 5) extreme events; current threshold: >285 events.

## 8. Volume Distribution

Volume type: `quote_volume_usd`

| Statistic | Value |
|-----------|-------|
| Median daily total volume | 5,274,851,208 |
| Median per-asset daily volume | 3,141,694 |
| Top-decile asset median volume | 21,424,990 |
| Bottom-decile asset median volume | 878,867 |
| Concentration: top-10 assets' share of total | 43.7% |

## 9. Presence Matrix Coverage

| Metric | Value |
|--------|-------|
| Presence rule | n/a |
| Liquidity lag | 1 day |
| Close observations allowed by presence | 111,450 / 603,192 (18.5%) |

## 10. Market Proxy Coverage

| Metric | Value |
|--------|-------|
| Valid observations | 2,737 / 2,738 (100.0%) |
| Annualization convention | 365 days/year |
| Annualized mean | 50.71% |
| Annualized vol | 67.22% |

## 11. Investable Universe Diagnostics

| Metric | Value |
|--------|-------|
| Historical price assets | 537 |
| Mean investable assets | 40.7 |
| Median investable assets | 50 |
| 10th percentile universe size | 17 |
| Minimum universe size | 0 |
| Maximum universe size | 50 |
| Days with <10 investable assets | 154 |
| Days with <20 investable assets | 417 |
