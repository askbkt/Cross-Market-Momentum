# Data Quality Report: MOEX

Generated: 2026-07-14 04:16
Source: MOEX ISS API
Universe: Historical TQBR (including delisted)

**Known caveats:**

- MOEX ISS API may return unadjusted prices — dividend gaps not corrected
- Small universe (<50 liquid stocks) limits quantile-based sorting
- 2022 sanctions / structural break — treat as regime event, not outlier

## 1. Market Coverage

| Metric | Value |
|--------|-------|
| First date | 2013-03-25 |
| Last date | 2024-12-30 |
| Trading days | 2,974 |
| Calendar years | 12 |
| Total assets (ever) | 435 |
| Avg trading days/year | 248 |

## 2. Assets Available Per Year

| Year | Mean | Min | Max |
|------|------|-----|-----|
| 2013 | 6 | 0 | 20 |
| 2014 | 28 | 19 | 46 |
| 2015 | 48 | 45 | 51 |
| 2016 | 48 | 42 | 55 |
| 2017 | 58 | 54 | 61 |
| 2018 | 59 | 56 | 63 |
| 2019 | 55 | 52 | 58 |
| 2020 | 68 | 57 | 74 |
| 2021 | 81 | 71 | 90 |
| 2022 | 78 | 0 | 94 |
| 2023 | 124 | 81 | 151 |
| 2024 | 127 | 104 | 140 |

## 3. Asset Lifetimes

### Calendar lifetime

| Statistic | Calendar days | Years |
|-----------|---------------|-------|
| Median lifetime | 3109 | 8.5 |
| Mean lifetime | 2455 | 6.7 |
| Min lifetime | 8 | 0.0 |
| Max lifetime | 4298 | 11.8 |

### Valid price observations

| Statistic | Observations |
|-----------|--------------|
| Median valid observations | 1705 |
| Mean valid observations | 1508 |
| Min valid observations | 3 |
| Max valid observations | 2956 |

Assets with <126 valid observations: 50 (11.5%)

Assets with <252 valid observations: 78 (17.9%)

## 4. Listings / Historical Exits by Year

An asset is classified as a historical exit when its last price observation is more than 30 calendar days before the panel end. This is an inactivity heuristic, not necessarily a formal exchange delisting event.

| Year | New listings | Historical exits |
|------|--------------|------------------|
| 2013 | 54 | 1 |
| 2014 | 291 | 40 |
| 2015 | 12 | 19 |
| 2016 | 6 | 16 |
| 2017 | 9 | 19 |
| 2018 | 3 | 13 |
| 2019 | 2 | 9 |
| 2020 | 12 | 11 |
| 2021 | 15 | 10 |
| 2022 | 2 | 18 |
| 2023 | 11 | 10 |
| 2024 | 18 | 8 |

**Historical exits before panel end: 174**
**Assets still observed near panel end: 261**

Historical coverage check: 40.0% of assets have no observations within the final 30 calendar days of the panel. The presence of inactive historical assets indicates that the universe is not restricted to end-of-sample survivors.

## 5. Missingness (within active lifetime)

| Statistic | Value |
|-----------|-------|
| Assets with zero gaps | 68 (15.6%) |
| Median missing % | 3.23% |
| Mean missing % | 12.67% |
| Worst asset missing % | 91.04% |

Worst 5 assets by gap share:

| Asset | Active days | Missing | % |
|-------|-------------|---------|---|
| MUGS | 1149 | 1046 | 91.0% |
| SKYC | 687 | 586 | 85.3% |
| GAZC | 2671 | 2255 | 84.4% |
| IDVP | 1280 | 1078 | 84.2% |
| GAZS | 2671 | 2218 | 83.0% |

## 6. Daily Return Distribution (pooled)

Total daily return observations: 630,579

| Quantile | Return |
|----------|--------|
| 0.1% | -18.71% |
| 1.0% | -8.07% |
| 5.0% | -3.74% |
| 25.0% | -1.01% |
| 50.0% | +0.00% |
| 75.0% | +0.90% |
| 95.0% | +4.16% |
| 99.0% | +11.06% |
| 99.9% | +39.67% |

Annualization convention: 252 days/year

Median asset annualized vol: 47.8%

## 7. Extreme Daily Returns (|r| > 30%)

Total: 1,397 events (0.222% of observations)

| Year | Extreme events |
|------|----------------|
| 2013 | 6 |
| 2014 | 78 |
| 2015 | 194 |
| 2016 | 191 |
| 2017 | 140 |
| 2018 | 92 |
| 2019 | 168 |
| 2020 | 117 |
| 2021 | 119 |
| 2022 | 104 |
| 2023 | 184 |
| 2024 | 4 |

⚠ marks years with more than max(3× median, 5) extreme events; current threshold: >354 events.

## 8. Volume Distribution

Volume type: `traded_value_rub`

| Statistic | Value |
|-----------|-------|
| Median daily total volume | 41,064,936,648 |
| Median per-asset daily volume | 743,682 |
| Top-decile asset median volume | 216,366,604 |
| Bottom-decile asset median volume | 43,812 |
| Concentration: top-10 assets' share of total | 67.0% |

## 9. Presence Matrix Coverage

| Metric | Value |
|--------|-------|
| Presence rule | traded >= 50% of last 126 days |
| Liquidity lag | 1 day |
| Close observations allowed by presence | 197,308 / 655,865 (30.1%) |

## 10. Market Proxy Coverage

| Metric | Value |
|--------|-------|
| Valid observations | 2,957 / 2,974 (99.4%) |
| Annualization convention | 252 days/year |
| Annualized mean | 8.71% |
| Annualized vol | 23.12% |

## 11. Investable Universe Diagnostics

| Metric | Value |
|--------|-------|
| Historical price assets | 435 |
| Mean investable assets | 66.3 |
| Median investable assets | 58 |
| 10th percentile universe size | 21 |
| Minimum universe size | 0 |
| Maximum universe size | 151 |
| Days with <10 investable assets | 162 |
| Days with <20 investable assets | 223 |
