# Roadmap: Cross-Market Momentum & Regime Protection Study

## Overview

```
Phase 1   Days 1-4    Architecture + Data
Phase 2   Days 5-7    Baseline Grids
Phase 3   Days 8-11   Validation + Protection
Phase 4   Days 12-14  Comparison + Report
```

---

## Phase 1: Architecture + Data (days 1-4)

### Goal

Единый multi-market pipeline: три рынка загружаются через один интерфейс, данные проверены и готовы к backtest.

### Tasks

**Architecture (days 1-2)**

- Перенести проект в новый репозиторий.
- Вынести data loading из backtest/research логики в `data_loaders/`.
- Единый интерфейс: `loader.load(config) -> (close_df, returns_df, volume_df, presence_matrix, market_proxy)`.
- Market-specific конфиги в `markets/{us,moex,crypto}.yaml`.
- Структура результатов: `results/{us,moex,crypto}/`.
- Проверить, что US baseline воспроизводится в новой структуре.

**MOEX data (day 2-3)**

- `moex_loader.py`: MOEX ISS API --> daily OHLCV.
- Universe: TQBR liquid stocks, dynamic presence matrix.
- Market proxy: IMOEX index.
- Сохранить: `data/moex/{close,returns,volume,presence_matrix,market_proxy}.parquet`.

**Crypto data (day 3-4)**

- `binance_loader.py`: Binance API --> daily OHLCV.
- Universe: top 30-50 spot pairs, без стейблкоинов и leveraged tokens.
- Dynamic presence matrix по listing dates.
- Market proxy: BTCUSDT.
- Сохранить: `data/crypto/{close,returns,volume,presence_matrix,market_proxy}.parquet`.

**Sanity checks (параллельно с загрузкой)**

- Число активов по годам, missing values, extreme returns, volume distribution.
- MOEX-specific: дивидендные гэпы, trading breaks, 2022 structural break, adjusted/unadjusted caveat.
- Crypto-specific: listing/delisting dates, pump/dump outliers, survivorship bias.
- Data quality reports: `reports/data_quality/{us,moex,crypto}_data_quality.md`.

### Deliverables

```
data_loaders/{base,us_loader,moex_loader,binance_loader}.py
markets/{us,moex,crypto}.yaml
data/{us,moex,crypto}/*.parquet
reports/data_quality/*.md
```

### Exit criterion

US, MOEX и crypto данные загружаются через единый интерфейс. Sanity checks пройдены. Все data caveats задокументированы.

### Buffer

1 день. Если API работают без проблем, Phase 1 заканчивается за 3 дня --> Phase 2 начинается раньше.

### Risks

- MOEX ISS API: rate limits, неполные данные по старым тикерам. Mitigation: кэшировать aggressively, fallback на CSV если API нестабилен.
- Binance API: может потребоваться пагинация для длинной истории. Mitigation: загружать по 1000 candles за запрос, retry logic.
- Adjusted prices на MOEX: ISS может не давать adjusted closes. Mitigation: явно зафиксировать caveat, использовать unadjusted с оговоркой.

---

## Phase 2: Baseline Momentum Grids (days 5-7)

### Goal

Baseline momentum results для всех трёх рынков. Первичный ответ: есть ли momentum premium gross и net of TC.

### Tasks

**MOEX grid (day 5)**

```yaml
window_days: [126, 252, 365]
exclude_last_days: [21, 30]
quantile: [0.2, 0.3, 0.4]
rebalancing: monthly
tc_bps: [10, 20]
```

- ~18-27 configs * gross/net = manageable grid.
- Сравнение с IMOEX benchmark.
- Config-level metrics: Sharpe, IR, max DD, turnover, NAV.

**Crypto grid (day 6)**

```yaml
window_days: [30, 90, 180]
exclude_last_days: [7, 14]
quantile: [0.2, 0.3, 0.4]
rebalancing: monthly  # weekly as stretch goal
tc_bps: [5, 10, 20]
```

- Crypto-specific: shorter lookbacks (30d), lower TC (5 bps Binance spot).
- Сравнение с BTC benchmark.

**US reference (day 5-6, параллельно)**

- Прогнать US grid в новой структуре.
- Сверить с результатами курсовой, должны совпадать.

**Preliminary analysis (day 7)**

- Для каждого рынка: ranked configs, gross vs net comparison, best/worst/median.
- Первичный вывод: momentum виден / не виден / нестабилен.
- `results/{market}/baseline_summary.md`.

### Deliverables

```
results/{us,moex,crypto}/grid_results.csv
results/{us,moex,crypto}/config_metrics.csv
results/{us,moex,crypto}/baseline_summary.md
```

### Exit criterion

Grid завершён для всех трёх рынков. Есть ranked configs. Есть gross vs net comparison. Предварительный вывод сформулирован.

### Buffer

0.5 дня. Grid runs are computational, не creative. Если данные чистые после Phase 1, это фаза с минимальным risk.

### Risks

- MOEX universe слишком маленький (<30 stocks) для quantile sorting. Mitigation: увеличить quantile до 0.4–0.5 или перейти на top/bottom N вместо quantile.
- Crypto: high turnover из-за volatile rankings. Mitigation: TC sensitivity analysis покажет, при каком TC momentum исчезает.

---

## Phase 3: Validation, Ensemble, Protection (days 8-11)

### Goal

Core research: walk-forward validation, rank-stability, ensemble, regime-aware protection. Для каждого рынка отдельно.

### Tasks

**Walk-forward validation (day 8)**

MOEX folds (expanding window, 3-4 folds):
```
Fold 1: train 2010–2015, test 2016–2017
Fold 2: train 2012–2017, test 2018–2019
Fold 3: train 2014–2019, test 2020–2021
Fold 4: train 2016–2021, test 2022–2024
```

Crypto folds (2-3 folds, limited history):
```
Fold 1: train 2018–2020, test 2021
Fold 2: train 2019–2021, test 2022
Fold 3: train 2020–2022, test 2023–2024/2025
```

US folds: reuse курсовой structure for reference.

**Rank-stability + finalists (day 9)**

- Rank-stability analysis per market.
- Combined score (Sharpe rank + IR rank) или single-metric rank если grid маленький.
- Finalist selection: ens-3 if ≥3 stable configs, ens-2 if 2, report "no stable ensemble" if <2.
- Finalists frozen before OOS.

**Ensemble + OOS test (day 9-10)**

- Build frozen ensemble per market.
- OOS evaluation: NAV, Sharpe, IR, max DD.
- Compare: ensemble vs best single vs benchmark.

**Regime-aware protection (day 10-11)**

IS calibration --> frozen thresholds → OOS apply. Per market:

| Signal | US proxy | MOEX proxy | Crypto proxy |
|--------|----------|------------|--------------|
| Internal vol | portfolio vol63 | portfolio vol63 | portfolio vol63 |
| External vol | SPX vol63 | IMOEX vol63 | BTC vol63 |

Minimum variants:
```
No protection
Portfolio vol binary (q90)
Market vol binary (q90)
Combo AND
```

Metrics: max DD reduction, Sharpe change, exposure %, switches, TC drag.

### Deliverables

```
results/{market}/walk_forward_results.csv
results/{market}/rank_stability.csv
results/{market}/finalists.yaml
results/{market}/ensemble_oos.csv
results/{market}/protection_thresholds.yaml
results/{market}/protection_oos.csv
results/{market}/protection_summary.md
```

### Exit criterion

Для каждого рынка: walk-forward done, finalists selected, ensemble built, protection tested OOS. Все thresholds frozen before OOS.

### Buffer

1 день. Это самая dense фаза. Если Phase 2 закончилась раньше, начинать Phase 3 сразу.

### Risks

- MOEX: слишком мало folds из-за короткой истории (2010–2024 = 14 лет, из которых 2022 - structural break). Mitigation: 3 folds minimum, acknowledge limitation.
- Crypto: 2-3 folds = low statistical power. Mitigation: explicit caveat, rely on directional findings not precise estimates.
- Rank-stability может не найти стабильных конфигов на MOEX/crypto. Mitigation: это валидный result — report it honestly.

---

## Phase 4: Cross-Market Comparison + Report (days 12-14)

### Goal

Сводный анализ, финальный отчёт, воспроизводимый проект.

### Tasks

**Cross-market comparison table (day 12)**

Единая таблица по всем рынкам:

| Metric | US | MOEX | Crypto |
|--------|-----|------|--------|
| Momentum premium (gross) | | | |
| Momentum premium (net TC) | | | |
| Rank-stability (ρ IS→OOS) | | | |
| Ensemble NAV (OOS) | | | |
| Protection: max DD reduction | | | |
| Protection: Sharpe change | | | |
| Protection: missed upside | | | |

Причины различий: universe size, liquidity, TC, volatility regime, concentration, survivorship bias.

**Final report (day 13)**

Markdown report, структура:

```
1. Research Question
2. Data Sources & Universe Construction
3. Data Quality & Limitations
4. Methodology (brief — reference курсовой for details)
5. Baseline Momentum Results (per market)
6. Walk-Forward & Rank-Stability
7. Ensemble Construction
8. Regime-Aware Protection
9. Cross-Market Comparison
10. Limitations & Caveats
11. Conclusions
12. Future Work
```

**Cleanup (day 14)**

- README с инструкцией запуска.
- Проверить воспроизводимость: `git clone` -> `pip install` -> `python scripts/build_dataset.py` -> `python scripts/run_grid.py --market moex`.
- Все configs сохранены. Все results traceable.

### Deliverables

```
results/cross_market_summary.csv
reports/cross_market_comparison.md
reports/final_report.md
README.md
```

### Exit criterion

Финальный отчёт отвечает на 10 вопросов из DoD Section 10. Проект воспроизводим с README. Все findings, включая отрицательные, честно задокументированы.

### Buffer

Встроен: если Phase 3 затянулась на день, Phase 4 сжимается до 2 дней (comparison + report в один день, cleanup параллельно). Отчёт можно начинать писать ещё в Phase 3 по мере получения результатов.

---

## Risk Summary

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API downtime/rate limits | Medium | Delays Phase 1 | Cache aggressively, fallback to CSV |
| MOEX universe too small | Medium | Weak momentum signal | Wider quantiles, top/bottom N |
| No stable configs on MOEX/crypto | Medium | No ensemble | Report as valid negative finding |
| Phase 3 overruns | Medium | Less time for report | Start report writing in Phase 3 |
| Momentum doesn't work on new markets | Low-Medium | No "positive" result | Negative finding = valid finding per DoD |

---

## Success Criterion

Практика успешна, если pipeline загружает US/MOEX/crypto через единый интерфейс, прогоняет baseline -> walk-forward -> rank-stability -> ensemble -> protection для каждого рынка, и даёт честный сравнительный вывод в финальном отчёте.
