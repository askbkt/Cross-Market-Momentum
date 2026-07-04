# Definition of Done: Cross-Market Momentum & Regime Protection Study

## 1. Цель работы

Цель двухнедельной практики — адаптировать существующий Enhanced Momentum framework для сравнительного исследования momentum premium и regime-aware crash protection на разных рынках:

- US equities — исходный рынок/framework baseline;
- MOEX equities — российский рынок акций;
- Crypto — Binance spot universe, топ ликвидных монет без стейблкоинов.

Работа не ставит целью создание полноценной quant research platform. Основной результат практики — воспроизводимый сравнительный анализ:

1. Где momentum premium воспроизводится, а где нет.
2. Насколько устойчивы momentum-конфигурации на разных рынках.
3. Как regime-aware crash protection ведёт себя на US, MOEX и crypto.
4. Насколько выводы зависят от transaction costs, universe construction и рыночных режимов.

Отрицательный результат считается валидным результатом, если эксперимент проведён честно: без look-ahead bias, без OOS leakage и с явно описанными ограничениями данных.

---

## 2. Scope проекта

### In scope

В рамках практики должно быть сделано:

- перенос существующего проекта в новый репозиторий;
- выделение data loaders из исследовательской логики;
- единый интерфейс загрузки данных `load() -> DataFrame`;
- market-specific конфиги для US, MOEX и crypto;
- единая структура сохранения результатов;
- загрузка и подготовка данных для MOEX и Binance;
- построение price/return panel и presence matrix;
- запуск baseline momentum grid;
- walk-forward / OOS validation;
- rank-stability analysis;
- ensemble selection;
- regime-aware protection analysis;
- сравнительный markdown-отчёт по рынкам.

### Out of scope

В рамках текущей двухнедельной практики не требуется:

- строить полноценную quant research platform;
- делать production-grade orchestration;
- подключать live trading;
- делать real-time data ingestion;
- строить UI/dashboard;
- реализовывать полноценный historical market-cap universe для crypto, если это не укладывается в срок;
- идеально восстанавливать исторический состав IMOEX, если для MVP используется liquidity-based dynamic universe.

---

## 3. Архитектурный DoD

Архитектурная часть считается выполненной, если:

- существующий проект перенесён в новый репозиторий на ветку `main`;
- загрузчики данных вынесены из основной backtest/research логики;
- для всех рынков используется единый интерфейс:

```python
load(config) -> pd.DataFrame
```

- market-specific параметры вынесены в директорию `markets/`:

```text
markets/
  us.yaml
  moex.yaml
  crypto.yaml
```

- результаты сохраняются в market-specific директории:

```text
results/
  us/
  moex/
  crypto/
```

- исследовательская логика не зависит напрямую от конкретного источника данных;
- US, MOEX и crypto могут запускаться через один общий pipeline с разными конфигами;
- структура проекта допускает дальнейшее расширение, но текущая реализация остаётся сфокусированной на сравнительном исследовании.

---

## 4. Data Layer DoD

### 4.1. Единый data contract

Для каждого рынка должен быть приведён единый формат данных.

Минимально необходимый формат:

```text
date | asset | open | high | low | close | volume | value | return | presence
```

Либо wide-format, если его ожидает существующий framework:

```text
close_df: date x asset
returns_df: date x asset
volume_df: date x asset
presence_matrix: date x asset
market_proxy_returns: date x 1
```

Data layer считается готовым, если для каждого рынка есть:

- close prices;
- returns;
- volume / value proxy;
- presence matrix;
- market proxy;
- metadata по активам;
- sanity checks.

---

### 4.2. US data

US market используется как исходный baseline existing framework.

DoD:

- существующие US данные читаются новым loader interface;
- результаты старого pipeline воспроизводятся в новой структуре;
- US результаты сохраняются в `results/us/`;
- US используется как reference point для сравнения MOEX и crypto.

---

### 4.3. MOEX data

MOEX data layer считается выполненным, если:

- реализован loader для MOEX ISS API;
- загружаются дневные OHLCV по акциям;
- выбран universe construction rule;
- построена dynamic presence matrix;
- рассчитаны daily returns;
- добавлен market proxy, например IMOEX;
- сохранены parquet-файлы;
- проведены sanity checks.

Минимальный набор артефактов:

```text
data/moex/close.parquet
data/moex/returns.parquet
data/moex/volume.parquet
data/moex/presence_matrix.parquet
data/moex/market_proxy.parquet
data/moex/metadata.parquet
```

Обязательные проверки:

- число активов по годам;
- доля missing values;
- coverage по каждому активу;
- экстремальные дневные доходности;
- ликвидность;
- наличие торговых разрывов;
- влияние 2022 года как structural break.

Ограничения должны быть явно описаны:

- используются adjusted или unadjusted prices;
- как учитываются дивиденды и сплиты;
- используется ли текущий universe или динамический liquidity-based universe;
- какие тикеры исключены и почему.

---

### 4.4. Crypto data

Crypto data layer считается выполненным, если:

- реализован loader для Binance daily OHLCV;
- universe строится по Binance-tradable spot pairs;
- исключены стейблкоины и нежелательные synthetic/leveraged assets;
- построена dynamic presence matrix;
- BTCUSDT используется как market proxy;
- рассчитаны returns;
- сохранены parquet-файлы;
- проведены sanity checks.

Минимальный набор артефактов:

```text
data/crypto/close.parquet
data/crypto/returns.parquet
data/crypto/volume.parquet
data/crypto/presence_matrix.parquet
data/crypto/market_proxy.parquet
data/crypto/metadata.parquet
```

Обязательные проверки:

- дата первого появления каждой монеты;
- число доступных активов по годам;
- missing candles;
- outlier returns;
- volume coverage;
- survivorship/listing bias caveats.

---

## 5. Backtest DoD

Baseline momentum считается выполненным, если:

- momentum grid запущен отдельно для US, MOEX и crypto;
- для каждого рынка сохранены config-level results;
- результаты посчитаны gross и net of transaction costs;
- рассчитан turnover;
- рассчитаны основные risk/return метрики;
- есть сравнение с market benchmark;
- OOS период не используется для выбора конфигураций;
- результаты сохраняются в соответствующие директории `results/{market}/`.

Минимальные метрики:

- total return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio, если реализован;
- max drawdown;
- Calmar ratio;
- turnover;
- exposure;
- hit rate;
- transaction costs impact.

---

## 6. Grid Search DoD

Grid считается готовым, если:

- определён компактный grid для каждого рынка;
- grid не подгоняется по OOS;
- сохранены все конфиги и результаты;
- есть таблица ranked configs;
- есть анализ чувствительности к transaction costs.

Пример компактного grid:

```text
MOEX:
  window_days: [126, 252, 365]
  exclude_last_days: [21, 30]
  quantile: [0.2, 0.3, 0.4]
  rebalancing: monthly

Crypto:
  window_days: [30, 90, 180]
  exclude_last_days: [7, 14]
  quantile: [0.2, 0.3, 0.4]
  rebalancing: weekly/monthly
```

---

## 7. Walk-Forward / OOS DoD

Validation считается выполненной, если:

- для каждого рынка заранее определены train/validation/OOS splits;
- для MOEX используется 3–4 walk-forward folds, если хватает истории;
- для crypto используется 2–3 folds из-за меньшей истории;
- config selection делается только по IS/validation;
- OOS используется только для финальной оценки;
- fold-level results сохранены;
- есть summary по стабильности конфигураций.

---

## 8. Rank-Stability & Ensemble DoD

Rank-stability analysis считается выполненным, если:

- для каждого рынка рассчитана стабильность рангов конфигураций;
- определены stable finalists;
- ensemble строится только из конфигураций, выбранных до OOS;
- если стабильных конфигураций мало, это явно фиксируется;
- ensemble сравнивается с лучшей single-config стратегией и benchmark.

Правило выбора ensemble:

```text
if stable_configs >= 3:
    build ens-3
elif stable_configs == 2:
    build ens-2
else:
    report no stable ensemble found
```

---

## 9. Regime-Aware Protection DoD

Regime-aware protection считается выполненной, если:

- thresholds калибруются только на IS;
- thresholds frozen before OOS;
- OOS protection применяется механически;
- сравниваются no-protection и protection варианты;
- protection тестируется отдельно для US, MOEX и crypto;
- market proxy задан отдельно для каждого рынка:
  - US: исходный benchmark/market proxy;
  - MOEX: IMOEX или другой выбранный proxy;
  - Crypto: BTCUSDT.

Минимальные варианты protection:

```text
No protection
Portfolio volatility protection
Market volatility protection
Combo AND
```

Желательный диагностический вариант:

```text
Combo OR
```

Метрики protection:

- max drawdown reduction;
- Sharpe change;
- Calmar change;
- exposure reduction;
- missed upside;
- turnover impact;
- performance during crisis windows.

---

## 10. Comparative Analysis DoD

Сравнительный анализ считается выполненным, если итоговый отчёт отвечает на вопросы:

1. Работает ли momentum на US baseline?
2. Работает ли momentum на MOEX?
3. Работает ли momentum на crypto?
4. Где momentum сильнее gross?
5. Где momentum сохраняется net of transaction costs?
6. Где rank-stability помогает выбрать устойчивые конфигурации?
7. Где ensemble улучшает результат?
8. Где crash protection снижает drawdown?
9. Где protection ухудшает return из-за missed upside?
10. Чем объясняются различия между рынками?

---

## 11. Final Report DoD

Финальный отчёт считается готовым, если содержит:

```text
1. Research Question
2. Data Sources
3. Universe Construction
4. Data Quality & Limitations
5. Methodology
6. Baseline Momentum Results
7. Walk-Forward Validation
8. Rank-Stability & Ensemble
9. Regime-Aware Protection
10. Cross-Market Comparison
11. Limitations
12. Conclusion
```

Обязательные выводы:

- где momentum premium найден;
- где momentum premium не найден;
- где результат нестабилен;
- где crash protection полезна;
- где crash protection вредит;
- какие ограничения мешают сильным выводам;
- что можно улучшить после практики.

---

## 12. Итоговый критерий готовности

Практика считается успешно выполненной, если есть воспроизводимый pipeline, который:

1. Загружает данные US, MOEX и crypto через единый интерфейс.
2. Строит единые panel data и presence matrix.
3. Запускает baseline momentum на каждом рынке.
4. Выполняет walk-forward/rank-stability validation.
5. Строит frozen ensemble.
6. Тестирует regime-aware crash protection.
7. Сохраняет результаты в market-specific директории.
8. Даёт честный сравнительный вывод: где momentum работает, где нет, и как protection ведёт себя на разных рынках.
