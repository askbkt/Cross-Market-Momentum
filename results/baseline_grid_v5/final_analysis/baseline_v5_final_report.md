# Baseline Momentum Grid v5 — Final Analysis

## Validation status

- Runs: 81.
- Markets: US, MOEX, Binance spot crypto.
- Portfolio accounting: fixed-notional long and short sleeves.
- Gross exposure: 1.0; target net exposure: 0.0.
- Runs with a zero-target rebalance: 3.

## Market overview

- **US**: 18/27 configurations with positive Sharpe; mean Sharpe 0.118; best configuration `us_lb6M_sk3M_q10` with Sharpe 0.489, annualized return 4.50%, maximum drawdown -42.82%.
- **MOEX**: 22/27 configurations with positive Sharpe; mean Sharpe 0.511; best configuration `moex_lb6M_sk0M_q30` with Sharpe 1.050, annualized return 7.96%, maximum drawdown -20.02%.
- **CRYPTO**: 15/27 configurations with positive Sharpe; mean Sharpe -0.002; best configuration `crypto_lb6M_sk1M_q20` with Sharpe 0.573, annualized return 13.90%, maximum drawdown -31.13%.

## Lookback effect

- **US** mean Sharpe — 6M: 0.350, 12M: 0.202, 24M: -0.196.
- **MOEX** mean Sharpe — 6M: 0.844, 12M: 0.720, 24M: -0.030.
- **CRYPTO** mean Sharpe — 6M: 0.216, 12M: 0.065, 24M: -0.288.

## Most robust clean cross-market configuration

`6M / 1M / q=20%`

- Mean Sharpe: 0.604.
- Worst-market Sharpe: 0.368.
- Mean annualized return: 8.28%.
- Mean maximum drawdown: -29.67%.

## MOEX data caveat

- Affected configurations: `moex_lb12M_sk0M_q10`, `moex_lb12M_sk0M_q20`, `moex_lb12M_sk0M_q30`.
- Affected effective rebalance date(s): 2023-02-01.
- Cause: the 12-month, zero-skip signal starts on 2022-01-07, a sparse MOEX observation with only 12 market-wide closes versus roughly 257–261 on adjacent sessions. None of the 90 assets present at the 2023-01-31 decision date had a comparable start price, so the portfolio remained in cash until the next scheduled rebalance.
- Treatment: retain the raw runs, flag them in all tables, and do not use them as primary evidence for cross-market robustness.

## Main interpretation

The baseline supports a cross-market medium-term momentum effect concentrated around the 6-month lookback. The 12-month horizon remains positive on US and MOEX but is weaker and less reliable on crypto. The 24-month horizon deteriorates across all three markets and is negative on average.

The strongest clean robustness region is the 6-month lookback with a 1-month skip and a 20–30% quantile. These configurations remain positive across all three markets and are not affected by the MOEX sparse-session caveat.
