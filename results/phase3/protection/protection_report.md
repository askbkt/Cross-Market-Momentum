# Phase 3 v2 — Frozen Regime-Aware Protection

## Design

- The base series is the corrected frozen Phase 3 construction for each market.
- When no stable ensemble exists, the report explicitly labels the base as a stable single or unstable validation reference.
- Portfolio and market volatility use a 63-observation rolling standard deviation shifted by one observation.
- q90 thresholds are calibrated only on pre-holdout history and mechanically applied to the 2023–2024 retrospective holdout.
- No protection winner is selected from holdout results; all frozen variants remain in the comparison.
- Base returns already include primary strategy transaction costs. Switching costs are incremental and use 0.5 × absolute exposure change.

## Base constructions

- US: `stable_single`; stable=True; members=['us_lb6M_sk1M_q10']
- MOEX: `stable_ensemble`; stable=True; members=['moex_lb6M_sk1M_q30', 'moex_lb6M_sk3M_q20', 'moex_lb6M_sk1M_q20']
- CRYPTO: `stable_ensemble`; stable=True; members=['crypto_lb6M_sk1M_q30', 'crypto_lb12M_sk3M_q20']

## Frozen thresholds

| market   | threshold         |   value_daily_vol |   annualized_value |   calibration_quantile |   rolling_window |
|:---------|:------------------|------------------:|-------------------:|-----------------------:|-----------------:|
| crypto   | market_vol_q90    |          0.048758 |           0.931530 |               0.900000 |               63 |
| crypto   | portfolio_vol_q90 |          0.020416 |           0.390053 |               0.900000 |               63 |
| moex     | market_vol_q90    |          0.018721 |           0.297180 |               0.900000 |               63 |
| moex     | portfolio_vol_q90 |          0.006285 |           0.099773 |               0.900000 |               63 |
| us       | market_vol_q90    |          0.015819 |           0.251117 |               0.900000 |               63 |
| us       | portfolio_vol_q90 |          0.011094 |           0.176115 |               0.900000 |               63 |

## Retrospective holdout comparison

| market   | base_strategy_type   | base_strategy_is_stable   | variant           |   protected_annualized_return |   protected_sharpe |   protected_max_drawdown |   protected_calmar |   protected_expected_shortfall_1pct |   mean_exposure |   risk_off_share |   n_switches |   switching_tc_drag_arithmetic |   delta_annualized_return |   delta_sharpe |   delta_max_drawdown |   delta_calmar |
|:---------|:---------------------|:--------------------------|:------------------|------------------------------:|-------------------:|-------------------------:|-------------------:|------------------------------------:|----------------:|-----------------:|-------------:|-------------------------------:|--------------------------:|---------------:|---------------------:|---------------:|
| crypto   | stable_ensemble      | True                      | combo_and_q90     |                       -0.1375 |            -0.6207 |                  -0.2915 |            -0.4717 |                             -0.0429 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| crypto   | stable_ensemble      | True                      | market_vol_q90    |                       -0.1375 |            -0.6207 |                  -0.2915 |            -0.4717 |                             -0.0429 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| crypto   | stable_ensemble      | True                      | no_protection     |                       -0.1375 |            -0.6207 |                  -0.2915 |            -0.4717 |                             -0.0429 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| crypto   | stable_ensemble      | True                      | portfolio_vol_q90 |                       -0.1375 |            -0.6207 |                  -0.2915 |            -0.4717 |                             -0.0429 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| moex     | stable_ensemble      | True                      | combo_and_q90     |                        0.0186 |             0.2647 |                  -0.1028 |             0.1813 |                             -0.0198 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| moex     | stable_ensemble      | True                      | market_vol_q90    |                        0.0186 |             0.2647 |                  -0.1028 |             0.1813 |                             -0.0198 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| moex     | stable_ensemble      | True                      | no_protection     |                        0.0186 |             0.2647 |                  -0.1028 |             0.1813 |                             -0.0198 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| moex     | stable_ensemble      | True                      | portfolio_vol_q90 |                       -0.0056 |            -0.0455 |                  -0.1115 |            -0.0500 |                             -0.0195 |          0.8059 |           0.1941 |            4 |                         0.0040 |                   -0.0242 |        -0.3102 |              -0.0087 |        -0.2313 |
| us       | stable_single        | True                      | combo_and_q90     |                       -0.0594 |            -0.5767 |                  -0.1388 |            -0.4282 |                             -0.0175 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| us       | stable_single        | True                      | market_vol_q90    |                       -0.0600 |            -0.5827 |                  -0.1388 |            -0.4325 |                             -0.0175 |          1.0000 |           0.0000 |            1 |                         0.0013 |                   -0.0006 |        -0.0060 |              -0.0000 |        -0.0043 |
| us       | stable_single        | True                      | no_protection     |                       -0.0594 |            -0.5767 |                  -0.1388 |            -0.4282 |                             -0.0175 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |
| us       | stable_single        | True                      | portfolio_vol_q90 |                       -0.0594 |            -0.5767 |                  -0.1388 |            -0.4282 |                             -0.0175 |          1.0000 |           0.0000 |            0 |                         0.0000 |                    0.0000 |         0.0000 |               0.0000 |         0.0000 |

## Interpretation guardrails

- Protection is evaluated as a risk-management mechanism, not as an alpha-search loop.
- A rule may be useful if it reduces drawdown or tail loss at an explicitly reported cost in exposure, switching and missed upside.
- A rule that does not fire, fires too often, or worsens results is a valid negative finding.
- Historical crisis-window tables are validation diagnostics and are not independent holdout evidence.
- No new threshold, window or logical variant may be introduced because of these holdout results.
