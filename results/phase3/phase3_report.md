# Phase 3 v2 — Train/Test Walk-Forward, Frozen Selection and Holdout

## Frozen protocol

- Source: frozen `baseline_grid_v5` daily returns and turnover; no baseline backtest was rerun.
- Primary selector: net Sharpe under US 25 bps, MOEX 20 bps and Crypto 10 bps.
- Every fold ranks configurations on train and measures rank transfer on the following test period.
- All eligible configurations share one common date index inside each fold/sample.
- Benchmark-relative IR is descriptive, not a selection objective, because the strategy is market-neutral and the proxy is long-only.
- The final 2023–2024 sample is a chronological retrospective holdout, not a fully pristine OOS sample.

## Fold-level train-to-test stability

| market   | fold      |   spearman_train_test_rho |   median_absolute_rank_change |   train_top_fraction_retention_share |   train_top3_to_test_top_fraction_share | train_winner_run_id   |   train_winner_test_rank |   train_winner_test_net_sharpe |
|:---------|:----------|--------------------------:|------------------------------:|-------------------------------------:|----------------------------------------:|:----------------------|-------------------------:|-------------------------------:|
| crypto   | crypto_f1 |                   -0.4096 |                       10.0000 |                               0.1111 |                                  0.0000 | crypto_lb24M_sk3M_q20 |                  23.0000 |                         0.1059 |
| crypto   | crypto_f2 |                   -0.1154 |                        9.0000 |                               0.1111 |                                  0.3333 | crypto_lb6M_sk1M_q20  |                   7.0000 |                         0.4061 |
| moex     | moex_f1   |                    0.2974 |                        7.0000 |                               0.5000 |                                  0.6667 | moex_lb6M_sk0M_q30    |                   8.0000 |                         0.7852 |
| moex     | moex_f2   |                    0.3948 |                        3.5000 |                               0.5000 |                                  0.3333 | moex_lb6M_sk0M_q30    |                  15.0000 |                        -0.0033 |
| moex     | moex_f3   |                    0.8904 |                        1.0000 |                               0.6250 |                                  0.6667 | moex_lb6M_sk1M_q30    |                   1.0000 |                         4.6457 |
| moex     | moex_f4   |                    0.0157 |                        7.0000 |                               0.2500 |                                  0.0000 | moex_lb6M_sk1M_q30    |                  21.0000 |                        -0.2199 |
| us       | us_f1     |                   -0.2833 |                        8.0000 |                               0.3333 |                                  0.0000 | us_lb12M_sk0M_q10     |                  25.0000 |                         0.8705 |
| us       | us_f2     |                   -0.2460 |                       12.0000 |                               0.3333 |                                  0.0000 | us_lb12M_sk1M_q20     |                  16.0000 |                        -1.2327 |
| us       | us_f3     |                   -0.2778 |                       11.0000 |                               0.0000 |                                  0.0000 | us_lb12M_sk1M_q20     |                  12.0000 |                         0.4136 |
| us       | us_f4     |                    0.2540 |                        5.0000 |                               0.2222 |                                  0.0000 | us_lb12M_sk1M_q20     |                  15.0000 |                         0.6936 |
| us       | us_f5     |                   -0.4695 |                       12.0000 |                               0.2222 |                                  0.3333 | us_lb12M_sk1M_q20     |                  13.0000 |                        -0.2431 |
| us       | us_f6     |                    0.0940 |                        6.0000 |                               0.2222 |                                  0.0000 | us_lb12M_sk1M_q20     |                  13.0000 |                        -0.2395 |
| us       | us_f7     |                    0.6642 |                        4.0000 |                               0.4444 |                                  0.3333 | us_lb6M_sk1M_q10      |                   1.0000 |                         1.7684 |
| us       | us_f8     |                    0.4951 |                        3.0000 |                               0.3333 |                                  0.0000 | us_lb6M_sk1M_q10      |                  16.0000 |                         0.5392 |

## Frozen construction

### US
- Eligible configurations: 27
- Stable configurations: 1
- Best frozen single: `us_lb6M_sk1M_q10` (stable=True)
- Ensemble status: `no_stable_ensemble`
- Ensemble members: none
- Protection base: `stable_single` — ['us_lb6M_sk1M_q10']

### MOEX
- Eligible configurations: 24
- Stable configurations: 4
- Best frozen single: `moex_lb6M_sk1M_q30` (stable=True)
- Ensemble status: `ens3`
- Ensemble members: ['moex_lb6M_sk1M_q30', 'moex_lb6M_sk3M_q20', 'moex_lb6M_sk1M_q20']
- Protection base: `stable_ensemble` — ['moex_lb6M_sk1M_q30', 'moex_lb6M_sk3M_q20', 'moex_lb6M_sk1M_q20']

### CRYPTO
- Eligible configurations: 27
- Stable configurations: 2
- Best frozen single: `crypto_lb6M_sk1M_q30` (stable=True)
- Ensemble status: `ens2`
- Ensemble members: ['crypto_lb6M_sk1M_q30', 'crypto_lb12M_sk3M_q20']
- Protection base: `stable_ensemble` — ['crypto_lb6M_sk1M_q30', 'crypto_lb12M_sk3M_q20']

## Stable-config summary

| market   | run_id                | stable_config   |   median_train_rank |   median_test_rank |   test_top_fraction_share |   positive_test_net_sharpe_share |   median_absolute_rank_change |   mean_test_net_sharpe |
|:---------|:----------------------|:----------------|--------------------:|-------------------:|--------------------------:|---------------------------------:|------------------------------:|-----------------------:|
| crypto   | crypto_lb6M_sk1M_q30  | True            |              6.5000 |             7.5000 |                    0.5000 |                           1.0000 |                        9.0000 |                 0.8054 |
| crypto   | crypto_lb12M_sk3M_q20 | True            |              5.5000 |             9.0000 |                    0.5000 |                           1.0000 |                        3.5000 |                 0.5842 |
| crypto   | crypto_lb6M_sk3M_q30  | False           |             17.5000 |             3.5000 |                    1.0000 |                           1.0000 |                       14.0000 |                 1.4856 |
| crypto   | crypto_lb6M_sk1M_q20  | False           |              9.5000 |             4.0000 |                    1.0000 |                           1.0000 |                       11.5000 |                 1.3160 |
| crypto   | crypto_lb12M_sk0M_q10 | False           |             21.5000 |             8.5000 |                    0.5000 |                           1.0000 |                       13.0000 |                 0.7222 |
| crypto   | crypto_lb6M_sk1M_q10  | False           |             17.0000 |            10.0000 |                    0.0000 |                           1.0000 |                        7.0000 |                 0.5105 |
| crypto   | crypto_lb6M_sk3M_q20  | False           |             25.5000 |            11.0000 |                    0.5000 |                           1.0000 |                       14.5000 |                 0.6821 |
| crypto   | crypto_lb24M_sk3M_q10 | False           |             10.0000 |            12.0000 |                    0.5000 |                           1.0000 |                       18.0000 |                 0.6402 |
| crypto   | crypto_lb6M_sk3M_q10  | False           |             13.0000 |            12.0000 |                    0.5000 |                           1.0000 |                        9.0000 |                 0.5466 |
| crypto   | crypto_lb12M_sk0M_q30 | False           |             16.0000 |            12.0000 |                    0.0000 |                           0.5000 |                        4.0000 |                 0.4386 |
| crypto   | crypto_lb12M_sk1M_q20 | False           |             17.0000 |            12.0000 |                    0.5000 |                           0.5000 |                        8.0000 |                 0.4312 |
| crypto   | crypto_lb12M_sk3M_q10 | False           |             11.0000 |            12.0000 |                    0.5000 |                           1.0000 |                        5.0000 |                 0.3536 |
| crypto   | crypto_lb12M_sk1M_q30 | False           |             12.0000 |            12.5000 |                    0.5000 |                           0.5000 |                       14.5000 |                 0.5353 |
| crypto   | crypto_lb6M_sk0M_q20  | False           |             12.0000 |            12.5000 |                    0.5000 |                           0.5000 |                        7.5000 |                 0.4046 |
| crypto   | crypto_lb24M_sk1M_q20 | False           |             11.0000 |            13.0000 |                    0.5000 |                           0.5000 |                       14.0000 |                 0.4922 |
| crypto   | crypto_lb12M_sk1M_q10 | False           |             12.0000 |            14.0000 |                    0.5000 |                           0.5000 |                       21.0000 |                 0.6203 |
| crypto   | crypto_lb24M_sk3M_q20 | False           |              9.0000 |            14.0000 |                    0.5000 |                           1.0000 |                       17.0000 |                 0.3387 |
| crypto   | crypto_lb12M_sk3M_q30 | False           |              6.5000 |            14.0000 |                    0.0000 |                           0.5000 |                        7.5000 |                 0.2642 |
| crypto   | crypto_lb24M_sk0M_q10 | False           |             17.0000 |            17.0000 |                    0.5000 |                           0.5000 |                       18.0000 |                -0.2277 |
| crypto   | crypto_lb24M_sk1M_q10 | False           |             24.5000 |            17.5000 |                    0.0000 |                           0.5000 |                        7.0000 |                 0.1199 |
| crypto   | crypto_lb12M_sk0M_q20 | False           |             23.0000 |            17.5000 |                    0.0000 |                           0.5000 |                        5.5000 |                 0.0916 |
| crypto   | crypto_lb24M_sk0M_q20 | False           |             12.5000 |            19.0000 |                    0.0000 |                           0.0000 |                       13.5000 |                -0.0410 |
| crypto   | crypto_lb6M_sk0M_q30  | False           |              6.5000 |            19.5000 |                    0.0000 |                           0.5000 |                       13.0000 |                 0.0120 |
| crypto   | crypto_lb24M_sk3M_q30 | False           |             16.0000 |            21.5000 |                    0.0000 |                           0.5000 |                        5.5000 |                -0.0388 |
| crypto   | crypto_lb24M_sk1M_q30 | False           |             18.5000 |            23.0000 |                    0.0000 |                           0.0000 |                        8.5000 |                -0.3024 |
| crypto   | crypto_lb24M_sk0M_q30 | False           |              7.0000 |            24.0000 |                    0.0000 |                           0.5000 |                       17.0000 |                -0.2768 |
| crypto   | crypto_lb6M_sk0M_q10  | False           |             20.5000 |            25.5000 |                    0.0000 |                           0.0000 |                        8.0000 |                -0.5489 |
| moex     | moex_lb6M_sk1M_q30    | True            |              1.5000 |             3.5000 |                    0.7500 |                           0.7500 |                        2.0000 |                 1.5509 |
| moex     | moex_lb6M_sk3M_q20    | True            |              7.0000 |             5.0000 |                    1.0000 |                           1.0000 |                        2.5000 |                 1.4432 |
| moex     | moex_lb6M_sk1M_q20    | True            |              4.0000 |             7.5000 |                    0.7500 |                           0.7500 |                        3.0000 |                 1.4270 |
| moex     | moex_lb12M_sk1M_q20   | True            |              6.0000 |             7.5000 |                    0.5000 |                           1.0000 |                        3.0000 |                 1.0920 |
| moex     | moex_lb12M_sk3M_q20   | False           |             15.0000 |             7.0000 |                    0.5000 |                           1.0000 |                       10.5000 |                 1.0527 |
| moex     | moex_lb6M_sk3M_q30    | False           |              9.0000 |             7.5000 |                    0.5000 |                           1.0000 |                        7.0000 |                 1.3983 |
| moex     | moex_lb12M_sk1M_q30   | False           |              9.0000 |             7.5000 |                    0.5000 |                           1.0000 |                        5.0000 |                 1.1347 |
| moex     | moex_lb12M_sk1M_q10   | False           |              9.5000 |             8.0000 |                    0.5000 |                           1.0000 |                        5.5000 |                 1.1150 |
| moex     | moex_lb12M_sk3M_q10   | False           |             16.5000 |             8.0000 |                    0.5000 |                           1.0000 |                        7.0000 |                 1.0298 |
| moex     | moex_lb12M_sk3M_q30   | False           |             16.0000 |             8.5000 |                    0.5000 |                           1.0000 |                        7.0000 |                 0.8297 |
| moex     | moex_lb6M_sk3M_q10    | False           |             11.0000 |             9.0000 |                    0.5000 |                           1.0000 |                        3.0000 |                 1.1803 |
| moex     | moex_lb6M_sk0M_q10    | False           |             12.5000 |             9.0000 |                    0.5000 |                           0.7500 |                        6.5000 |                 1.0543 |
| moex     | moex_lb6M_sk1M_q10    | False           |              6.5000 |            10.0000 |                    0.2500 |                           0.7500 |                        4.0000 |                 0.9914 |
| moex     | moex_lb6M_sk0M_q30    | False           |              1.5000 |            11.5000 |                    0.5000 |                           0.5000 |                       10.5000 |                 1.1382 |
| moex     | moex_lb24M_sk0M_q20   | False           |             11.5000 |            11.5000 |                    0.0000 |                           1.0000 |                        3.0000 |                 0.4550 |
| moex     | moex_lb6M_sk0M_q20    | False           |              3.5000 |            16.5000 |                    0.2500 |                           0.7500 |                       13.5000 |                 1.0863 |
| moex     | moex_lb24M_sk0M_q30   | False           |             15.5000 |            16.5000 |                    0.0000 |                           0.7500 |                        5.0000 |                 0.1560 |
| moex     | moex_lb24M_sk1M_q20   | False           |             17.5000 |            17.0000 |                    0.0000 |                           0.5000 |                        0.5000 |                 0.1601 |
| moex     | moex_lb24M_sk3M_q20   | False           |             21.0000 |            18.0000 |                    0.0000 |                           0.5000 |                        4.0000 |                -0.0715 |
| moex     | moex_lb24M_sk3M_q10   | False           |             21.0000 |            19.5000 |                    0.0000 |                           0.5000 |                        1.0000 |                -0.0440 |
| moex     | moex_lb24M_sk0M_q10   | False           |             17.5000 |            20.0000 |                    0.0000 |                           0.5000 |                        4.0000 |                -0.0442 |
| moex     | moex_lb24M_sk1M_q30   | False           |             23.0000 |            21.0000 |                    0.0000 |                           0.5000 |                        2.0000 |                -0.1492 |
| moex     | moex_lb24M_sk1M_q10   | False           |             20.0000 |            21.5000 |                    0.0000 |                           0.0000 |                        4.0000 |                -0.1785 |
| moex     | moex_lb24M_sk3M_q30   | False           |             24.0000 |            22.5000 |                    0.0000 |                           0.5000 |                        1.5000 |                -0.2881 |
| moex     | moex_lb12M_sk0M_q30   | False           |            nan      |           nan      |                  nan      |                         nan      |                      nan      |                 1.1794 |
| moex     | moex_lb12M_sk0M_q20   | False           |            nan      |           nan      |                  nan      |                         nan      |                      nan      |                 1.0795 |
| moex     | moex_lb12M_sk0M_q10   | False           |            nan      |           nan      |                  nan      |                         nan      |                      nan      |                 0.9459 |
| us       | us_lb6M_sk1M_q10      | True            |              4.5000 |             6.5000 |                    0.5000 |                           0.7500 |                        6.0000 |                 0.5615 |
| us       | us_lb6M_sk0M_q20      | False           |             16.0000 |             6.0000 |                    0.6250 |                           0.6250 |                       10.5000 |                 0.3684 |
| us       | us_lb6M_sk1M_q20      | False           |             15.5000 |             7.5000 |                    0.5000 |                           0.7500 |                       13.0000 |                 0.5302 |
| us       | us_lb6M_sk1M_q30      | False           |             24.0000 |             8.5000 |                    0.6250 |                           0.7500 |                       14.0000 |                 0.4535 |
| us       | us_lb6M_sk3M_q30      | False           |             18.5000 |             9.0000 |                    0.5000 |                           0.7500 |                        9.0000 |                 0.4765 |
| us       | us_lb6M_sk0M_q30      | False           |             21.0000 |             9.0000 |                    0.5000 |                           0.6250 |                       11.0000 |                 0.3768 |
| us       | us_lb6M_sk3M_q10      | False           |             10.0000 |             9.5000 |                    0.5000 |                           0.6250 |                        5.0000 |                 0.5007 |
| us       | us_lb12M_sk3M_q20     | False           |              8.5000 |            10.5000 |                    0.3750 |                           0.5000 |                        5.0000 |                 0.1484 |
| us       | us_lb6M_sk3M_q20      | False           |             14.5000 |            11.5000 |                    0.2500 |                           0.6250 |                        7.5000 |                 0.4588 |
| us       | us_lb12M_sk1M_q20     | False           |              1.0000 |            12.5000 |                    0.1250 |                           0.6250 |                       11.5000 |                 0.2356 |
| us       | us_lb6M_sk0M_q10      | False           |              9.5000 |            13.0000 |                    0.3750 |                           0.6250 |                        6.0000 |                 0.3127 |
| us       | us_lb12M_sk0M_q10     | False           |              2.0000 |            13.0000 |                    0.2500 |                           0.6250 |                       11.0000 |                 0.2356 |
| us       | us_lb24M_sk1M_q20     | False           |             17.5000 |            13.5000 |                    0.3750 |                           0.3750 |                        8.5000 |                -0.1401 |
| us       | us_lb12M_sk0M_q30     | False           |              8.5000 |            14.0000 |                    0.3750 |                           0.6250 |                        6.5000 |                 0.2168 |
| us       | us_lb12M_sk3M_q30     | False           |              9.0000 |            14.0000 |                    0.3750 |                           0.5000 |                        7.0000 |                 0.1325 |
| us       | us_lb12M_sk1M_q30     | False           |              4.0000 |            14.5000 |                    0.3750 |                           0.6250 |                       11.0000 |                 0.2026 |
| us       | us_lb12M_sk0M_q20     | False           |              6.5000 |            15.0000 |                    0.1250 |                           0.6250 |                       10.0000 |                 0.2284 |
| us       | us_lb24M_sk1M_q30     | False           |             21.0000 |            15.5000 |                    0.3750 |                           0.3750 |                        9.5000 |                -0.1293 |
| us       | us_lb12M_sk3M_q10     | False           |             12.0000 |            16.0000 |                    0.3750 |                           0.5000 |                        7.0000 |                 0.0901 |
| us       | us_lb24M_sk0M_q30     | False           |             19.5000 |            16.0000 |                    0.1250 |                           0.3750 |                        2.5000 |                -0.1439 |
| us       | us_lb12M_sk1M_q10     | False           |              4.0000 |            17.0000 |                    0.1250 |                           0.6250 |                       13.0000 |                 0.2114 |
| us       | us_lb24M_sk3M_q10     | False           |             25.0000 |            17.0000 |                    0.2500 |                           0.3750 |                        8.0000 |                -0.2512 |
| us       | us_lb24M_sk1M_q10     | False           |             15.0000 |            18.5000 |                    0.2500 |                           0.3750 |                        5.5000 |                -0.1627 |
| us       | us_lb24M_sk0M_q20     | False           |             18.0000 |            19.5000 |                    0.1250 |                           0.3750 |                        5.5000 |                -0.1566 |
| us       | us_lb24M_sk0M_q10     | False           |             23.0000 |            21.5000 |                    0.1250 |                           0.2500 |                        5.0000 |                -0.2534 |
| us       | us_lb24M_sk3M_q30     | False           |             27.0000 |            21.5000 |                    0.2500 |                           0.2500 |                        5.0000 |                -0.2548 |
| us       | us_lb24M_sk3M_q20     | False           |             26.0000 |            22.0000 |                    0.2500 |                           0.2500 |                        4.0000 |                -0.2119 |

## Retrospective holdout comparison

| market   | portfolio_type     | portfolio_name                                           | construction_is_stable   |   net_annualized_return |   net_annualized_volatility |   net_sharpe |   net_max_drawdown |   net_information_ratio_vs_benchmark |   net_correlation_to_benchmark |   net_beta_to_benchmark |
|:---------|:-------------------|:---------------------------------------------------------|:-------------------------|------------------------:|----------------------------:|-------------:|-------------------:|-------------------------------------:|-------------------------------:|------------------------:|
| crypto   | benchmark          | BTCUSDT                                                  | True                     |                  1.3756 |                      0.4875 |       2.0175 |            -0.2615 |                               0.0000 |                         1.0000 |                  1.0000 |
| crypto   | best_frozen_single | crypto_lb6M_sk1M_q30                                     | True                     |                  0.0236 |                      0.2337 |       0.2162 |            -0.2342 |                              -1.7859 |                         0.0850 |                  0.0407 |
| crypto   | stable_ensemble    | crypto_lb6M_sk1M_q30+crypto_lb12M_sk3M_q20               | True                     |                 -0.1375 |                      0.2046 |      -0.6207 |            -0.2915 |                              -2.1331 |                         0.0424 |                  0.0178 |
| moex     | benchmark          | IMOEX                                                    | True                     |                  0.1549 |                      0.1793 |       0.8924 |            -0.3212 |                               0.0000 |                         1.0000 |                  1.0000 |
| moex     | best_frozen_single | moex_lb6M_sk1M_q30                                       | True                     |                  0.0122 |                      0.0757 |       0.1980 |            -0.0928 |                              -0.7087 |                        -0.1467 |                 -0.0620 |
| moex     | stable_ensemble    | moex_lb6M_sk1M_q30+moex_lb6M_sk3M_q20+moex_lb6M_sk1M_q20 | True                     |                  0.0186 |                      0.0826 |       0.2647 |            -0.1028 |                              -0.6741 |                        -0.1019 |                 -0.0470 |
| us       | benchmark          | SPX                                                      | True                     |                  0.2577 |                      0.1287 |       1.8469 |            -0.0994 |                               0.0000 |                         1.0000 |                  1.0000 |
| us       | best_frozen_single | us_lb6M_sk1M_q10                                         | True                     |                 -0.0594 |                      0.0979 |      -0.5767 |            -0.1388 |                              -1.7303 |                        -0.1090 |                 -0.0830 |

## Guardrails

- The holdout was not used to change folds, stability criteria, selection or transaction-cost assumptions.
- No unstable configuration was added merely to force an ens-3.
- A missing stable ensemble is a valid finding.
- Protection must use the frozen base strategy and the unchanged 63-day/q90 rule family.
- US and MOEX strategy results remain price-return based; ordinary cash dividends are not included.
