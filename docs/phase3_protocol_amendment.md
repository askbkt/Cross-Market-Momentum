# Phase 3 Protocol Amendment — Cross-Market Momentum Study

## Status

This document freezes the corrected Phase 3 protocol before the corrected analysis is run.
It supersedes the provisional `results/phase3` implementation, but does not alter the
frozen baseline grid or its underlying daily return series.

## Why the amendment is required

The provisional Phase 3 implementation split the pre-holdout history into temporal
segments and aggregated segment-level ranks. That was a useful temporal robustness
analysis, but it was not a complete train-to-test walk-forward protocol: each fold did
not separately rank configurations on a past train window and then measure rank transfer
on the subsequent test window.

The provisional selector also forced an ensemble of three configurations, even when the
number of demonstrably stable configurations was unknown. The corrected protocol follows
the project DoD: stable configurations are identified before the holdout; an ens-3 is built
only when at least three stable configurations exist, an ens-2 when two exist, and otherwise
no stable ensemble is claimed.

## Frozen research sequence

1. Use the already frozen 81-run baseline grid.
2. Reconstruct gross and primary-cost net daily returns from the saved return and turnover files.
3. For every market and every fold, rank all eligible configurations on the train window.
4. Apply the frozen configurations mechanically to the next test window and measure train-to-test rank transfer.
5. Classify stable configurations using only pre-holdout train/test results.
6. Freeze the best single configuration and, when supported, an equal-weight stable ensemble.
7. Evaluate the frozen construction on the chronological retrospective holdout, 2023–2024.
8. Compare the frozen construction with the best single and the market proxy.
9. Run the pre-specified regime-aware protection rules with thresholds calibrated only on pre-holdout history.
10. Preserve positive and negative findings without post-holdout retuning.

## Holdout terminology

The 2023–2024 period is excluded from algorithmic selection and threshold calibration.
However, full-sample baseline summaries through 2024 had already been viewed before this
amendment. It is therefore reported as a **chronological retrospective holdout** or
**pseudo-OOS**, not as a fully pristine untouched OOS sample.

This limitation is disclosed rather than hidden. The protocol must not be changed again
in response to the corrected holdout results.

## Fold design

The folds are stored in `config/phase3_protocol_v2.yaml`.

- US uses expanding train windows and annual tests from 2015 through 2022.
- MOEX uses four pre-holdout tests, ending in 2022. Its actual common train start may be
  later than the nominal date because all configurations must share the same available
  return dates.
- Crypto uses two folds because of limited history. Conclusions are directional and have
  low statistical power.

For each fold and sample, all eligible configurations are evaluated on a common date index.
This prevents shorter-lookback configurations from receiving an artificial advantage from
having more observations than longer-lookback configurations.

## Ranking and stability

The primary ranking statistic is net Sharpe under the market-specific primary transaction
cost assumption:

- US: 25 bps per unit turnover;
- MOEX: 20 bps;
- Crypto: 10 bps.

Benchmark-relative Information Ratio is reported descriptively but is not used for
selection. The momentum strategy is market-neutral while SPX, IMOEX and BTC are long-only
market proxies, so a combined Sharpe/IR selector would mix two economically different
objectives.

The eligible universe is divided into thirds. A configuration is stable only when all of
the following hold:

- median train rank is in the top third;
- median test rank is in the top third;
- it is in the test top third in at least 50% of folds;
- its test net Sharpe is positive in at least 50% of folds;
- median absolute train-to-test rank change is no more than one top-third width.

The three known MOEX 12M/0M sparse-signal configurations remain visible in descriptive
outputs but are excluded from selection.

## Ensemble construction

- Three or more stable configurations: target ens-3.
- Exactly two stable configurations: ens-2.
- Fewer than two stable configurations: report no stable ensemble.

Among stable configurations, selection prioritizes median test rank, mean test net Sharpe,
worst test rank and then return-series diversification. The correlation filter may reduce
an ens-3 to an ens-2, but no unstable configuration may be added to fill an ensemble slot.

A frozen best single is always retained for comparison. When no stable configuration exists,
the best pre-holdout validation reference may be used as the base series for the protection
diagnostic, but it must be labelled as an unstable reference and not as a stable finalist.

## Transaction-cost sensitivity

Primary-cost selection is not repeated at alternative cost levels. Instead, the already
frozen best single and base construction are evaluated under pre-specified cost scenarios.
This measures the dependence of conclusions on transaction costs without creating another
selection loop.

## Protection protocol

The protection family is unchanged:

- no protection;
- portfolio volatility q90;
- market volatility q90;
- portfolio AND market volatility q90.

Volatility uses a 63-observation rolling standard deviation shifted by one observation.
Thresholds are estimated only on pre-holdout history and then frozen. OOS is not used to
choose a winning variant. Switching turnover follows the unified engine convention:
`0.5 × absolute exposure change`.

Historical crisis-window results are validation diagnostics only, not independent evidence.

## Prohibited changes after the corrected run

The following are not allowed in response to holdout results:

- changing fold dates;
- changing stability thresholds;
- adding unstable ensemble members;
- moving the holdout;
- trying q80/q85 or a different volatility window;
- adding an OR rule and selecting it because it looks better;
- choosing a different transaction-cost assumption as the primary result.

Any such extension belongs in explicitly labelled future work.
