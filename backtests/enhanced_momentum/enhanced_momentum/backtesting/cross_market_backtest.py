from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from enhanced_momentum.data_loaders.base import MarketData


StalePricePolicy = Literal["last_observation_carried_forward"]


@dataclass(frozen=True)
class CrossMarketBacktestConfig:
    """Configuration for the unified cross-market momentum backtest.

    Semantics
    ---------
    lookback_days
        Number of observations in the momentum measurement window.
    skip_days
        Number of most-recent observations excluded from the signal.
    quantile
        Fraction of eligible assets held in each tail.
    rebal_freq
        Rebalance frequency. The baseline uses month-end decisions ("ME").
    gross_exposure
        Total absolute target exposure. With the default 1.0, the portfolio
        allocates +0.5 gross to longs and -0.5 gross to shorts.
    transaction_cost_bps
        One-way transaction cost applied to 0.5 * L1 weight turnover.
        The gross baseline uses 0.0 bps.
    min_eligible_assets
        Minimum number of eligible assets required to form a portfolio.
    annualization_days
        Optional override. Defaults to 365 for crypto and 252 otherwise.
    stale_price_policy
        Valuation policy for a held asset whose raw close is temporarily
        unavailable. The baseline uses last-observation-carried-forward:
        the position is marked at its last observed close (0% return while the
        quote is stale), and the full price change is recognized when a new
        close appears.
    """

    lookback_days: int
    skip_days: int
    quantile: float
    rebal_freq: str = "ME"
    gross_exposure: float = 1.0
    transaction_cost_bps: float = 0.0
    min_eligible_assets: int = 2
    annualization_days: int | None = None
    stale_price_policy: StalePricePolicy = "last_observation_carried_forward"

    def __post_init__(self) -> None:
        if self.lookback_days < 2:
            raise ValueError("lookback_days must be >= 2.")
        if self.skip_days < 0:
            raise ValueError("skip_days must be >= 0.")
        if not 0.0 < self.quantile <= 0.5:
            raise ValueError("quantile must be in (0, 0.5].")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be > 0.")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be >= 0.")
        if self.min_eligible_assets < 2:
            raise ValueError("min_eligible_assets must be >= 2.")
        if self.stale_price_policy != "last_observation_carried_forward":
            raise ValueError(
                "Only stale_price_policy='last_observation_carried_forward' "
                "is currently supported."
            )


@dataclass
class CrossMarketBacktestResult:
    """Outputs of one unified cross-market backtest."""

    market: str
    config: CrossMarketBacktestConfig
    daily_returns: pd.Series
    nav: pd.Series
    turnover: pd.Series
    rebalance_diagnostics: pd.DataFrame
    stale_held_gross_exposure: pd.Series
    stale_price_events: pd.DataFrame
    stale_recovery_events: pd.DataFrame
    holdings: pd.DataFrame | None = None

    @property
    def annualization_days(self) -> int:
        if self.config.annualization_days is not None:
            return self.config.annualization_days
        return 365 if self.market.lower() == "crypto" else 252

    def metrics(self) -> pd.Series:
        """Return a compact, machine-readable metric set for grid summaries."""
        r = self.daily_returns.dropna().astype(float)
        ann = self.annualization_days

        if r.empty:
            raise ValueError("Cannot compute metrics: daily_returns is empty.")

        total_return = float((1.0 + r).prod() - 1.0)
        n_days = int(len(r))
        years = n_days / ann

        if total_return > -1.0 and years > 0:
            annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
        else:
            annualized_return = np.nan

        daily_vol = float(r.std(ddof=1))
        annualized_vol = daily_vol * np.sqrt(ann) if daily_vol > 0 else 0.0
        sharpe = (
            float(r.mean() / daily_vol * np.sqrt(ann))
            if daily_vol > 0
            else np.nan
        )

        nav = self.nav.reindex(r.index)
        drawdown = nav / nav.cummax() - 1.0
        max_drawdown = float(drawdown.min())

        calmar = (
            float(annualized_return / abs(max_drawdown))
            if np.isfinite(annualized_return) and max_drawdown < 0
            else np.nan
        )

        rebalance_turnover = self.turnover[self.turnover > 0]
        diagnostics = self.rebalance_diagnostics

        avg_n_eligible = (
            float(diagnostics["n_eligible"].mean())
            if not diagnostics.empty
            else np.nan
        )
        avg_n_long = (
            float(diagnostics["n_long"].mean())
            if not diagnostics.empty
            else np.nan
        )
        avg_n_short = (
            float(diagnostics["n_short"].mean())
            if not diagnostics.empty
            else np.nan
        )

        stale = self.stale_held_gross_exposure.reindex(r.index).fillna(0.0)
        recoveries = self.stale_recovery_events

        return pd.Series(
            {
                "market": self.market,
                "lookback_days": self.config.lookback_days,
                "skip_days": self.config.skip_days,
                "quantile": self.config.quantile,
                "rebal_freq": self.config.rebal_freq,
                "gross_exposure": self.config.gross_exposure,
                "transaction_cost_bps": self.config.transaction_cost_bps,
                "annualization_days": ann,
                "start_date": r.index.min().date().isoformat(),
                "end_date": r.index.max().date().isoformat(),
                "n_days": n_days,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_vol": annualized_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "calmar": calmar,
                "daily_hit_rate": float((r > 0).mean()),
                "n_rebalances": int(len(diagnostics)),
                "total_turnover": float(self.turnover.sum()),
                "annualized_turnover": (
                    float(self.turnover.sum() / years) if years > 0 else np.nan
                ),
                "avg_rebalance_turnover": (
                    float(rebalance_turnover.mean())
                    if not rebalance_turnover.empty
                    else 0.0
                ),
                "avg_n_eligible": avg_n_eligible,
                "avg_n_long": avg_n_long,
                "avg_n_short": avg_n_short,
                "days_with_stale_held_prices": int((stale > 0).sum()),
                "avg_stale_held_gross_exposure": float(stale.mean()),
                "max_stale_held_gross_exposure": float(stale.max()),
                "n_stale_recovery_events": int(len(recoveries)),
                "max_abs_stale_recovery_return": (
                    float(recoveries["valuation_return"].abs().max())
                    if not recoveries.empty
                    else 0.0
                ),
                "max_abs_stale_recovery_contribution": (
                    float(
                        recoveries["weighted_return_contribution"]
                        .abs()
                        .max()
                    )
                    if not recoveries.empty
                    else 0.0
                ),
            },
            dtype=object,
        )

    def metrics_frame(self) -> pd.DataFrame:
        return self.metrics().to_frame().T


def _validate_market_data(
    data: MarketData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and align MarketData without eagerly copying the full US panel."""
    close = data.close
    returns = data.returns
    presence = data.presence_matrix

    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = pd.to_datetime(close.index)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)
    if not isinstance(presence.index, pd.DatetimeIndex):
        presence = presence.copy()
        presence.index = pd.to_datetime(presence.index)

    if close.index.has_duplicates:
        raise ValueError("close index contains duplicate dates.")
    if close.columns.has_duplicates:
        raise ValueError("close contains duplicate asset columns.")

    if not close.index.is_monotonic_increasing:
        close = close.sort_index()

    close_cols = close.columns.astype(str)
    if not close.columns.equals(close_cols):
        close = close.copy()
        close.columns = close_cols

    returns_cols = returns.columns.astype(str)
    if not returns.columns.equals(returns_cols):
        returns = returns.copy()
        returns.columns = returns_cols

    presence_cols = presence.columns.astype(str)
    if not presence.columns.equals(presence_cols):
        presence = presence.copy()
        presence.columns = presence_cols

    if not returns.index.equals(close.index) or not returns.columns.equals(close.columns):
        returns = returns.reindex(index=close.index, columns=close.columns)

    if not presence.index.equals(close.index) or not presence.columns.equals(close.columns):
        presence = presence.reindex(index=close.index, columns=close.columns)

    # Avoid an expensive full-panel conversion when loaders already returned
    # numeric frames, which is the normal path for all three markets.
    if not all(is_numeric_dtype(dtype) for dtype in close.dtypes):
        close = close.apply(pd.to_numeric, errors="coerce")
    if not all(is_numeric_dtype(dtype) for dtype in returns.dtypes):
        returns = returns.apply(pd.to_numeric, errors="coerce")

    # Keep the full presence panel in its loader-provided dtype. Converting the
    # 10k+ asset US history to bool on every grid run would be unnecessarily
    # expensive; only the rebalance-date row is converted below.

    if close.empty:
        raise ValueError("MarketData.close is empty.")
    if returns.empty:
        raise ValueError("MarketData.returns is empty.")

    return close, returns, presence


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return the last available observation for each rebalance period."""
    freq_norm = freq.strip().upper()

    if freq_norm in {"M", "ME", "MONTHLY"}:
        dates = (
            pd.Series(index, index=index)
            .groupby(index.to_period("M"))
            .last()
            .tolist()
        )
        return pd.DatetimeIndex(dates)

    dates = pd.Series(index, index=index).resample(freq).last().dropna()
    return pd.DatetimeIndex(dates.to_numpy())


def compute_momentum_scores(
    close: pd.DataFrame,
    *,
    rebalance_pos: int,
    lookback_days: int,
    skip_days: int,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp] | None:
    """Compute classic price momentum at one rebalance point.

    The signal uses exactly ``lookback_days`` observations and excludes the
    most recent ``skip_days`` observations. The score is P_end / P_start - 1.
    Exact start and end prices are required.
    """
    signal_end_pos = rebalance_pos - skip_days
    signal_start_pos = signal_end_pos - lookback_days + 1

    if signal_start_pos < 0 or signal_end_pos < 0:
        return None

    p_start = close.iloc[signal_start_pos]
    p_end = close.iloc[signal_end_pos]

    scores = p_end / p_start - 1.0
    scores = scores.where((p_start > 0) & (p_end > 0))
    scores = scores.replace([np.inf, -np.inf], np.nan).astype(float)

    return (
        scores,
        pd.Timestamp(close.index[signal_start_pos]),
        pd.Timestamp(close.index[signal_end_pos]),
    )


def build_long_short_target(
    scores: pd.Series,
    eligible: pd.Series,
    *,
    quantile: float,
    gross_exposure: float = 1.0,
    min_eligible_assets: int = 2,
) -> tuple[pd.Series, dict[str, int]]:
    """Build an equal-weight, market-neutral long-short target portfolio."""
    eligible = eligible.reindex(scores.index).fillna(False).astype(bool)
    valid_scores = scores[eligible & scores.notna()].sort_values()

    n_eligible = int(len(valid_scores))
    # Keep targets sparse. This matters for the 10k+ asset US history and
    # makes repeated grid runs substantially lighter.
    target = pd.Series(dtype=float)

    if n_eligible < min_eligible_assets:
        return target, {
            "n_eligible": n_eligible,
            "n_long": 0,
            "n_short": 0,
        }

    n_side = int(np.floor(n_eligible * quantile))
    n_side = max(1, n_side)
    n_side = min(n_side, n_eligible // 2)

    if n_side < 1:
        return target, {
            "n_eligible": n_eligible,
            "n_long": 0,
            "n_short": 0,
        }

    short_assets = valid_scores.index[:n_side]
    long_assets = valid_scores.index[-n_side:]

    sleeve_gross = gross_exposure / 2.0
    target = pd.concat(
        [
            pd.Series(sleeve_gross / len(long_assets), index=long_assets, dtype=float),
            pd.Series(-sleeve_gross / len(short_assets), index=short_assets, dtype=float),
        ]
    )

    return target, {
        "n_eligible": n_eligible,
        "n_long": int(len(long_assets)),
        "n_short": int(len(short_assets)),
    }


def run_cross_market_backtest(
    data: MarketData,
    config: CrossMarketBacktestConfig,
    *,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    store_holdings: bool = False,
) -> CrossMarketBacktestResult:
    """Run the unified momentum backtest for US, MOEX, or crypto.

    Timing convention
    -----------------
    1. Signal is computed after the close of a rebalance decision date.
    2. Eligibility is read from raw close + presence on that decision date.
    3. The target portfolio becomes effective on the next available date.
    4. Returns are realized forward, so there is no same-day look-ahead.
    5. Between rebalances, positions drift with realized asset returns.

    Valuation convention
    --------------------
    Signal construction and eligibility always use the raw, unfilled close
    panel. Portfolio valuation uses last-observation-carried-forward (LOCF):

    - if a held asset has no raw close on a date, its valuation return is 0%;
    - the last observed mark is retained;
    - when a new close appears, the full move from the retained mark to the
      new close is recognized on that date;
    - presence=False does not trigger a forced exit, because presence is an
      investability filter rather than proof of permanent delisting;
    - positions leave the portfolio only through the next scheduled rebalance.

    This avoids both look-ahead delisting logic and the loss of price moves
    across temporary quotation gaps.
    """
    close, _raw_returns, presence = _validate_market_data(data)

    market = str(data.metadata.get("market", "unknown")).lower()
    idx = close.index

    start_ts = pd.Timestamp(start_date) if start_date is not None else idx.min()
    end_ts = pd.Timestamp(end_date) if end_date is not None else idx.max()

    if start_ts > end_ts:
        raise ValueError("start_date must be <= end_date.")

    # Build decisions over the full history and filter by EFFECTIVE date below.
    # This lets a month-end decision immediately before start_date create the
    # portfolio held from the first available observation in the test period.
    candidate_rebalances = _rebalance_dates(idx, config.rebal_freq)

    effective_targets: dict[pd.Timestamp, pd.Series] = {}
    effective_decision_dates: dict[pd.Timestamp, pd.Timestamp] = {}
    diagnostics_records: list[dict[str, object]] = []
    holdings_records: list[dict[str, object]] = []

    # ================================================================
    # Build target portfolios
    # ================================================================
    for decision_date in candidate_rebalances:
        decision_date = pd.Timestamp(decision_date)
        rebalance_pos = idx.get_loc(decision_date)

        if not isinstance(rebalance_pos, (int, np.integer)):
            raise RuntimeError(
                "Unexpected non-scalar rebalance index location."
            )

        score_result = compute_momentum_scores(
            close,
            rebalance_pos=int(rebalance_pos),
            lookback_days=config.lookback_days,
            skip_days=config.skip_days,
        )
        if score_result is None:
            continue

        scores, signal_start, signal_end = score_result

        presence_row = (
            presence.loc[decision_date]
            .fillna(False)
            .astype(bool)
        )

        raw_close_on_decision = close.loc[decision_date]
        present_now = presence_row & raw_close_on_decision.notna()
        valid_score_mask = scores.notna()
        eligible = present_now & valid_score_mask

        n_present = int(present_now.sum())
        n_valid_scores = int((present_now & valid_score_mask).sum())

        target, counts = build_long_short_target(
            scores,
            eligible,
            quantile=config.quantile,
            gross_exposure=config.gross_exposure,
            min_eligible_assets=config.min_eligible_assets,
        )

        next_pos = int(rebalance_pos) + 1
        if next_pos >= len(idx):
            continue

        effective_date = pd.Timestamp(idx[next_pos])

        if effective_date < start_ts or effective_date > end_ts:
            continue

        target = target[target != 0.0].astype(float)

        effective_targets[effective_date] = target
        effective_decision_dates[effective_date] = decision_date

        diagnostics_records.append(
            {
                "decision_date": decision_date,
                "effective_date": effective_date,
                "signal_start": signal_start,
                "signal_end": signal_end,
                "n_present": n_present,
                "n_valid_scores": n_valid_scores,
                **counts,
            }
        )

        if store_holdings and not target.empty:
            for asset, weight in target.items():
                holdings_records.append(
                    {
                        "decision_date": decision_date,
                        "effective_date": effective_date,
                        "asset": str(asset),
                        "weight": float(weight),
                        "side": "long" if weight > 0 else "short",
                    }
                )

    if not effective_targets:
        raise ValueError(
            "No valid rebalance targets were produced. Check the requested "
            "date range, lookback_days, skip_days, and presence matrix."
        )

    first_effective_date = min(effective_targets)
    simulation_index = idx[
        (idx >= first_effective_date)
        & (idx <= end_ts)
    ]

    # Sparse state: only currently held assets.
    current_weights = pd.Series(dtype=float)

    # Last observed raw close used as the valuation mark for each held asset.
    current_marks = pd.Series(dtype=float)

    # Date on which the current mark was last observed in the raw close panel.
    current_mark_dates = pd.Series(dtype="datetime64[ns]")

    # Number of consecutive simulation observations for which a held asset
    # has had no raw close since its last observed valuation mark.
    current_stale_observations = pd.Series(dtype=int)

    tc_rate = config.transaction_cost_bps / 10_000.0

    daily_returns: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    stale_exposure: dict[pd.Timestamp, float] = {}
    stale_price_records: list[dict[str, object]] = []
    stale_recovery_records: list[dict[str, object]] = []

    # ================================================================
    # Daily simulation
    # ================================================================
    for date in simulation_index:
        date = pd.Timestamp(date)

        day_turnover = 0.0
        transaction_cost = 0.0

        # ------------------------------------------------------------
        # Scheduled rebalance
        # ------------------------------------------------------------
        if date in effective_targets:
            target = effective_targets[date].copy().astype(float)

            all_assets = current_weights.index.union(target.index)
            old_aligned = current_weights.reindex(
                all_assets,
                fill_value=0.0,
            )
            target_aligned = target.reindex(
                all_assets,
                fill_value=0.0,
            )

            rebalance_turnover = 0.5 * float(
                (target_aligned - old_aligned).abs().sum()
            )
            day_turnover += rebalance_turnover
            transaction_cost += rebalance_turnover * tc_rate

            current_weights = target.copy()

            if current_weights.empty:
                current_marks = pd.Series(dtype=float)
                current_mark_dates = pd.Series(dtype="datetime64[ns]")
                current_stale_observations = pd.Series(dtype=int)
            else:
                decision_date = effective_decision_dates[date]

                # Every target asset is required to have a raw close on the
                # decision date by the eligibility rule above.
                marks = (
                    close.loc[
                        decision_date,
                        current_weights.index,
                    ]
                    .astype(float)
                )

                if marks.isna().any():
                    bad_assets = marks.index[marks.isna()].tolist()
                    raise RuntimeError(
                        "Target contains assets without a decision-date close: "
                        f"{bad_assets[:10]}"
                    )

                if (marks <= 0).any():
                    bad_assets = marks.index[marks <= 0].tolist()
                    raise RuntimeError(
                        "Target contains non-positive decision-date closes: "
                        f"{bad_assets[:10]}"
                    )

                current_marks = marks.copy()
                current_mark_dates = pd.Series(
                    decision_date,
                    index=current_weights.index,
                    dtype="datetime64[ns]",
                )
                current_stale_observations = pd.Series(
                    0,
                    index=current_weights.index,
                    dtype=int,
                )

        # ------------------------------------------------------------
        # No positions: portfolio is entirely in cash
        # ------------------------------------------------------------
        if current_weights.empty:
            gross_portfolio_return = 0.0
            stale_gross = 0.0
            realized = pd.Series(dtype=float)

        else:
            raw_close_today = (
                close.loc[
                    date,
                    current_weights.index,
                ]
                .astype(float)
            )

            presence_today = (
                presence.loc[
                    date,
                    current_weights.index,
                ]
                .fillna(False)
                .astype(bool)
            )

            stale_mask = raw_close_today.isna()
            observed_mask = ~stale_mask

            stale_gross = float(
                current_weights[stale_mask]
                .abs()
                .sum()
            )

            if stale_mask.any():
                stale_assets = current_weights.index[stale_mask]
                current_stale_observations.loc[stale_assets] = (
                    current_stale_observations.loc[stale_assets] + 1
                )

            # --------------------------------------------------------
            # Stale-price audit trail
            # --------------------------------------------------------
            if stale_mask.any():
                for asset in current_weights.index[stale_mask]:
                    last_mark_date = pd.Timestamp(
                        current_mark_dates.loc[asset]
                    )

                    stale_price_records.append(
                        {
                            "date": date,
                            "asset": str(asset),
                            "weight": float(current_weights.loc[asset]),
                            "abs_weight": float(
                                abs(current_weights.loc[asset])
                            ),
                            "side": (
                                "long"
                                if current_weights.loc[asset] > 0
                                else "short"
                            ),
                            "presence_on_date": bool(
                                presence_today.loc[asset]
                            ),
                            "last_mark_price": float(
                                current_marks.loc[asset]
                            ),
                            "last_observed_close_date": last_mark_date,
                            "calendar_days_since_last_observed_close": int(
                                (date - last_mark_date).days
                            ),
                            "stale_observation_number": int(
                                current_stale_observations.loc[asset]
                            ),
                        }
                    )

            # --------------------------------------------------------
            # Mark-to-last valuation returns
            # --------------------------------------------------------
            realized = pd.Series(
                0.0,
                index=current_weights.index,
                dtype=float,
            )

            if observed_mask.any():
                observed_assets = current_weights.index[observed_mask]

                observed_closes = raw_close_today.loc[
                    observed_assets
                ]
                previous_marks = current_marks.loc[
                    observed_assets
                ]

                if (observed_closes <= 0).any():
                    bad_assets = observed_closes.index[
                        observed_closes <= 0
                    ].tolist()
                    raise RuntimeError(
                        "Observed non-positive close while holding assets: "
                        f"{bad_assets[:10]}"
                    )

                if previous_marks.isna().any() or (previous_marks <= 0).any():
                    bad_assets = previous_marks.index[
                        previous_marks.isna() | (previous_marks <= 0)
                    ].tolist()
                    raise RuntimeError(
                        "Invalid previous valuation mark for held assets: "
                        f"{bad_assets[:10]}"
                    )

                observed_returns = (
                    observed_closes
                    / previous_marks
                    - 1.0
                ).replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

                if observed_returns.isna().any():
                    bad_assets = observed_returns.index[
                        observed_returns.isna()
                    ].tolist()
                    raise RuntimeError(
                        "Could not compute valuation return for held assets: "
                        f"{bad_assets[:10]}"
                    )

                # A recovery is a newly observed raw close after one or more
                # stale simulation observations while the asset remained held.
                recovery_assets = observed_assets[
                    current_stale_observations.loc[observed_assets] > 0
                ]

                for asset in recovery_assets:
                    weight_before_return = float(current_weights.loc[asset])
                    valuation_return = float(observed_returns.loc[asset])
                    last_mark_date = pd.Timestamp(
                        current_mark_dates.loc[asset]
                    )
                    stale_observations = int(
                        current_stale_observations.loc[asset]
                    )
                    weighted_contribution = (
                        weight_before_return * valuation_return
                    )

                    stale_recovery_records.append(
                        {
                            "date": date,
                            "asset": str(asset),
                            "weight_before_return": weight_before_return,
                            "abs_weight_before_return": abs(
                                weight_before_return
                            ),
                            "side": (
                                "long"
                                if weight_before_return > 0
                                else "short"
                            ),
                            "presence_on_date": bool(
                                presence_today.loc[asset]
                            ),
                            "previous_mark_price": float(
                                previous_marks.loc[asset]
                            ),
                            "recovery_close": float(
                                observed_closes.loc[asset]
                            ),
                            "last_observed_close_date": last_mark_date,
                            "stale_observations": stale_observations,
                            "calendar_days_since_last_observed_close": int(
                                (date - last_mark_date).days
                            ),
                            "valuation_return": valuation_return,
                            "weighted_return_contribution": float(
                                weighted_contribution
                            ),
                            "abs_weighted_return_contribution": float(
                                abs(weighted_contribution)
                            ),
                        }
                    )

                realized.loc[observed_assets] = observed_returns

                # The stale streak ends when a new raw close is observed.
                current_stale_observations.loc[observed_assets] = 0

                # Update marks only when a new raw close is actually observed.
                current_marks.loc[observed_assets] = observed_closes
                current_mark_dates.loc[observed_assets] = date

            gross_portfolio_return = float(
                (current_weights * realized).sum()
            )

        # ------------------------------------------------------------
        # Net daily return
        # ------------------------------------------------------------
        net_portfolio_return = (
            gross_portfolio_return
            - transaction_cost
        )

        daily_returns[date] = net_portfolio_return
        turnover[date] = day_turnover
        stale_exposure[date] = stale_gross

        # ------------------------------------------------------------
        # Drift positions to end-of-day weights
        # ------------------------------------------------------------
        if current_weights.empty:
            continue

        denominator = 1.0 + net_portfolio_return

        if denominator <= 0:
            # Portfolio has lost 100% or more of NAV. Stop rather than
            # continue with economically meaningless negative NAV.
            break

        current_weights = (
            current_weights
            * (1.0 + realized)
            / denominator
        )

        current_weights = current_weights[
            current_weights != 0.0
        ]

        # Keep valuation state aligned with the sparse holdings vector.
        current_marks = current_marks.reindex(
            current_weights.index
        )
        current_mark_dates = current_mark_dates.reindex(
            current_weights.index
        )
        current_stale_observations = current_stale_observations.reindex(
            current_weights.index,
            fill_value=0,
        ).astype(int)

    # ================================================================
    # Build result objects
    # ================================================================
    daily_returns_s = pd.Series(
        daily_returns,
        name="strategy_return",
        dtype=float,
    )

    turnover_s = pd.Series(
        turnover,
        name="turnover",
        dtype=float,
    ).reindex(
        daily_returns_s.index,
        fill_value=0.0,
    )

    stale_exposure_s = pd.Series(
        stale_exposure,
        name="stale_held_gross_exposure",
        dtype=float,
    ).reindex(
        daily_returns_s.index,
        fill_value=0.0,
    )

    nav = (1.0 + daily_returns_s).cumprod()
    nav.name = "nav"

    diagnostics = pd.DataFrame(diagnostics_records)
    if not diagnostics.empty:
        diagnostics = (
            diagnostics
            .sort_values("effective_date")
            .reset_index(drop=True)
        )

    stale_price_events = pd.DataFrame(
        stale_price_records,
        columns=[
            "date",
            "asset",
            "weight",
            "abs_weight",
            "side",
            "presence_on_date",
            "last_mark_price",
            "last_observed_close_date",
            "calendar_days_since_last_observed_close",
            "stale_observation_number",
        ],
    )

    if not stale_price_events.empty:
        stale_price_events = (
            stale_price_events
            .sort_values(["date", "asset"])
            .reset_index(drop=True)
        )

    stale_recovery_events = pd.DataFrame(
        stale_recovery_records,
        columns=[
            "date",
            "asset",
            "weight_before_return",
            "abs_weight_before_return",
            "side",
            "presence_on_date",
            "previous_mark_price",
            "recovery_close",
            "last_observed_close_date",
            "stale_observations",
            "calendar_days_since_last_observed_close",
            "valuation_return",
            "weighted_return_contribution",
            "abs_weighted_return_contribution",
        ],
    )

    if not stale_recovery_events.empty:
        stale_recovery_events = (
            stale_recovery_events
            .sort_values(["date", "asset"])
            .reset_index(drop=True)
        )

    holdings = (
        pd.DataFrame(holdings_records)
        if store_holdings
        else None
    )

    return CrossMarketBacktestResult(
        market=market,
        config=config,
        daily_returns=daily_returns_s,
        nav=nav,
        turnover=turnover_s,
        rebalance_diagnostics=diagnostics,
        stale_held_gross_exposure=stale_exposure_s,
        stale_price_events=stale_price_events,
        stale_recovery_events=stale_recovery_events,
        holdings=holdings,
    )

def config_to_dict(config: CrossMarketBacktestConfig) -> dict[str, object]:
    """Serialize a config for JSON provenance."""
    return asdict(config)