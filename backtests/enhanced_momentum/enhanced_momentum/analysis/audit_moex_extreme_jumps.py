from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from enhanced_momentum.data_loaders.registry import load_market


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"
REPORT_DIR = (
    PROJECT_ROOT
    / "backtests"
    / "enhanced_momentum"
    / "reports"
    / "data_quality"
)

STANDARD_CORPORATE_ACTION_FACTORS = np.array(
    [
        2,
        3,
        4,
        5,
        10,
        20,
        25,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
    ],
    dtype=float,
)


def _nearest_standard_factor(multiplier: float) -> tuple[float, float]:
    """Return the closest common split/consolidation factor and relative gap."""
    if not np.isfinite(multiplier) or multiplier <= 1.0:
        return np.nan, np.nan

    log_distance = np.abs(
        np.log(multiplier)
        - np.log(STANDARD_CORPORATE_ACTION_FACTORS)
    )
    factor = float(
        STANDARD_CORPORATE_ACTION_FACTORS[
            int(np.argmin(log_distance))
        ]
    )
    relative_gap = float(abs(multiplier / factor - 1.0))
    return factor, relative_gap


def _build_event_table(
    close: pd.DataFrame,
    presence: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Find extreme observed-to-observed raw price jumps.

    A new raw close is compared with the last previously observed raw close,
    even when one or more exchange observations between them are missing.
    This matches the mark-to-last valuation logic used by the frozen engine
    and catches corporate-action jumps after quotation gaps.
    """
    close = close.sort_index().astype(float)
    presence = presence.reindex(
        index=close.index,
        columns=close.columns,
    )

    valid_close = close.notna() & (close > 0)

    previous_observed_close = (
        close.ffill().shift(1)
    )

    date_matrix = pd.DataFrame(
        np.broadcast_to(
            close.index.to_numpy()[:, None],
            close.shape,
        ),
        index=close.index,
        columns=close.columns,
    ).where(valid_close)

    previous_observed_date = (
        date_matrix.ffill().shift(1)
    )

    observed_return = (
        close
        / previous_observed_close
        - 1.0
    ).where(
        valid_close
        & previous_observed_close.notna()
        & (previous_observed_close > 0)
    )

    extreme_mask = (
        observed_return.abs() >= threshold
    )

    row_positions, col_positions = np.where(
        extreme_mask.to_numpy()
    )

    records: list[dict[str, object]] = []

    for row_pos, col_pos in zip(
        row_positions,
        col_positions,
        strict=False,
    ):
        date = pd.Timestamp(
            close.index[row_pos]
        )
        ticker = str(
            close.columns[col_pos]
        )

        prev_date_raw = previous_observed_date.iat[
            row_pos,
            col_pos,
        ]

        prev_date = (
            pd.Timestamp(prev_date_raw)
            if pd.notna(prev_date_raw)
            else pd.NaT
        )

        prev_close = float(
            previous_observed_close.iat[
                row_pos,
                col_pos,
            ]
        )
        current_close = float(
            close.iat[
                row_pos,
                col_pos,
            ]
        )
        ret = float(
            observed_return.iat[
                row_pos,
                col_pos,
            ]
        )

        price_ratio = (
            current_close
            / prev_close
        )

        corporate_action_multiplier = max(
            price_ratio,
            1.0 / price_ratio,
        )

        nearest_factor, factor_relative_gap = (
            _nearest_standard_factor(
                corporate_action_multiplier
            )
        )

        presence_on_date = bool(
            presence.iat[
                row_pos,
                col_pos,
            ]
        ) if pd.notna(
            presence.iat[
                row_pos,
                col_pos,
            ]
        ) else False

        presence_on_prev_date = np.nan

        if pd.notna(prev_date):
            try:
                prev_presence_value = presence.loc[
                    prev_date,
                    ticker,
                ]
                presence_on_prev_date = (
                    bool(prev_presence_value)
                    if pd.notna(prev_presence_value)
                    else False
                )
            except KeyError:
                presence_on_prev_date = np.nan

        records.append(
            {
                "date": date,
                "ticker": ticker,
                "previous_observed_date": prev_date,
                "previous_close": prev_close,
                "close": current_close,
                "observed_to_observed_return": ret,
                "abs_return": abs(ret),
                "price_ratio": price_ratio,
                "direction": (
                    "up"
                    if ret > 0
                    else "down"
                ),
                "calendar_days_since_previous_observation": (
                    int(
                        (
                            date
                            - prev_date
                        ).days
                    )
                    if pd.notna(prev_date)
                    else np.nan
                ),
                "presence_on_previous_observed_date":
                    presence_on_prev_date,
                "presence_on_date":
                    presence_on_date,
                "corporate_action_multiplier":
                    corporate_action_multiplier,
                "nearest_standard_factor":
                    nearest_factor,
                "factor_relative_gap":
                    factor_relative_gap,
            }
        )

    events = pd.DataFrame(records)

    if not events.empty:
        events = (
            events
            .sort_values(
                [
                    "abs_return",
                    "date",
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

    return events


def _build_threshold_summary(
    observed_returns: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = [
        0.50,
        1.00,
        3.00,
        10.00,
        100.00,
        1000.00,
    ]

    rows: list[dict[str, object]] = []

    for threshold in thresholds:
        mask = (
            observed_returns.abs()
            >= threshold
        )

        tickers = (
            mask.any(axis=0)
        )

        rows.append(
            {
                "abs_return_threshold": threshold,
                "n_events": int(
                    mask.sum().sum()
                ),
                "n_tickers": int(
                    tickers.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_context(
    close: pd.DataFrame,
    presence: pd.DataFrame,
    events: pd.DataFrame,
    *,
    n_events: int,
    observations_each_side: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    if events.empty:
        return pd.DataFrame()

    for event_rank, event in (
        events.head(n_events)
        .reset_index(drop=True)
        .iterrows()
    ):
        ticker = str(event["ticker"])
        event_date = pd.Timestamp(
            event["date"]
        )

        series = (
            close[ticker]
            .dropna()
            .sort_index()
        )

        if event_date not in series.index:
            continue

        event_pos = int(
            series.index.get_loc(
                event_date
            )
        )

        start = max(
            0,
            event_pos
            - observations_each_side,
        )
        end = min(
            len(series),
            event_pos
            + observations_each_side
            + 1,
        )

        window = series.iloc[
            start:end
        ]

        for observed_date, price in (
            window.items()
        ):
            offset = int(
                series.index.get_loc(
                    observed_date
                )
                - event_pos
            )

            presence_value = np.nan

            if (
                observed_date
                in presence.index
                and ticker
                in presence.columns
            ):
                raw_presence = presence.loc[
                    observed_date,
                    ticker,
                ]
                presence_value = (
                    bool(raw_presence)
                    if pd.notna(raw_presence)
                    else False
                )

            records.append(
                {
                    "event_rank":
                        event_rank + 1,
                    "event_date":
                        event_date,
                    "ticker":
                        ticker,
                    "event_return":
                        float(
                            event[
                                "observed_to_observed_return"
                            ]
                        ),
                    "relative_observed_offset":
                        offset,
                    "observed_date":
                        pd.Timestamp(
                            observed_date
                        ),
                    "close":
                        float(price),
                    "presence":
                        presence_value,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit MOEX raw close history for extreme observed-to-observed "
            "price jumps that may reflect splits, reverse splits, unit changes, "
            "or other corporate-action artefacts."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help=(
            "Minimum absolute observed-to-observed return to report. "
            "Default: 1.0 (= 100%%)."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help=(
            "Number of largest events to print. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--context-events",
        type=int,
        default=20,
        help=(
            "Number of largest events for which to save local raw-price "
            "context. Default: 20."
        ),
    )

    parser.add_argument(
        "--context-observations",
        type=int,
        default=3,
        help=(
            "Observed raw closes to save before and after each context event. "
            "Default: 3."
        ),
    )

    args = parser.parse_args()

    if args.threshold <= 0:
        raise ValueError(
            "--threshold must be > 0."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading MOEX MarketData...")
    data = load_market(
        "moex",
        config_dir=MARKETS_DIR,
    )
    print(data.summary())

    close = (
        data.close
        .sort_index()
        .astype(float)
    )

    presence = (
        data.presence_matrix
        .reindex(
            index=close.index,
            columns=close.columns,
        )
    )

    valid_close = (
        close.notna()
        & (close > 0)
    )

    previous_observed_close = (
        close.ffill().shift(1)
    )

    observed_returns = (
        close
        / previous_observed_close
        - 1.0
    ).where(
        valid_close
        & previous_observed_close.notna()
        & (previous_observed_close > 0)
    )

    events = _build_event_table(
        close,
        presence,
        threshold=args.threshold,
    )

    threshold_summary = (
        _build_threshold_summary(
            observed_returns
        )
    )

    context = _build_context(
        close,
        presence,
        events,
        n_events=args.context_events,
        observations_each_side=(
            args.context_observations
        ),
    )

    events_csv = (
        REPORT_DIR
        / "moex_extreme_price_jumps.csv"
    )
    events_parquet = (
        REPORT_DIR
        / "moex_extreme_price_jumps.parquet"
    )
    summary_csv = (
        REPORT_DIR
        / "moex_extreme_price_jump_summary.csv"
    )
    context_csv = (
        REPORT_DIR
        / "moex_extreme_price_jump_context.csv"
    )

    events.to_csv(
        events_csv,
        index=False,
    )
    events.to_parquet(
        events_parquet,
        index=False,
    )
    threshold_summary.to_csv(
        summary_csv,
        index=False,
    )
    context.to_csv(
        context_csv,
        index=False,
    )

    print()
    print("=" * 110)
    print(
        "MOEX EXTREME RAW PRICE JUMP AUDIT"
    )
    print("=" * 110)

    print(
        f"Threshold: |observed-to-observed return| "
        f">= {args.threshold:.0%}"
    )
    print(
        f"Extreme events: {len(events)}"
    )
    print(
        "Unique tickers: "
        f"{events['ticker'].nunique() if not events.empty else 0}"
    )

    print()
    print("Threshold summary:")
    print(
        threshold_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Top {min(args.top, len(events))} events:"
    )

    if events.empty:
        print("No events found.")
    else:
        columns = [
            "date",
            "ticker",
            "previous_observed_date",
            "previous_close",
            "close",
            "observed_to_observed_return",
            "price_ratio",
            "calendar_days_since_previous_observation",
            "presence_on_previous_observed_date",
            "presence_on_date",
            "corporate_action_multiplier",
            "nearest_standard_factor",
            "factor_relative_gap",
        ]

        print(
            events[
                columns
            ]
            .head(args.top)
            .to_string(
                index=False
            )
        )

    print()
    print("Saved:")
    print(f"  {events_csv}")
    print(f"  {events_parquet}")
    print(f"  {summary_csv}")
    print(f"  {context_csv}")


if __name__ == "__main__":
    main()
