from __future__ import annotations

from pathlib import Path

import pandas as pd

from enhanced_momentum.backtesting.cross_market_backtest import (
    CrossMarketBacktestConfig,
    run_cross_market_backtest,
)
from enhanced_momentum.data_loaders.registry import load_market


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"

EVENT_DATE = pd.Timestamp("2021-04-13")


def main() -> None:
    print("Loading Crypto...")

    data = load_market(
        "crypto",
        config_dir=MARKETS_DIR,
    )

    close = data.close.astype(float)

    # Same last-observation-carried-forward valuation logic
    # used by the unified backtest engine.
    valuation_close = close.ffill()

    valuation_returns = valuation_close.pct_change(
        fill_method=None
    )

    print()
    print("=" * 100)
    print("TOP 30 ABSOLUTE VALUATION RETURNS ON 2021-04-13")
    print("=" * 100)

    event_returns = (
        valuation_returns.loc[EVENT_DATE]
        .dropna()
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
    )

    print(
        event_returns.head(30).to_string()
    )

    # Worst configuration:
    # 24M lookback / 3M skip / q=10%
    config = CrossMarketBacktestConfig(
        lookback_days=730,
        skip_days=91,
        quantile=0.10,
        rebal_freq="ME",
        gross_exposure=1.0,
        transaction_cost_bps=0.0,
    )

    print()
    print("=" * 100)
    print("RERUNNING WORST CONFIG: 24M / 3M / q10")
    print("=" * 100)

    result = run_cross_market_backtest(
        data=data,
        config=config,
        start_date="2020-04-01",
        end_date="2021-04-13",
        store_holdings=True,
    )

    if result.holdings is None or result.holdings.empty:
        raise RuntimeError(
            "Backtest returned no holdings; cannot reconstruct event-day contributions."
        )

    holdings = result.holdings.copy()

    holdings["effective_date"] = pd.to_datetime(
        holdings["effective_date"]
    )

    latest_effective_date = (
        holdings.loc[
            holdings["effective_date"] <= EVENT_DATE,
            "effective_date",
        ]
        .max()
    )

    if pd.isna(latest_effective_date):
        raise RuntimeError(
            "No effective rebalance found on or before the event date."
        )

    target = (
        holdings[
            holdings["effective_date"]
            == latest_effective_date
        ]
        .set_index("asset")["weight"]
        .astype(float)
    )

    print()
    print(
        "Latest effective rebalance before event:",
        latest_effective_date.date(),
    )
    print(
        "Target holdings:",
        len(target),
    )

    # Replay drifting weights from the latest effective rebalance
    # through the day BEFORE the event.
    current_weights = target.copy()

    replay_dates = valuation_returns.index[
        (
            valuation_returns.index
            >= latest_effective_date
        )
        & (
            valuation_returns.index
            < EVENT_DATE
        )
    ]

    for date in replay_dates:
        realized = (
            valuation_returns.loc[
                date,
                current_weights.index,
            ]
            .fillna(0.0)
            .astype(float)
        )

        portfolio_return = float(
            (
                current_weights
                * realized
            ).sum()
        )

        denominator = (
            1.0
            + portfolio_return
        )

        if denominator <= 0:
            raise RuntimeError(
                f"Portfolio failed before event date: "
                f"{date} return={portfolio_return}"
            )

        current_weights = (
            current_weights
            * (
                1.0
                + realized
            )
            / denominator
        )

    held_event_returns = (
        event_returns
        .reindex(
            current_weights.index
        )
        .fillna(0.0)
    )

    contributions = (
        current_weights
        * held_event_returns
    )

    diagnostics = pd.DataFrame(
        {
            "weight_before_event":
                current_weights,
            "valuation_return":
                held_event_returns,
            "contribution":
                contributions,
        }
    )

    diagnostics["abs_contribution"] = (
        diagnostics["contribution"].abs()
    )

    diagnostics = diagnostics.sort_values(
        "abs_contribution",
        ascending=False,
    )

    print()
    print("=" * 100)
    print("TOP PORTFOLIO CONTRIBUTIONS ON 2021-04-13")
    print("=" * 100)

    print(
        diagnostics.to_string()
    )

    print()
    print(
        "Reconstructed event-day portfolio return:",
        float(contributions.sum()),
    )

    print()
    print("=" * 100)
    print("HELD ASSETS WITH |VALUATION RETURN| >= 100%")
    print("=" * 100)

    extreme = diagnostics[
        diagnostics[
            "valuation_return"
        ].abs() >= 1.0
    ]

    print(
        extreme.to_string()
        if not extreme.empty
        else "None"
    )


if __name__ == "__main__":
    main()
