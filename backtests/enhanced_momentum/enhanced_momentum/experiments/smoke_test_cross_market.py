from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from enhanced_momentum.backtesting.cross_market_backtest import (
    CrossMarketBacktestConfig,
    run_cross_market_backtest,
)
from enhanced_momentum.data_loaders.registry import load_market


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"

# Same economic baseline idea: ~12-month lookback, ~1-month skip, 10% tails.
# Observation counts differ because crypto trades 365 days/year.
SMOKE_SPECS = {
    "us": {
        "lookback_days": 252,
        "skip_days": 21,
        "quantile": 0.10,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    },
    "moex": {
        "lookback_days": 252,
        "skip_days": 21,
        "quantile": 0.10,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    },
    "crypto": {
        "lookback_days": 365,
        "skip_days": 30,
        "quantile": 0.10,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    },
}


def _assert_smoke_invariants(result) -> None:
    returns = result.daily_returns
    nav = result.nav
    turnover = result.turnover
    gross_exposure = result.daily_gross_exposure
    net_exposure = result.daily_net_exposure
    diag = result.rebalance_diagnostics

    assert not returns.empty, "daily_returns is empty"
    assert returns.index.is_monotonic_increasing, "daily_returns index is not sorted"
    assert not returns.index.has_duplicates, "daily_returns index has duplicates"
    assert np.isfinite(returns.to_numpy()).all(), "daily_returns contains NaN/inf"

    assert not nav.empty, "nav is empty"
    assert np.isfinite(nav.to_numpy()).all(), "nav contains NaN/inf"
    assert (nav > 0).all(), "NAV became non-positive"

    assert (turnover >= 0).all(), "turnover contains negative values"
    assert not diag.empty, "rebalance diagnostics are empty"

    assert (
        pd.to_datetime(diag["effective_date"])
        > pd.to_datetime(diag["decision_date"])
    ).all(), "effective_date must be strictly after decision_date"

    assert (
        pd.to_datetime(diag["signal_end"])
        <= pd.to_datetime(diag["decision_date"])
    ).all(), "signal_end is after decision_date"

    nonempty = diag[(diag["n_long"] > 0) | (diag["n_short"] > 0)]
    if not nonempty.empty:
        assert (
            nonempty["n_long"] == nonempty["n_short"]
        ).all(), "long and short counts differ"

    assert (
        result.stale_held_gross_exposure >= 0
    ).all(), "stale held exposure contains negative values"

    invested = gross_exposure > 0
    if invested.any():
        assert np.allclose(
            gross_exposure[invested].to_numpy(dtype=float),
            result.config.gross_exposure,
            atol=1e-10,
            rtol=0.0,
        ), "fixed-sleeve gross exposure drifted"
        assert np.allclose(
            net_exposure[invested].to_numpy(dtype=float),
            0.0,
            atol=1e-10,
            rtol=0.0,
        ), "fixed-sleeve net exposure drifted"

    assert (returns > -1.0).all(), (
        "daily return reached or crossed -100%"
    )

    recoveries = result.stale_recovery_events
    if not recoveries.empty:
        assert np.isfinite(
            recoveries["valuation_return"].to_numpy(dtype=float)
        ).all(), "stale recovery returns contain NaN/inf"
        assert np.isfinite(
            recoveries[
                "weighted_return_contribution"
            ].to_numpy(dtype=float)
        ).all(), "stale recovery contributions contain NaN/inf"


def run_one_market(market: str) -> None:
    spec = SMOKE_SPECS[market]

    print()
    print("=" * 88)
    print(f"SMOKE TEST: {market.upper()}")
    print("=" * 88)

    print("Loading MarketData...")
    data = load_market(market, config_dir=MARKETS_DIR)
    print(data.summary())

    config = CrossMarketBacktestConfig(
        lookback_days=spec["lookback_days"],
        skip_days=spec["skip_days"],
        quantile=spec["quantile"],
        rebal_freq="ME",
        gross_exposure=1.0,
        transaction_cost_bps=0.0,
    )

    print()
    print(
        "Running unified backtest: "
        f"lookback={config.lookback_days}, "
        f"skip={config.skip_days}, "
        f"q={config.quantile:.0%}, "
        f"period={spec['start_date']} -> {spec['end_date']}"
    )

    result = run_cross_market_backtest(
        data=data,
        config=config,
        start_date=spec["start_date"],
        end_date=spec["end_date"],
        store_holdings=True,
    )

    _assert_smoke_invariants(result)

    metrics = result.metrics()
    diag = result.rebalance_diagnostics

    print()
    print("Key metrics:")
    for key in [
        "start_date",
        "end_date",
        "n_days",
        "n_rebalances",
        "total_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "total_turnover",
        "avg_n_eligible",
        "avg_n_long",
        "avg_n_short",
        "min_n_long",
        "min_n_short",
        "min_daily_gross_exposure",
        "max_daily_gross_exposure",
        "max_abs_daily_net_exposure",
        "days_with_stale_held_prices",
        "avg_stale_held_gross_exposure",
        "max_stale_held_gross_exposure",
        "n_stale_recovery_events",
        "max_abs_stale_recovery_return",
        "max_abs_stale_recovery_contribution",
    ]:
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print()
    print("First 3 rebalance diagnostics:")
    print(diag.head(3).to_string(index=False))

    print()
    print("Last 3 rebalance diagnostics:")
    print(diag.tail(3).to_string(index=False))

    if result.holdings is not None and not result.holdings.empty:
        gross_by_rebalance = (
            result.holdings.groupby("effective_date")["weight"]
            .agg(
                gross=lambda x: x.abs().sum(),
                net="sum",
                n="size",
            )
        )
        print()
        print("Target portfolio checks (first 3 effective dates):")
        print(gross_by_rebalance.head(3).to_string())

        gross_error = (gross_by_rebalance["gross"] - 1.0).abs().max()
        net_error = gross_by_rebalance["net"].abs().max()

        assert gross_error < 1e-10, (
            f"Target gross exposure deviates from 1.0; max error={gross_error}"
        )
        assert net_error < 1e-10, (
            f"Target net exposure deviates from 0.0; max error={net_error}"
        )

    stale_events = result.stale_price_events.copy()

    if not stale_events.empty:
        print()
        print("Stale-price diagnostics:")
        print(f"  n_events: {len(stale_events)}")
        print(
            "  n_days: "
            f"{stale_events['date'].nunique()}"
        )
        print(
            "  n_assets: "
            f"{stale_events['asset'].nunique()}"
        )

        presence_counts = (
            stale_events["presence_on_date"]
            .value_counts(dropna=False)
            .rename_axis("presence_on_date")
            .to_frame("count")
        )

        print()
        print("Stale-price events by presence flag:")
        print(presence_counts.to_string())

        print()
        print("Staleness duration summary (calendar days):")
        print(
            stale_events[
                "calendar_days_since_last_observed_close"
            ]
            .describe(
                percentiles=[
                    0.50,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        print()
        print(
            "Top 20 stale-price events "
            "by absolute portfolio weight:"
        )
        print(
            stale_events.nlargest(
                20,
                "abs_weight",
            ).to_string(index=False)
        )

        print()
        print(
            "Top 20 longest stale-price events:"
        )
        print(
            stale_events.nlargest(
                20,
                "calendar_days_since_last_observed_close",
            ).to_string(index=False)
        )

    recovery_events = result.stale_recovery_events.copy()

    if not recovery_events.empty:
        abs_returns = recovery_events[
            "valuation_return"
        ].abs()

        print()
        print("Stale-price recovery diagnostics:")
        print(
            f"  n_recovery_events: "
            f"{len(recovery_events)}"
        )
        print(
            f"  n_assets: "
            f"{recovery_events['asset'].nunique()}"
        )
        print(
            "  recoveries_with_|return|_>_50%: "
            f"{int((abs_returns > 0.50).sum())}"
        )
        print(
            "  recoveries_with_|return|_>_100%: "
            f"{int((abs_returns > 1.00).sum())}"
        )

        print()
        print("Recovery valuation-return summary:")
        print(
            recovery_events["valuation_return"]
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        print()
        print("Absolute recovery-return summary:")
        print(
            abs_returns
            .describe(
                percentiles=[
                    0.50,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        print()
        print(
            "Weighted recovery-contribution summary:"
        )
        print(
            recovery_events[
                "weighted_return_contribution"
            ]
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

        print()
        print(
            "Top 20 recoveries by absolute valuation return:"
        )
        print(
            recovery_events.loc[
                abs_returns.nlargest(20).index
            ]
            .sort_values(
                "valuation_return",
                key=lambda s: s.abs(),
                ascending=False,
            )
            .to_string(index=False)
        )

        print()
        print(
            "Top 20 recoveries by absolute portfolio contribution:"
        )
        print(
            recovery_events.nlargest(
                20,
                "abs_weighted_return_contribution",
            ).to_string(index=False)
        )

    print()
    print(f"{market.upper()} SMOKE TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the unified cross-market momentum backtest."
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["moex", "crypto", "us"],
        choices=["us", "moex", "crypto"],
    )
    args = parser.parse_args()

    for market in args.markets:
        run_one_market(market)

    print()
    print("=" * 88)
    print("ALL REQUESTED SMOKE TESTS PASSED")
    print("=" * 88)


if __name__ == "__main__":
    main()
