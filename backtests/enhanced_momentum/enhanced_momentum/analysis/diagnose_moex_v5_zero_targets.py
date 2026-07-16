from __future__ import annotations

from pathlib import Path

import pandas as pd

from enhanced_momentum.backtesting.cross_market_backtest import (
    market_observation_index,
)
from enhanced_momentum.data_loaders.registry import load_market


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"
RUNS_DIR = PROJECT_ROOT / "results" / "baseline_grid_v5" / "runs"


def main() -> None:
    data = load_market(
        "moex",
        config_dir=MARKETS_DIR,
    )

    close = data.close.astype(float)
    presence = (
        data.presence_matrix
        .reindex(index=close.index, columns=close.columns)
        .fillna(False)
        .astype(bool)
    )
    proxy = (
        pd.to_numeric(
            data.market_proxy_returns,
            errors="coerce",
        )
        .reindex(close.index)
    )
    session_index = market_observation_index(
        data,
        close=close,
    )
    session_set = set(session_index)

    zero_rows: list[pd.DataFrame] = []

    for run_dir in sorted(RUNS_DIR.glob("moex_*")):
        diag_path = run_dir / "rebalance_diagnostics.parquet"
        if not diag_path.exists():
            continue

        diag = pd.read_parquet(diag_path)
        zero = diag[
            (diag["n_long"] == 0)
            | (diag["n_short"] == 0)
        ].copy()

        if zero.empty:
            continue

        zero["run_id"] = run_dir.name
        zero_rows.append(zero)

    if not zero_rows:
        print("No zero-target MOEX rebalances found.")
        return

    zero = pd.concat(zero_rows, ignore_index=True)

    print("=" * 110)
    print("ZERO-TARGET REBALANCES")
    print("=" * 110)
    print(
        zero[
            [
                "run_id",
                "decision_date",
                "effective_date",
                "signal_start",
                "signal_end",
                "n_present",
                "n_valid_scores",
                "n_eligible",
                "n_long",
                "n_short",
            ]
        ]
        .sort_values(["effective_date", "run_id"])
        .to_string(index=False)
    )

    unique_cases = zero[
        [
            "decision_date",
            "effective_date",
            "signal_start",
            "signal_end",
        ]
    ].drop_duplicates()

    for case in unique_cases.itertuples(index=False):
        decision_date = pd.Timestamp(case.decision_date)
        effective_date = pd.Timestamp(case.effective_date)
        signal_start = pd.Timestamp(case.signal_start)
        signal_end = pd.Timestamp(case.signal_end)

        present_now = (
            presence.loc[decision_date]
            & close.loc[decision_date].notna()
        )
        assets = present_now[present_now].index

        start_prices = close.loc[signal_start].reindex(assets)
        end_prices = close.loc[signal_end].reindex(assets)

        print()
        print("-" * 110)
        print(f"Decision date: {decision_date.date()}")
        print(f"Effective date: {effective_date.date()}")
        print(f"Signal start: {signal_start.date()}")
        print(f"Signal end: {signal_end.date()}")
        print(f"Signal start is selected session: {signal_start in session_set}")
        print(f"Signal end is selected session: {signal_end in session_set}")
        print(f"Proxy return at signal start: {proxy.get(signal_start)}")
        print(f"Proxy return at signal end: {proxy.get(signal_end)}")
        print(
            "Market-wide close breadth at signal start:",
            int(close.loc[signal_start].notna().sum()),
        )
        print(
            "Market-wide close breadth at signal end:",
            int(close.loc[signal_end].notna().sum()),
        )
        print(
            "Current eligible/present assets:",
            len(assets),
        )
        print(
            "Current assets with start price:",
            int(start_prices.notna().sum()),
        )
        print(
            "Current assets with end price:",
            int(end_prices.notna().sum()),
        )
        print(
            "Current assets with both endpoint prices:",
            int(
                (
                    start_prices.notna()
                    & end_prices.notna()
                ).sum()
            ),
        )

        loc = close.index.get_loc(signal_start)
        if not isinstance(loc, int):
            loc = int(loc)

        lo = max(0, loc - 7)
        hi = min(len(close.index), loc + 8)
        nearby_index = close.index[lo:hi]

        nearby = pd.DataFrame(
            {
                "proxy_return": proxy.reindex(nearby_index),
                "close_breadth": (
                    close.reindex(nearby_index)
                    .notna()
                    .sum(axis=1)
                ),
                "selected_session": [
                    date in session_set
                    for date in nearby_index
                ],
            }
        )

        print()
        print("NEARBY CALENDAR AROUND SIGNAL START:")
        print(nearby.to_string())

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(
        "Affected runs:",
        zero["run_id"].nunique(),
    )
    print(
        "Unique zero-target rebalance cases:",
        len(unique_cases),
    )


if __name__ == "__main__":
    main()
