from __future__ import annotations

from pathlib import Path

import pandas as pd

from enhanced_momentum.data_loaders.registry import load_market


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"
RUNS_DIR = PROJECT_ROOT / "results" / "baseline_grid_v3" / "runs"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline_grid_v3"
    / "moex_zero_signal_audit.csv"
)


def _previous_row_with_coverage(
    close: pd.DataFrame,
    endpoint: pd.Timestamp,
    assets: pd.Index,
) -> tuple[pd.Timestamp | pd.NaT, int, int]:
    pos = close.index.get_loc(endpoint)
    if not isinstance(pos, int):
        pos = int(pos)

    for candidate_pos in range(pos, -1, -1):
        row = close.iloc[candidate_pos].reindex(assets)
        count = int(row.notna().sum())
        if count > 0:
            candidate_date = pd.Timestamp(close.index[candidate_pos])
            gap = int((endpoint - candidate_date).days)
            return candidate_date, count, gap

    return pd.NaT, 0, -1


def main() -> None:
    print("Loading MOEX...")
    data = load_market(
        "moex",
        config_dir=MARKETS_DIR,
    )

    close = data.close.astype(float)
    presence = (
        data.presence_matrix
        .reindex(index=close.index, columns=close.columns)
        .fillna(0)
        .astype(bool)
    )

    records: list[dict[str, object]] = []

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

        for row in zero.itertuples(index=False):
            decision_date = pd.Timestamp(row.decision_date)
            effective_date = pd.Timestamp(row.effective_date)
            signal_start = pd.Timestamp(row.signal_start)
            signal_end = pd.Timestamp(row.signal_end)

            present_now = (
                presence.loc[decision_date]
                & close.loc[decision_date].notna()
            )
            assets = present_now[present_now].index

            p_start = close.loc[signal_start].reindex(assets)
            p_end = close.loc[signal_end].reindex(assets)

            n_start = int(p_start.notna().sum())
            n_end = int(p_end.notna().sum())
            n_both = int((p_start.notna() & p_end.notna()).sum())

            previous_start_date, previous_start_count, start_gap_days = (
                _previous_row_with_coverage(
                    close,
                    signal_start,
                    assets,
                )
            )
            previous_end_date, previous_end_count, end_gap_days = (
                _previous_row_with_coverage(
                    close,
                    signal_end,
                    assets,
                )
            )

            if n_start == 0 and n_end == 0:
                cause = "both_signal_endpoints_missing"
            elif n_start == 0:
                cause = "signal_start_missing"
            elif n_end == 0:
                cause = "signal_end_missing"
            elif n_both == 0:
                cause = "no_asset_has_both_endpoints"
            else:
                cause = "other"

            records.append(
                {
                    "run_id": run_dir.name,
                    "decision_date": decision_date,
                    "effective_date": effective_date,
                    "signal_start": signal_start,
                    "signal_end": signal_end,
                    "n_present": int(row.n_present),
                    "n_valid_scores": int(row.n_valid_scores),
                    "n_prices_at_signal_start": n_start,
                    "n_prices_at_signal_end": n_end,
                    "n_assets_with_both_prices": n_both,
                    "cause": cause,
                    "previous_start_date_with_any_price": previous_start_date,
                    "previous_start_price_count": previous_start_count,
                    "signal_start_gap_calendar_days": start_gap_days,
                    "previous_end_date_with_any_price": previous_end_date,
                    "previous_end_price_count": previous_end_count,
                    "signal_end_gap_calendar_days": end_gap_days,
                }
            )

    audit = pd.DataFrame(records)

    if audit.empty:
        print("No zero-position MOEX rebalances found.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 110)
    print("CAUSE COUNTS")
    print("=" * 110)
    print(
        audit["cause"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 110)
    print("ZERO-SIGNAL CASES BY RUN")
    print("=" * 110)
    columns = [
        "run_id",
        "decision_date",
        "effective_date",
        "signal_start",
        "signal_end",
        "n_present",
        "n_prices_at_signal_start",
        "n_prices_at_signal_end",
        "n_assets_with_both_prices",
        "cause",
        "previous_start_date_with_any_price",
        "signal_start_gap_calendar_days",
        "previous_end_date_with_any_price",
        "signal_end_gap_calendar_days",
    ]
    print(
        audit[columns]
        .sort_values(
            ["effective_date", "run_id"]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 110)
    print("UNIQUE PROBLEM ENDPOINTS")
    print("=" * 110)

    endpoints = pd.concat(
        [
            audit[
                [
                    "signal_start",
                    "n_prices_at_signal_start",
                    "previous_start_date_with_any_price",
                    "signal_start_gap_calendar_days",
                ]
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "signal_start": "endpoint",
                    "n_prices_at_signal_start": "n_prices",
                    "previous_start_date_with_any_price": "previous_date_with_any_price",
                    "signal_start_gap_calendar_days": "gap_calendar_days",
                }
            )
            .assign(endpoint_type="start"),
            audit[
                [
                    "signal_end",
                    "n_prices_at_signal_end",
                    "previous_end_date_with_any_price",
                    "signal_end_gap_calendar_days",
                ]
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "signal_end": "endpoint",
                    "n_prices_at_signal_end": "n_prices",
                    "previous_end_date_with_any_price": "previous_date_with_any_price",
                    "signal_end_gap_calendar_days": "gap_calendar_days",
                }
            )
            .assign(endpoint_type="end"),
        ],
        ignore_index=True,
    )

    endpoints = endpoints[endpoints["n_prices"] == 0]

    print(
        endpoints.sort_values(
            ["endpoint", "endpoint_type"]
        ).to_string(index=False)
        if not endpoints.empty
        else "None"
    )

    print()
    print(f"Saved full audit to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
