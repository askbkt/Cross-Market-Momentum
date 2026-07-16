from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enhanced_momentum.backtesting.cross_market_backtest import (
    CrossMarketBacktestConfig,
    config_to_dict,
    market_observation_index,
    run_cross_market_backtest,
)
from enhanced_momentum.data_loaders.registry import load_market


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_grid_v5"

ENGINE_VERSION = "unified_cross_market_baseline_v4_proxy_sessions"
QUANTILES = (0.10, 0.20, 0.30)

MARKET_HORIZONS = {
    "us": {
        "lookbacks": {"6M": 126, "12M": 252, "24M": 504},
        "skips": {"0M": 0, "1M": 21, "3M": 63},
    },
    "moex": {
        "lookbacks": {"6M": 126, "12M": 252, "24M": 504},
        "skips": {"0M": 0, "1M": 21, "3M": 63},
    },
    "crypto": {
        "lookbacks": {"6M": 182, "12M": 365, "24M": 730},
        "skips": {"0M": 0, "1M": 30, "3M": 91},
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _month_end_decision_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = (
        pd.Series(index, index=index)
        .groupby(index.to_period("M"))
        .last()
        .tolist()
    )
    return pd.DatetimeIndex(dates)


def _common_evaluation_start(
    observation_index: pd.DatetimeIndex,
    *,
    lookbacks: list[int],
    skips: list[int],
) -> pd.Timestamp:
    """First effective market session feasible for every grid config."""
    max_required_position = max(
        lookback_days + skip_days - 1
        for lookback_days, skip_days in product(lookbacks, skips)
    )

    for decision_date in _month_end_decision_dates(observation_index):
        observation_location = observation_index.get_loc(decision_date)

        if not isinstance(
            observation_location,
            (int, np.integer),
        ):
            raise RuntimeError(
                "Unexpected non-scalar month-end observation location."
            )

        observation_pos = int(observation_location)
        next_session_pos = observation_pos + 1

        if (
            observation_pos >= max_required_position
            and next_session_pos < len(observation_index)
        ):
            return pd.Timestamp(
                observation_index[next_session_pos]
            )

    raise ValueError(
        "Could not find a common evaluation start for the full parameter grid."
    )


def _run_id(
    market: str,
    lookback_label: str,
    skip_label: str,
    quantile: float,
) -> str:
    quantile_pct = int(round(quantile * 100))
    return (
        f"{market}_lb{lookback_label}_sk{skip_label}_q{quantile_pct:02d}"
    )


def _save_series_parquet(
    series: pd.Series,
    path: Path,
    *,
    value_name: str,
) -> None:
    frame = series.rename(value_name).to_frame()
    frame.index.name = "date"
    frame.to_parquet(path)


def _grid_specs(market: str) -> list[dict[str, Any]]:
    horizons = MARKET_HORIZONS[market]
    specs: list[dict[str, Any]] = []

    for (
        lookback_label,
        lookback_days,
    ), (
        skip_label,
        skip_days,
    ), quantile in product(
        horizons["lookbacks"].items(),
        horizons["skips"].items(),
        QUANTILES,
    ):
        specs.append(
            {
                "market": market,
                "lookback_label": lookback_label,
                "lookback_days": int(lookback_days),
                "skip_label": skip_label,
                "skip_days": int(skip_days),
                "quantile": float(quantile),
            }
        )

    return specs


def _engine_provenance() -> dict[str, Any]:
    source_path_raw = inspect.getsourcefile(run_cross_market_backtest)
    if source_path_raw is None:
        return {
            "engine_version": ENGINE_VERSION,
            "engine_source_path": None,
            "engine_source_sha256": None,
        }

    source_path = Path(source_path_raw).resolve()
    return {
        "engine_version": ENGINE_VERSION,
        "engine_source_path": str(source_path),
        "engine_source_sha256": (
            _sha256_file(source_path) if source_path.exists() else None
        ),
    }


def run_market_grid(
    market: str,
    *,
    output_dir: Path,
    overwrite: bool,
    requested_start: pd.Timestamp | None,
    requested_end: pd.Timestamp | None,
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print()
    print("=" * 96)
    print(f"BASELINE GRID: {market.upper()}")
    print("=" * 96)

    print("Loading MarketData once for all 27 configurations...")
    data = load_market(market, config_dir=MARKETS_DIR)
    print(data.summary())

    index = pd.DatetimeIndex(data.close.index)
    observation_index = market_observation_index(
        data,
        close=data.close,
    )

    horizons = MARKET_HORIZONS[market]

    auto_start = _common_evaluation_start(
        observation_index,
        lookbacks=list(horizons["lookbacks"].values()),
        skips=list(horizons["skips"].values()),
    )

    evaluation_start = (
        max(auto_start, requested_start)
        if requested_start is not None
        else auto_start
    )

    evaluation_end = pd.Timestamp(index.max())
    if requested_end is not None:
        evaluation_end = min(evaluation_end, requested_end)

    if evaluation_start > evaluation_end:
        raise ValueError(
            f"{market}: evaluation_start={evaluation_start.date()} "
            f"is after evaluation_end={evaluation_end.date()}."
        )

    print()
    print(
        "Common evaluation window for all 27 configurations: "
        f"{evaluation_start.date()} -> {evaluation_end.date()}"
    )
    print(
        "Auto common start from max lookback+skip requirement: "
        f"{auto_start.date()}"
    )

    config_path = MARKETS_DIR / f"{market}.yaml"
    _write_json(
        output_dir / "provenance" / f"{market}.json",
        {
            "market": market,
            "data_start": pd.Timestamp(index.min()).date().isoformat(),
            "data_end": pd.Timestamp(index.max()).date().isoformat(),
            "n_data_dates": int(len(index)),
            "n_market_observation_dates": int(len(observation_index)),
            "n_market_wide_missing_dates": int(
                len(index) - len(observation_index)
            ),
            "n_assets": int(data.close.shape[1]),
            "evaluation_start": evaluation_start.date().isoformat(),
            "evaluation_end": evaluation_end.date().isoformat(),
            "auto_common_start": auto_start.date().isoformat(),
            "market_config_path": str(config_path),
            "market_config_sha256": (
                _sha256_file(config_path) if config_path.exists() else None
            ),
            **_engine_provenance(),
        },
    )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run_number, spec in enumerate(_grid_specs(market), start=1):
        run_id = _run_id(
            market,
            spec["lookback_label"],
            spec["skip_label"],
            spec["quantile"],
        )

        run_dir = output_dir / "runs" / run_id
        metrics_path = run_dir / "metrics.json"
        config_json_path = run_dir / "config.json"

        print()
        print("-" * 96)
        print(
            f"[{run_number:02d}/27] {run_id} | "
            f"lookback={spec['lookback_label']} "
            f"({spec['lookback_days']} obs), "
            f"skip={spec['skip_label']} "
            f"({spec['skip_days']} obs), "
            f"q={spec['quantile']:.0%}"
        )

        if (
            metrics_path.exists()
            and config_json_path.exists()
            and not overwrite
        ):
            print("Existing completed run found; loading metrics and skipping.")
            records.append(_read_json(metrics_path))
            continue

        config = CrossMarketBacktestConfig(
            lookback_days=spec["lookback_days"],
            skip_days=spec["skip_days"],
            quantile=spec["quantile"],
            rebal_freq="ME",
            gross_exposure=1.0,
            transaction_cost_bps=0.0,
        )

        _write_json(
            config_json_path,
            {
                "run_id": run_id,
                "market": market,
                "lookback_label": spec["lookback_label"],
                "skip_label": spec["skip_label"],
                "evaluation_start": evaluation_start.date().isoformat(),
                "evaluation_end": evaluation_end.date().isoformat(),
                "engine_version": ENGINE_VERSION,
                "backtest_config": config_to_dict(config),
                "research_role": (
                    "baseline robustness-map configuration; "
                    "not a hyperparameter-selection candidate"
                ),
            },
        )

        started = time.perf_counter()

        try:
            result = run_cross_market_backtest(
                data=data,
                config=config,
                start_date=evaluation_start,
                end_date=evaluation_end,
                store_holdings=False,
            )

            runtime_seconds = time.perf_counter() - started
            metrics = result.metrics().to_dict()

            record = {
                "run_id": run_id,
                "market": market,
                "lookback_label": spec["lookback_label"],
                "lookback_days": spec["lookback_days"],
                "skip_label": spec["skip_label"],
                "skip_days": spec["skip_days"],
                "quantile": spec["quantile"],
                "evaluation_start": evaluation_start.date().isoformat(),
                "evaluation_end": evaluation_end.date().isoformat(),
                "runtime_seconds": runtime_seconds,
                **metrics,
            }

            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(metrics_path, record)

            _save_series_parquet(
                result.daily_returns,
                run_dir / "daily_returns.parquet",
                value_name="strategy_return",
            )
            _save_series_parquet(
                result.turnover,
                run_dir / "turnover.parquet",
                value_name="turnover",
            )
            _save_series_parquet(
                result.daily_gross_exposure,
                run_dir / "daily_gross_exposure.parquet",
                value_name="daily_gross_exposure",
            )
            _save_series_parquet(
                result.daily_net_exposure,
                run_dir / "daily_net_exposure.parquet",
                value_name="daily_net_exposure",
            )
            result.rebalance_diagnostics.to_parquet(
                run_dir / "rebalance_diagnostics.parquet",
                index=False,
            )

            records.append(record)

            print(
                "Completed | "
                f"Sharpe={float(record['sharpe']):.4f} | "
                f"AnnRet={float(record['annualized_return']):.2%} | "
                f"MaxDD={float(record['max_drawdown']):.2%} | "
                f"runtime={runtime_seconds:.1f}s"
            )

        except Exception as exc:
            runtime_seconds = time.perf_counter() - started
            failure = {
                "run_id": run_id,
                "market": market,
                "lookback_label": spec["lookback_label"],
                "lookback_days": spec["lookback_days"],
                "skip_label": spec["skip_label"],
                "skip_days": spec["skip_days"],
                "quantile": spec["quantile"],
                "runtime_seconds": runtime_seconds,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            failures.append(failure)

            print(f"FAILED | {type(exc).__name__}: {exc}")

            if fail_fast:
                raise

        finally:
            gc.collect()

    return records, failures


def _write_summary(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> pd.DataFrame:
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary

    market_order = {"us": 0, "moex": 1, "crypto": 2}
    lookback_order = {"6M": 0, "12M": 1, "24M": 2}
    skip_order = {"0M": 0, "1M": 1, "3M": 2}

    summary["_market_order"] = summary["market"].map(market_order).fillna(999)
    summary["_lookback_order"] = (
        summary["lookback_label"].map(lookback_order).fillna(999)
    )
    summary["_skip_order"] = (
        summary["skip_label"].map(skip_order).fillna(999)
    )

    summary = (
        summary
        .sort_values(
            [
                "_market_order",
                "_lookback_order",
                "_skip_order",
                "quantile",
            ]
        )
        .drop(
            columns=[
                "_market_order",
                "_lookback_order",
                "_skip_order",
            ]
        )
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summary.to_parquet(output_dir / "summary.parquet", index=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 27-config-per-market baseline robustness grid "
            "through the unified cross-market momentum engine."
        )
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["us", "moex", "crypto"],
        choices=["us", "moex", "crypto"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun configurations even when completed metrics already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed configuration.",
    )
    parser.add_argument(
        "--start-date",
        type=pd.Timestamp,
        default=None,
        help=(
            "Optional lower bound. The common feasible start required by "
            "the largest lookback+skip is still enforced."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=pd.Timestamp,
        default=None,
        help="Optional upper bound for the evaluation period.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_dir / "manifest.json",
        {
            "engine_version": ENGINE_VERSION,
            "engine_provenance": _engine_provenance(),
            "markets": args.markets,
            "quantiles": list(QUANTILES),
            "market_horizons": MARKET_HORIZONS,
            "gross_exposure": 1.0,
            "transaction_cost_bps": 0.0,
            "portfolio_accounting": "fixed_notional_sleeves",
            "signal_calendar": "market_proxy_confirmed_sessions",
            "effective_date_rule": "next_market_session",
            "rebal_freq": "ME",
            "sample_policy": (
                "Within each market, every grid configuration uses one common "
                "evaluation window. The default start is the first effective "
                "monthly rebalance date feasible for the largest lookback+skip "
                "requirement. Markets may have different evaluation windows."
            ),
            "research_role": (
                "The grid is a robustness map. It must not be used to select "
                "the single best full-sample Sharpe configuration."
            ),
        },
    )

    all_records: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    for market in args.markets:
        records, failures = run_market_grid(
            market,
            output_dir=output_dir,
            overwrite=args.overwrite,
            requested_start=args.start_date,
            requested_end=args.end_date,
            fail_fast=args.fail_fast,
        )

        all_records.extend(records)
        all_failures.extend(failures)

        summary = _write_summary(
            all_records,
            output_dir=output_dir,
        )
        print()
        print(
            f"Checkpoint summary written: "
            f"{output_dir / 'summary.csv'} "
            f"({len(summary)} completed runs)"
        )

    if all_failures:
        failures_df = pd.DataFrame(all_failures)
        failures_df.to_csv(
            output_dir / "failures.csv",
            index=False,
        )

        print()
        print("=" * 96)
        print(f"GRID COMPLETED WITH {len(all_failures)} FAILURE(S)")
        print(f"See: {output_dir / 'failures.csv'}")
        print("=" * 96)

        raise RuntimeError("One or more baseline grid runs failed.")

    expected_runs = 27 * len(args.markets)
    summary = _write_summary(all_records, output_dir=output_dir)

    if len(summary) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} completed runs, "
            f"but summary contains {len(summary)}."
        )

    failures_path = output_dir / "failures.csv"
    if failures_path.exists():
        failures_path.unlink()

    print()
    print("=" * 96)
    print("BASELINE GRID COMPLETED SUCCESSFULLY")
    print(f"Completed runs: {len(summary)} / {expected_runs}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Summary Parquet: {output_dir / 'summary.parquet'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
