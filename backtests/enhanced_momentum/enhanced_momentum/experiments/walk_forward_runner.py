from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from enhanced_momentum.data_loaders.registry import get_loader, load_market_config


MARKETS = ("us", "moex", "crypto")


@dataclass(frozen=True)
class Fold:
    market: str
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repository root")


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_protocol(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 3 protocol not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError(f"Invalid protocol file: {path}")
    return protocol


def protocol_folds(protocol: dict[str, Any]) -> list[Fold]:
    folds: list[Fold] = []
    for market in MARKETS:
        for raw in protocol["folds"][market]:
            folds.append(
                Fold(
                    market=market,
                    name=str(raw["name"]),
                    train_start=str(raw["train_start"]),
                    train_end=str(raw["train_end"]),
                    test_start=str(raw["test_start"]),
                    test_end=str(raw["test_end"]),
                )
            )
    return folds


def read_series(path: Path, expected_column: str) -> pd.Series:
    frame = pd.read_parquet(path)
    if expected_column in frame.columns:
        series = frame[expected_column]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        raise ValueError(
            f"{path}: expected {expected_column!r}; found {frame.columns.tolist()}"
        )
    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce").sort_index()
    series.name = expected_column
    return series


def known_exclusions(protocol: dict[str, Any], market: str) -> dict[str, str]:
    raw = protocol.get("known_exclusions", {}).get(market, {})
    return {str(key): str(value) for key, value in raw.items()}


def load_runs(
    runs_dir: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    primary_tc = {
        market: float(protocol["primary_transaction_cost_bps"][market])
        for market in MARKETS
    }
    runs: dict[str, dict[str, Any]] = {}

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        returns_path = run_dir / "daily_returns.parquet"
        turnover_path = run_dir / "turnover.parquet"
        if not (config_path.exists() and returns_path.exists() and turnover_path.exists()):
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        market = str(config["market"])
        run_id = str(config["run_id"])
        if market not in MARKETS:
            continue

        gross = read_series(returns_path, "strategy_return")
        turnover = read_series(turnover_path, "turnover")
        common = gross.index.intersection(turnover.index).sort_values()
        gross = gross.reindex(common)
        turnover = turnover.reindex(common).fillna(0.0)

        if gross.isna().any():
            raise ValueError(f"{run_id}: gross returns contain missing values")
        if (turnover < 0.0).any():
            raise ValueError(f"{run_id}: turnover contains negative values")

        net = gross - turnover * primary_tc[market] / 10_000.0
        exclusion_reason = known_exclusions(protocol, market).get(run_id, "")

        runs[run_id] = {
            "market": market,
            "run_id": run_id,
            "config": config,
            "gross": gross.astype(float),
            "turnover": turnover.astype(float),
            "net_primary": net.astype(float),
            "selection_eligible": not bool(exclusion_reason),
            "exclusion_reason": exclusion_reason,
        }

    counts = pd.Series([item["market"] for item in runs.values()]).value_counts().to_dict()
    expected = {"us": 27, "moex": 27, "crypto": 27}
    if counts != expected:
        raise RuntimeError(f"Expected 27 runs per market, got {counts}")
    return runs


def resolved_market_config(
    root: Path,
    protocol: dict[str, Any],
    market: str,
) -> dict[str, Any]:
    config_dir = root / str(
        protocol.get("markets_dir", "markets")
    )
    config = dict(
        load_market_config(
            market,
            config_dir=config_dir,
        )
    )

    research_root = root / str(
        protocol.get(
            "research_root",
            "backtests/enhanced_momentum",
        )
    )

    # ------------------------------------------------------------
    # Input source directory
    # ------------------------------------------------------------
    #
    # US reads the supervisor dataset from:
    #   <repo>/data/datasets
    #
    # source_dir is an INPUT directory and must already exist.
    source_dir = config.get("source_dir")
    if source_dir:
        source_path = Path(str(source_dir))

        if not source_path.is_absolute():
            candidates = [
                (root / source_path).resolve(),
                (research_root / source_path).resolve(),
            ]

            resolved_source = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.exists()
                ),
                None,
            )

            if resolved_source is None:
                attempted = "\n".join(
                    f"  - {candidate}"
                    for candidate in candidates
                )
                raise FileNotFoundError(
                    f"Cannot resolve input source_dir for "
                    f"{market}: {source_dir!r}. Tried:\n"
                    f"{attempted}"
                )

            config["source_dir"] = str(resolved_source)
        else:
            resolved_source = source_path.resolve()

            if not resolved_source.exists():
                raise FileNotFoundError(
                    f"Configured source_dir does not exist: "
                    f"{resolved_source}"
                )

            config["source_dir"] = str(resolved_source)

    # ------------------------------------------------------------
    # Market data/cache directory
    # ------------------------------------------------------------
    #
    # US data_dir is an output/cache location and does not need to
    # exist before loading the supervisor dataset.
    #
    # MOEX and Crypto load their already-downloaded cached datasets
    # from backtests/enhanced_momentum/data/{market}.
    data_dir = config.get("data_dir")
    if data_dir:
        data_path = Path(str(data_dir))

        if data_path.is_absolute():
            resolved_data = data_path.resolve()
        elif market == "us":
            resolved_data = (root / data_path).resolve()
        else:
            candidates = [
                (research_root / data_path).resolve(),
                (root / data_path).resolve(),
            ]

            resolved_data = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.exists()
                ),
                None,
            )

            if resolved_data is None:
                attempted = "\n".join(
                    f"  - {candidate}"
                    for candidate in candidates
                )
                raise FileNotFoundError(
                    f"Cannot resolve cached data_dir for "
                    f"{market}: {data_dir!r}. Tried:\n"
                    f"{attempted}"
                )

        config["data_dir"] = str(resolved_data)

    return config


def load_market_proxies(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, pd.Series]:
    proxies: dict[str, pd.Series] = {}
    for market in MARKETS:
        config = resolved_market_config(root, protocol, market)
        loader = get_loader(market, config)
        data = loader.load()
        proxy = pd.to_numeric(
            data.market_proxy_returns,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        proxy.index = pd.to_datetime(proxy.index)
        proxy = proxy.sort_index()
        proxy.name = "benchmark_return"
        if proxy.dropna().empty:
            raise RuntimeError(f"Market proxy is empty for {market}")
        proxies[market] = proxy
    return proxies


def clean_returns(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if (result <= -1.0).any():
        bad = result[result <= -1.0]
        raise ValueError(
            "Returns contain values <= -100%; "
            f"first observations: {bad.head().to_dict()}"
        )
    return result.astype(float)


def max_drawdown(returns: pd.Series) -> float:
    r = clean_returns(returns)
    if r.empty:
        return math.nan
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def annualized_return(returns: pd.Series, annualization: int) -> float:
    r = clean_returns(returns)
    if r.empty:
        return math.nan
    growth = float((1.0 + r).prod())
    years = len(r) / annualization
    if growth <= 0.0 or years <= 0.0:
        return math.nan
    return float(growth ** (1.0 / years) - 1.0)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization: int,
) -> float:
    aligned = pd.concat(
        [
            clean_returns(strategy_returns).rename("strategy"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return math.nan
    active = aligned["strategy"] - aligned["benchmark"]
    std = float(active.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return math.nan
    return float(active.mean() / std * np.sqrt(annualization))


def correlation_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    aligned = pd.concat(
        [
            clean_returns(strategy_returns).rename("strategy"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return math.nan, math.nan
    correlation = float(aligned.corr().iloc[0, 1])
    variance = float(aligned["benchmark"].var(ddof=1))
    beta = (
        float(aligned[["strategy", "benchmark"]].cov().iloc[0, 1]) / variance
        if np.isfinite(variance) and variance > 0.0
        else math.nan
    )
    return correlation, beta


def compute_metrics(
    returns: pd.Series,
    *,
    annualization: int,
    turnover: pd.Series | None = None,
) -> dict[str, float | int | str]:
    r = clean_returns(returns)
    if r.empty:
        return {
            "n_days": 0,
            "start": "",
            "end": "",
            "total_return": math.nan,
            "annualized_return": math.nan,
            "annualized_volatility": math.nan,
            "sharpe": math.nan,
            "sortino": math.nan,
            "max_drawdown": math.nan,
            "calmar": math.nan,
            "hit_rate": math.nan,
            "annualized_turnover": math.nan,
        }

    std = float(r.std(ddof=1)) if len(r) > 1 else math.nan
    annualized_volatility = (
        std * np.sqrt(annualization)
        if np.isfinite(std)
        else math.nan
    )
    sharpe = (
        float(r.mean()) / std * np.sqrt(annualization)
        if np.isfinite(std) and std > 0.0
        else math.nan
    )
    downside = r[r < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else math.nan
    sortino = (
        float(r.mean()) / downside_std * np.sqrt(annualization)
        if np.isfinite(downside_std) and downside_std > 0.0
        else math.nan
    )
    ann_return = annualized_return(r, annualization)
    drawdown = max_drawdown(r)
    calmar = (
        ann_return / abs(drawdown)
        if np.isfinite(ann_return) and np.isfinite(drawdown) and drawdown < 0.0
        else math.nan
    )
    annualized_turnover = math.nan
    if turnover is not None:
        t = pd.to_numeric(turnover.reindex(r.index), errors="coerce").fillna(0.0)
        annualized_turnover = float(t.mean() * annualization)

    return {
        "n_days": int(len(r)),
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "total_return": float((1.0 + r).prod() - 1.0),
        "annualized_return": ann_return,
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": drawdown,
        "calmar": float(calmar),
        "hit_rate": float((r > 0.0).mean()),
        "annualized_turnover": annualized_turnover,
    }


def market_run_ids(
    runs: dict[str, dict[str, Any]],
    market: str,
    *,
    eligible_only: bool = False,
) -> list[str]:
    ids = [
        run_id
        for run_id, item in runs.items()
        if item["market"] == market
        and (item["selection_eligible"] or not eligible_only)
    ]
    return sorted(ids)


def common_period_index(
    runs: dict[str, dict[str, Any]],
    run_ids: Iterable[str],
    start: str,
    end: str,
) -> pd.DatetimeIndex:
    common: pd.DatetimeIndex | None = None
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for run_id in run_ids:
        series = runs[run_id]["net_primary"].loc[start_ts:end_ts].dropna()
        index = pd.DatetimeIndex(series.index)
        common = index if common is None else common.intersection(index)
    if common is None:
        return pd.DatetimeIndex([])
    return common.sort_values()


def fold_long_results(
    runs: dict[str, dict[str, Any]],
    proxies: dict[str, pd.Series],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    annualization = {
        market: int(protocol["annualization_days"][market])
        for market in MARKETS
    }
    minimum_train = int(protocol["selection"]["min_train_observations"])
    minimum_test = int(protocol["selection"]["min_test_observations"])
    rows: list[dict[str, Any]] = []

    for fold in protocol_folds(protocol):
        eligible_ids = market_run_ids(runs, fold.market, eligible_only=True)
        all_ids = market_run_ids(runs, fold.market, eligible_only=False)
        samples = {
            "train": (fold.train_start, fold.train_end, minimum_train),
            "test": (fold.test_start, fold.test_end, minimum_test),
        }

        for sample, (start, end, minimum) in samples.items():
            common_index = common_period_index(
                runs,
                eligible_ids,
                start,
                end,
            )
            if len(common_index) < minimum:
                raise RuntimeError(
                    f"{fold.market}/{fold.name}/{sample}: only {len(common_index)} "
                    f"common observations; minimum is {minimum}. "
                    "Inspect baseline run start dates before changing the frozen protocol."
                )

            benchmark = proxies[fold.market].reindex(common_index)
            actual_start = str(common_index.min().date())
            actual_end = str(common_index.max().date())

            for run_id in all_ids:
                item = runs[run_id]
                gross = item["gross"].reindex(common_index)
                net = item["net_primary"].reindex(common_index)
                turnover = item["turnover"].reindex(common_index)
                if gross.isna().any() or net.isna().any() or turnover.isna().any():
                    raise RuntimeError(
                        f"{run_id}/{fold.name}/{sample}: missing values on common index"
                    )

                gross_metrics = compute_metrics(
                    gross,
                    annualization=annualization[fold.market],
                    turnover=turnover,
                )
                net_metrics = compute_metrics(
                    net,
                    annualization=annualization[fold.market],
                    turnover=turnover,
                )
                corr, beta = correlation_beta(net, benchmark)

                rows.append(
                    {
                        "market": fold.market,
                        "fold": fold.name,
                        "sample": sample,
                        "run_id": run_id,
                        "selection_eligible": bool(item["selection_eligible"]),
                        "exclusion_reason": item["exclusion_reason"],
                        "planned_start": start,
                        "planned_end": end,
                        "actual_common_start": actual_start,
                        "actual_common_end": actual_end,
                        "common_n_days": int(len(common_index)),
                        **{f"gross_{key}": value for key, value in gross_metrics.items()},
                        **{f"net_{key}": value for key, value in net_metrics.items()},
                        "net_information_ratio_vs_benchmark": information_ratio(
                            net,
                            benchmark,
                            annualization[fold.market],
                        ),
                        "net_correlation_to_benchmark": corr,
                        "net_beta_to_benchmark": beta,
                        "transaction_cost_annualized_return_drag": (
                            float(gross_metrics["annualized_return"])
                            - float(net_metrics["annualized_return"])
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    result["rank_all"] = result.groupby(
        ["market", "fold", "sample"]
    )["net_sharpe"].rank(ascending=False, method="min")
    result["rank_eligible"] = np.nan

    for _, group in result.groupby(["market", "fold", "sample"]):
        eligible_index = group.index[group["selection_eligible"]]
        result.loc[eligible_index, "rank_eligible"] = result.loc[
            eligible_index, "net_sharpe"
        ].rank(ascending=False, method="min")

    return result.sort_values(
        ["market", "fold", "sample", "rank_all", "run_id"]
    ).reset_index(drop=True)


def wide_walk_forward_results(
    long_results: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    keys = [
        "market",
        "fold",
        "run_id",
        "selection_eligible",
        "exclusion_reason",
    ]
    train = long_results[long_results["sample"] == "train"].copy()
    test = long_results[long_results["sample"] == "test"].copy()

    train = train.drop(columns=["sample"]).rename(
        columns={column: f"train_{column}" for column in train.columns if column not in keys}
    )
    test = test.drop(columns=["sample"]).rename(
        columns={column: f"test_{column}" for column in test.columns if column not in keys}
    )
    result = train.merge(test, on=keys, how="inner", validate="one_to_one")

    top_fraction = float(protocol["selection"]["top_fraction"])
    eligible_counts = (
        result[result["selection_eligible"]]
        .groupby("market")["run_id"]
        .nunique()
        .to_dict()
    )
    top_cutoffs = {
        market: int(math.ceil(count * top_fraction))
        for market, count in eligible_counts.items()
    }
    result["top_fraction_rank_cutoff"] = result["market"].map(top_cutoffs)
    result["rank_change"] = (
        result["test_rank_eligible"] - result["train_rank_eligible"]
    )
    result["absolute_rank_change"] = result["rank_change"].abs()
    result["train_top_fraction"] = (
        result["train_rank_eligible"] <= result["top_fraction_rank_cutoff"]
    ).fillna(False)
    result["test_top_fraction"] = (
        result["test_rank_eligible"] <= result["top_fraction_rank_cutoff"]
    ).fillna(False)
    result["top_fraction_retained"] = (
        result["train_top_fraction"] & result["test_top_fraction"]
    )
    return result.sort_values(
        ["market", "fold", "selection_eligible", "train_rank_eligible", "run_id"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)


def fold_rank_stability(walk_forward: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (market, fold), group in walk_forward.groupby(["market", "fold"]):
        eligible = group[group["selection_eligible"]].dropna(
            subset=["train_rank_eligible", "test_rank_eligible"]
        )
        rho = (
            float(
                eligible[["train_rank_eligible", "test_rank_eligible"]]
                .corr(method="spearman")
                .iloc[0, 1]
            )
            if len(eligible) >= 3
            else math.nan
        )
        cutoff = int(eligible["top_fraction_rank_cutoff"].iloc[0])
        train_top = eligible[eligible["train_rank_eligible"] <= cutoff]
        train_top3 = eligible.nsmallest(3, "train_rank_eligible")
        winner = eligible.nsmallest(1, "train_rank_eligible").iloc[0]

        rows.append(
            {
                "market": market,
                "fold": fold,
                "n_eligible_configs": int(len(eligible)),
                "top_fraction_rank_cutoff": cutoff,
                "spearman_train_test_rho": rho,
                "mean_absolute_rank_change": float(
                    eligible["absolute_rank_change"].mean()
                ),
                "median_absolute_rank_change": float(
                    eligible["absolute_rank_change"].median()
                ),
                "train_top_fraction_retention_share": float(
                    train_top["test_top_fraction"].mean()
                ),
                "train_top3_to_test_top_fraction_share": float(
                    train_top3["test_top_fraction"].mean()
                ),
                "train_winner_run_id": str(winner["run_id"]),
                "train_winner_test_rank": float(winner["test_rank_eligible"]),
                "train_winner_test_net_sharpe": float(winner["test_net_sharpe"]),
                "train_actual_start": str(eligible["train_actual_common_start"].iloc[0]),
                "train_actual_end": str(eligible["train_actual_common_end"].iloc[0]),
                "test_actual_start": str(eligible["test_actual_common_start"].iloc[0]),
                "test_actual_end": str(eligible["test_actual_common_end"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["market", "fold"]).reset_index(drop=True)


def config_stability(
    walk_forward: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    criteria = protocol["selection"]["stable_criteria"]
    minimum_top_share = float(criteria["min_test_top_fraction_share"])
    minimum_positive_share = float(criteria["min_positive_test_sharpe_share"])
    rank_change_units = float(
        criteria["max_median_absolute_rank_change_in_top_fraction_units"]
    )

    rows: list[dict[str, Any]] = []
    for (market, run_id), group in walk_forward.groupby(["market", "run_id"]):
        eligible = bool(group["selection_eligible"].iloc[0])
        cutoff = int(group["top_fraction_rank_cutoff"].iloc[0])
        median_train_rank = float(group["train_rank_eligible"].median()) if eligible else math.nan
        median_test_rank = float(group["test_rank_eligible"].median()) if eligible else math.nan
        test_top_share = float(group["test_top_fraction"].mean()) if eligible else math.nan
        positive_test_share = float((group["test_net_sharpe"] > 0.0).mean()) if eligible else math.nan
        median_abs_change = float(group["absolute_rank_change"].median()) if eligible else math.nan

        passes_train = eligible and median_train_rank <= cutoff
        passes_test = eligible and median_test_rank <= cutoff
        passes_top_share = eligible and test_top_share >= minimum_top_share
        passes_positive = eligible and positive_test_share >= minimum_positive_share
        passes_rank_change = eligible and median_abs_change <= cutoff * rank_change_units
        stable = all(
            [
                passes_train,
                passes_test,
                passes_top_share,
                passes_positive,
                passes_rank_change,
            ]
        )

        rows.append(
            {
                "market": market,
                "run_id": run_id,
                "selection_eligible": eligible,
                "exclusion_reason": str(group["exclusion_reason"].iloc[0]),
                "n_folds": int(group["fold"].nunique()),
                "top_fraction_rank_cutoff": cutoff,
                "median_train_rank": median_train_rank,
                "mean_train_rank": float(group["train_rank_eligible"].mean()) if eligible else math.nan,
                "median_test_rank": median_test_rank,
                "mean_test_rank": float(group["test_rank_eligible"].mean()) if eligible else math.nan,
                "worst_test_rank": float(group["test_rank_eligible"].max()) if eligible else math.nan,
                "median_absolute_rank_change": median_abs_change,
                "test_top_fraction_share": test_top_share,
                "positive_test_net_sharpe_share": positive_test_share,
                "mean_test_net_sharpe": float(group["test_net_sharpe"].mean()),
                "median_test_net_sharpe": float(group["test_net_sharpe"].median()),
                "worst_test_net_sharpe": float(group["test_net_sharpe"].min()),
                "mean_test_net_information_ratio": float(
                    group["test_net_information_ratio_vs_benchmark"].mean()
                ),
                "passes_median_train_top_fraction": passes_train,
                "passes_median_test_top_fraction": passes_test,
                "passes_test_top_fraction_share": passes_top_share,
                "passes_positive_test_sharpe_share": passes_positive,
                "passes_rank_change_limit": passes_rank_change,
                "stable_config": stable,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "market",
            "stable_config",
            "selection_eligible",
            "median_test_rank",
            "mean_test_net_sharpe",
            "worst_test_rank",
            "run_id",
        ],
        ascending=[True, False, False, True, False, True, True],
    ).reset_index(drop=True)


def period_series(series: pd.Series, start: str, end: str) -> pd.Series:
    return series.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()


def equal_weight_series(
    series_by_run: dict[str, pd.Series],
    run_ids: list[str],
    name: str,
) -> pd.Series:
    panel = pd.concat(
        [series_by_run[run_id].rename(run_id) for run_id in run_ids],
        axis=1,
        join="inner",
    ).dropna(how="any")
    if panel.empty:
        raise RuntimeError(f"Empty aligned panel for {run_ids}")
    result = panel.mean(axis=1)
    result.name = name
    return result


def max_abs_correlation_to_selected(
    candidate: str,
    selected: list[str],
    validation_returns: dict[str, pd.Series],
) -> float:
    correlations: list[float] = []
    for other in selected:
        aligned = pd.concat(
            [
                validation_returns[candidate].rename("candidate"),
                validation_returns[other].rename("selected"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) >= 20:
            value = float(aligned.corr().iloc[0, 1])
            if np.isfinite(value):
                correlations.append(abs(value))
    return max(correlations) if correlations else math.nan


def selection_history_bounds(
    protocol: dict[str, Any],
    market: str,
) -> tuple[str, str]:
    starts = [str(item["train_start"]) for item in protocol["folds"][market]]
    holdout_start = pd.Timestamp(protocol["retrospective_holdout"]["start"])
    end = str((holdout_start - pd.Timedelta(days=1)).date())
    return min(starts), end


def build_validation_return_map(
    runs: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    market: str,
) -> dict[str, pd.Series]:
    start, end = selection_history_bounds(protocol, market)
    return {
        run_id: period_series(item["net_primary"], start, end)
        for run_id, item in runs.items()
        if item["market"] == market
    }


def select_frozen_portfolios(
    stability: pd.DataFrame,
    runs: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    max_correlation = float(
        protocol["selection"]["max_abs_return_correlation"]
    )
    max_ensemble_size = int(protocol["selection"]["max_ensemble_size"])
    selections: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []

    for market in MARKETS:
        market_table = stability[stability["market"] == market].copy()
        eligible = market_table[market_table["selection_eligible"]].copy()
        eligible = eligible.sort_values(
            [
                "stable_config",
                "median_test_rank",
                "mean_test_net_sharpe",
                "worst_test_rank",
                "median_absolute_rank_change",
                "run_id",
            ],
            ascending=[False, True, False, True, True, True],
        )
        stable = eligible[eligible["stable_config"]].copy()
        validation_returns = build_validation_return_map(runs, protocol, market)

        if stable.empty:
            best_single = str(eligible.iloc[0]["run_id"])
            best_single_is_stable = False
        else:
            best_single = str(stable.iloc[0]["run_id"])
            best_single_is_stable = True

        ensemble_members: list[str] = []
        ensemble_status = "no_stable_ensemble"
        if len(stable) >= 2:
            target = min(max_ensemble_size, len(stable))
            for run_id in stable["run_id"].astype(str):
                correlation = max_abs_correlation_to_selected(
                    run_id,
                    ensemble_members,
                    validation_returns,
                )
                if ensemble_members and np.isfinite(correlation) and correlation > max_correlation:
                    audit_rows.append(
                        {
                            "market": market,
                            "run_id": run_id,
                            "stable_config": True,
                            "decision": "skip_for_ensemble",
                            "reason": (
                                f"max absolute validation correlation {correlation:.6f} "
                                f"> frozen limit {max_correlation:.2f}"
                            ),
                            "max_abs_correlation_to_selected": correlation,
                        }
                    )
                    continue
                ensemble_members.append(run_id)
                audit_rows.append(
                    {
                        "market": market,
                        "run_id": run_id,
                        "stable_config": True,
                        "decision": "select_for_ensemble",
                        "reason": "stable and passes correlation preference",
                        "max_abs_correlation_to_selected": correlation,
                    }
                )
                if len(ensemble_members) == target:
                    break

            # The DoD requires an ens-2 when at least two stable configurations
            # exist. If the correlation preference leaves only one member, add
            # the least-correlated remaining stable configuration. It remains a
            # stable member; no unstable fallback is allowed.
            if len(ensemble_members) == 1:
                remaining: list[tuple[float, str]] = []
                for run_id in stable["run_id"].astype(str):
                    if run_id in ensemble_members:
                        continue
                    correlation = max_abs_correlation_to_selected(
                        run_id,
                        ensemble_members,
                        validation_returns,
                    )
                    score = correlation if np.isfinite(correlation) else math.inf
                    remaining.append((score, run_id))
                if remaining:
                    correlation, run_id = min(remaining, key=lambda item: item[0])
                    ensemble_members.append(run_id)
                    audit_rows.append(
                        {
                            "market": market,
                            "run_id": run_id,
                            "stable_config": True,
                            "decision": "select_for_ens2_high_correlation",
                            "reason": (
                                "second stable member required by ens-2 DoD; "
                                "chosen as least-correlated remaining stable config"
                            ),
                            "max_abs_correlation_to_selected": correlation,
                        }
                    )

            if len(ensemble_members) >= 3:
                ensemble_status = "ens3"
                ensemble_members = ensemble_members[:3]
            elif len(ensemble_members) == 2:
                ensemble_status = (
                    "ens2" if len(stable) == 2 else "ens2_after_correlation_filter"
                )
            else:
                ensemble_members = []
                ensemble_status = "no_stable_ensemble"

        if ensemble_members:
            base_ids = ensemble_members
            base_type = "stable_ensemble"
            base_is_stable = True
        else:
            base_ids = [best_single]
            base_type = (
                "stable_single" if best_single_is_stable else "unstable_validation_reference"
            )
            base_is_stable = best_single_is_stable

        selections[market] = {
            "n_eligible_configs": int(len(eligible)),
            "n_stable_configs": int(len(stable)),
            "stable_config_ids": stable["run_id"].astype(str).tolist(),
            "best_frozen_single": best_single,
            "best_frozen_single_is_stable": best_single_is_stable,
            "ensemble_status": ensemble_status,
            "ensemble_members": ensemble_members,
            "base_strategy_type": base_type,
            "base_strategy_is_stable": base_is_stable,
            "base_strategy_members": base_ids,
        }

        selected_set = set(ensemble_members)
        already_audited = {
            str(item["run_id"])
            for item in audit_rows
            if item["market"] == market
        }
        for _, row in market_table.iterrows():
            run_id = str(row["run_id"])
            if run_id in already_audited or run_id in selected_set:
                continue
            if not bool(row["selection_eligible"]):
                decision = "ineligible"
                reason = str(row["exclusion_reason"])
            elif not bool(row["stable_config"]):
                decision = "not_stable"
                failed = [
                    column.removeprefix("passes_")
                    for column in [
                        "passes_median_train_top_fraction",
                        "passes_median_test_top_fraction",
                        "passes_test_top_fraction_share",
                        "passes_positive_test_sharpe_share",
                        "passes_rank_change_limit",
                    ]
                    if not bool(row[column])
                ]
                reason = "failed: " + ", ".join(failed)
            elif run_id == best_single and not ensemble_members:
                decision = "select_as_base_single"
                reason = (
                    "only stable single available"
                    if best_single_is_stable
                    else "best pre-holdout validation reference; not stable"
                )
            else:
                decision = "stable_not_selected"
                reason = "stable but not required for frozen ensemble"
            audit_rows.append(
                {
                    "market": market,
                    "run_id": run_id,
                    "stable_config": bool(row["stable_config"]),
                    "decision": decision,
                    "reason": reason,
                    "max_abs_correlation_to_selected": math.nan,
                }
            )

    audit = pd.DataFrame(audit_rows).sort_values(
        ["market", "decision", "run_id"]
    ).reset_index(drop=True)
    return selections, audit


def portfolio_components(
    runs: dict[str, dict[str, Any]],
    run_ids: list[str],
    start: str,
    end: str,
    tc_bps: float,
) -> dict[str, pd.Series]:
    gross_map = {
        run_id: period_series(runs[run_id]["gross"], start, end)
        for run_id in run_ids
    }
    turnover_map = {
        run_id: period_series(runs[run_id]["turnover"], start, end)
        for run_id in run_ids
    }
    gross = equal_weight_series(gross_map, run_ids, "gross_return")
    turnover = equal_weight_series(turnover_map, run_ids, "member_average_turnover")
    common = gross.index.intersection(turnover.index)
    gross = gross.reindex(common)
    turnover = turnover.reindex(common)
    net = gross - turnover * tc_bps / 10_000.0
    net.name = "net_return"
    return {"gross": gross, "turnover": turnover, "net": net}


def save_series_panel(series_by_market: dict[str, pd.Series], path: Path) -> None:
    panel = pd.concat(
        [series.rename(market) for market, series in series_by_market.items()],
        axis=1,
    )
    panel.index.name = "date"
    panel.to_parquet(path)


def build_frozen_series(
    runs: dict[str, dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, dict[str, dict[str, pd.Series]]]:
    holdout = protocol["retrospective_holdout"]
    holdout_start = str(holdout["start"])
    holdout_end = str(holdout["end"])
    primary_tc = protocol["primary_transaction_cost_bps"]
    output: dict[str, dict[str, dict[str, pd.Series]]] = {}

    for market in MARKETS:
        validation_start, validation_end = selection_history_bounds(protocol, market)
        best_id = str(selections[market]["best_frozen_single"])
        base_ids = [str(value) for value in selections[market]["base_strategy_members"]]
        ensemble_ids = [str(value) for value in selections[market]["ensemble_members"]]
        tc = float(primary_tc[market])

        market_output: dict[str, dict[str, pd.Series]] = {
            "best_single_validation": portfolio_components(
                runs, [best_id], validation_start, validation_end, tc
            ),
            "best_single_oos": portfolio_components(
                runs, [best_id], holdout_start, holdout_end, tc
            ),
            "base_validation": portfolio_components(
                runs, base_ids, validation_start, validation_end, tc
            ),
            "base_oos": portfolio_components(
                runs, base_ids, holdout_start, holdout_end, tc
            ),
        }
        if ensemble_ids:
            market_output["ensemble_validation"] = portfolio_components(
                runs, ensemble_ids, validation_start, validation_end, tc
            )
            market_output["ensemble_oos"] = portfolio_components(
                runs, ensemble_ids, holdout_start, holdout_end, tc
            )
        output[market] = market_output
    return output


def benchmark_comparison(
    frozen_series: dict[str, dict[str, dict[str, pd.Series]]],
    selections: dict[str, dict[str, Any]],
    proxies: dict[str, pd.Series],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annualization = protocol["annualization_days"]
    holdout = protocol["retrospective_holdout"]
    holdout_start = str(holdout["start"])
    holdout_end = str(holdout["end"])
    rows: list[dict[str, Any]] = []
    nav_rows: list[pd.DataFrame] = []

    for market in MARKETS:
        benchmark = period_series(proxies[market], holdout_start, holdout_end)
        constructions: list[tuple[str, str, dict[str, pd.Series]]] = [
            (
                "best_frozen_single",
                str(selections[market]["best_frozen_single"]),
                frozen_series[market]["best_single_oos"],
            )
        ]
        if selections[market]["ensemble_members"]:
            constructions.append(
                (
                    "stable_ensemble",
                    "+".join(selections[market]["ensemble_members"]),
                    frozen_series[market]["ensemble_oos"],
                )
            )

        for portfolio_type, portfolio_name, components in constructions:
            common = components["net"].index.intersection(benchmark.dropna().index)
            gross = components["gross"].reindex(common)
            net = components["net"].reindex(common)
            turnover = components["turnover"].reindex(common)
            bench = benchmark.reindex(common)
            gross_metrics = compute_metrics(
                gross,
                annualization=int(annualization[market]),
                turnover=turnover,
            )
            net_metrics = compute_metrics(
                net,
                annualization=int(annualization[market]),
                turnover=turnover,
            )
            corr, beta = correlation_beta(net, bench)
            rows.append(
                {
                    "market": market,
                    "portfolio_type": portfolio_type,
                    "portfolio_name": portfolio_name,
                    "construction_is_stable": (
                        bool(selections[market]["best_frozen_single_is_stable"])
                        if portfolio_type == "best_frozen_single"
                        else True
                    ),
                    **{f"gross_{key}": value for key, value in gross_metrics.items()},
                    **{f"net_{key}": value for key, value in net_metrics.items()},
                    "net_information_ratio_vs_benchmark": information_ratio(
                        net,
                        bench,
                        int(annualization[market]),
                    ),
                    "net_correlation_to_benchmark": corr,
                    "net_beta_to_benchmark": beta,
                    "transaction_cost_annualized_return_drag": (
                        float(gross_metrics["annualized_return"])
                        - float(net_metrics["annualized_return"])
                    ),
                }
            )
            nav = pd.DataFrame(
                {
                    "date": common,
                    "market": market,
                    "portfolio_type": portfolio_type,
                    "portfolio_name": portfolio_name,
                    "return": net.to_numpy(),
                    "nav": (1.0 + net).cumprod().to_numpy(),
                }
            )
            nav_rows.append(nav)

        benchmark_metrics = compute_metrics(
            benchmark,
            annualization=int(annualization[market]),
            turnover=None,
        )
        rows.append(
            {
                "market": market,
                "portfolio_type": "benchmark",
                "portfolio_name": {
                    "us": "SPX",
                    "moex": "IMOEX",
                    "crypto": "BTCUSDT",
                }[market],
                "construction_is_stable": True,
                **{f"gross_{key}": value for key, value in benchmark_metrics.items()},
                **{f"net_{key}": value for key, value in benchmark_metrics.items()},
                "net_information_ratio_vs_benchmark": 0.0,
                "net_correlation_to_benchmark": 1.0,
                "net_beta_to_benchmark": 1.0,
                "transaction_cost_annualized_return_drag": 0.0,
            }
        )
        nav_rows.append(
            pd.DataFrame(
                {
                    "date": benchmark.index,
                    "market": market,
                    "portfolio_type": "benchmark",
                    "portfolio_name": {
                        "us": "SPX",
                        "moex": "IMOEX",
                        "crypto": "BTCUSDT",
                    }[market],
                    "return": benchmark.to_numpy(),
                    "nav": (1.0 + benchmark).cumprod().to_numpy(),
                }
            )
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["market", "portfolio_type"]
    ).reset_index(drop=True)
    nav_long = pd.concat(nav_rows, ignore_index=True).sort_values(
        ["market", "portfolio_type", "date"]
    ).reset_index(drop=True)
    return comparison, nav_long


def tc_sensitivity(
    runs: dict[str, dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    holdout = protocol["retrospective_holdout"]
    annualization = protocol["annualization_days"]
    rows: list[dict[str, Any]] = []

    for market in MARKETS:
        validation_start, validation_end = selection_history_bounds(protocol, market)
        sample_bounds = {
            "validation": (validation_start, validation_end),
            "retrospective_holdout": (
                str(holdout["start"]),
                str(holdout["end"]),
            ),
        }
        portfolio_specs = {
            "best_frozen_single": [str(selections[market]["best_frozen_single"])],
            "base_strategy": [str(value) for value in selections[market]["base_strategy_members"]],
        }
        for portfolio_type, run_ids in portfolio_specs.items():
            for sample, (start, end) in sample_bounds.items():
                # Build gross and member-average turnover once. Alternative
                # transaction costs are descriptive sensitivity scenarios only;
                # selection remains frozen under the primary cost.
                gross_map = {
                    run_id: period_series(runs[run_id]["gross"], start, end)
                    for run_id in run_ids
                }
                turnover_map = {
                    run_id: period_series(runs[run_id]["turnover"], start, end)
                    for run_id in run_ids
                }
                gross = equal_weight_series(gross_map, run_ids, "gross_return")
                turnover = equal_weight_series(
                    turnover_map,
                    run_ids,
                    "member_average_turnover",
                )
                common = gross.index.intersection(turnover.index)
                gross = gross.reindex(common)
                turnover = turnover.reindex(common)

                for cost_bps in protocol["transaction_cost_sensitivity_bps"][market]:
                    cost = float(cost_bps)
                    net = gross - turnover * cost / 10_000.0
                    metrics = compute_metrics(
                        net,
                        annualization=int(annualization[market]),
                        turnover=turnover,
                    )
                    rows.append(
                        {
                            "market": market,
                            "portfolio_type": portfolio_type,
                            "portfolio_members": "+".join(run_ids),
                            "sample": sample,
                            "transaction_cost_bps": cost,
                            "is_primary_cost": cost
                            == float(protocol["primary_transaction_cost_bps"][market]),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["market", "portfolio_type", "sample", "transaction_cost_bps"]
    ).reset_index(drop=True)


def safe_markdown(frame: pd.DataFrame, **kwargs: Any) -> str:
    try:
        return frame.to_markdown(**kwargs)
    except ImportError:
        return frame.to_string(index=kwargs.get("index", False))


def build_report(
    output_dir: Path,
    stability_by_fold: pd.DataFrame,
    config_table: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
    protocol: dict[str, Any],
) -> None:
    lines = [
        "# Phase 3 v2 — Train/Test Walk-Forward, Frozen Selection and Holdout",
        "",
        "## Frozen protocol",
        "",
        "- Source: frozen `baseline_grid_v5` daily returns and turnover; no baseline backtest was rerun.",
        "- Primary selector: net Sharpe under US 25 bps, MOEX 20 bps and Crypto 10 bps.",
        "- Every fold ranks configurations on train and measures rank transfer on the following test period.",
        "- All eligible configurations share one common date index inside each fold/sample.",
        "- Benchmark-relative IR is descriptive, not a selection objective, because the strategy is market-neutral and the proxy is long-only.",
        "- The final 2023–2024 sample is a chronological retrospective holdout, not a fully pristine OOS sample.",
        "",
        "## Fold-level train-to-test stability",
        "",
    ]
    fold_columns = [
        "market",
        "fold",
        "spearman_train_test_rho",
        "median_absolute_rank_change",
        "train_top_fraction_retention_share",
        "train_top3_to_test_top_fraction_share",
        "train_winner_run_id",
        "train_winner_test_rank",
        "train_winner_test_net_sharpe",
    ]
    lines.append(
        safe_markdown(
            stability_by_fold[fold_columns],
            index=False,
            floatfmt=".4f",
        )
    )
    lines.extend(["", "## Frozen construction", ""])
    for market in MARKETS:
        selection = selections[market]
        lines.extend(
            [
                f"### {market.upper()}",
                f"- Eligible configurations: {selection['n_eligible_configs']}",
                f"- Stable configurations: {selection['n_stable_configs']}",
                f"- Best frozen single: `{selection['best_frozen_single']}` "
                f"(stable={selection['best_frozen_single_is_stable']})",
                f"- Ensemble status: `{selection['ensemble_status']}`",
                f"- Ensemble members: {selection['ensemble_members'] or 'none'}",
                f"- Protection base: `{selection['base_strategy_type']}` — "
                f"{selection['base_strategy_members']}",
                "",
            ]
        )

    lines.extend(["## Stable-config summary", ""])
    display_configs = config_table[
        [
            "market",
            "run_id",
            "stable_config",
            "median_train_rank",
            "median_test_rank",
            "test_top_fraction_share",
            "positive_test_net_sharpe_share",
            "median_absolute_rank_change",
            "mean_test_net_sharpe",
        ]
    ]
    lines.append(safe_markdown(display_configs, index=False, floatfmt=".4f"))

    lines.extend(["", "## Retrospective holdout comparison", ""])
    display_comparison = comparison[
        [
            "market",
            "portfolio_type",
            "portfolio_name",
            "construction_is_stable",
            "net_annualized_return",
            "net_annualized_volatility",
            "net_sharpe",
            "net_max_drawdown",
            "net_information_ratio_vs_benchmark",
            "net_correlation_to_benchmark",
            "net_beta_to_benchmark",
        ]
    ]
    lines.append(
        safe_markdown(display_comparison, index=False, floatfmt=".4f")
    )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- The holdout was not used to change folds, stability criteria, selection or transaction-cost assumptions.",
            "- No unstable configuration was added merely to force an ens-3.",
            "- A missing stable ensemble is a valid finding.",
            "- Protection must use the frozen base strategy and the unchanged 63-day/q90 rule family.",
            "- US and MOEX strategy results remain price-return based; ordinary cash dividends are not included.",
            "",
        ]
    )
    (output_dir / "phase3_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="config/phase3_protocol_v2.yaml",
    )
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = repo_root()
    protocol_path = root / args.protocol
    protocol = load_protocol(protocol_path)
    runs_dir = root / str(args.runs_dir or protocol["source_runs_dir"])
    output_dir = root / str(args.output_dir or protocol["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen baseline runs...")
    runs = load_runs(runs_dir, protocol)
    print("Loading cached market proxies...")
    proxies = load_market_proxies(root, protocol)

    print("Building train/test walk-forward results...")
    long_results = fold_long_results(runs, proxies, protocol)
    walk_forward = wide_walk_forward_results(long_results, protocol)
    rank_stability = fold_rank_stability(walk_forward)
    config_table = config_stability(walk_forward, protocol)

    print("Freezing stable configurations and portfolio construction...")
    selections, selection_audit = select_frozen_portfolios(
        config_table,
        runs,
        protocol,
    )
    frozen_series = build_frozen_series(runs, selections, protocol)
    comparison, nav_long = benchmark_comparison(
        frozen_series,
        selections,
        proxies,
        protocol,
    )
    cost_sensitivity = tc_sensitivity(runs, selections, protocol)

    long_results.to_csv(output_dir / "walk_forward_long.csv", index=False)
    walk_forward.to_csv(output_dir / "walk_forward_results.csv", index=False)
    rank_stability.to_csv(output_dir / "fold_rank_stability.csv", index=False)
    config_table.to_csv(output_dir / "config_stability.csv", index=False)
    selection_audit.to_csv(
        output_dir / "finalist_selection_audit.csv",
        index=False,
    )
    comparison.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    comparison.to_csv(output_dir / "oos_results.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "tc_sensitivity.csv", index=False)
    nav_long.to_parquet(output_dir / "oos_nav.parquet", index=False)

    save_series_panel(
        {
            market: frozen_series[market]["base_validation"]["net"]
            for market in MARKETS
        },
        output_dir / "base_validation.parquet",
    )
    save_series_panel(
        {
            market: frozen_series[market]["base_oos"]["net"]
            for market in MARKETS
        },
        output_dir / "base_oos.parquet",
    )
    save_series_panel(
        {
            market: frozen_series[market]["base_validation"]["gross"]
            for market in MARKETS
        },
        output_dir / "base_validation_gross.parquet",
    )
    save_series_panel(
        {
            market: frozen_series[market]["base_oos"]["gross"]
            for market in MARKETS
        },
        output_dir / "base_oos_gross.parquet",
    )
    save_series_panel(
        {
            market: frozen_series[market]["best_single_validation"]["net"]
            for market in MARKETS
        },
        output_dir / "best_single_validation.parquet",
    )
    save_series_panel(
        {
            market: frozen_series[market]["best_single_oos"]["net"]
            for market in MARKETS
        },
        output_dir / "best_single_oos.parquet",
    )

    ensemble_validation = {
        market: frozen_series[market]["ensemble_validation"]["net"]
        for market in MARKETS
        if "ensemble_validation" in frozen_series[market]
    }
    ensemble_oos = {
        market: frozen_series[market]["ensemble_oos"]["net"]
        for market in MARKETS
        if "ensemble_oos" in frozen_series[market]
    }
    if ensemble_validation:
        save_series_panel(
            ensemble_validation,
            output_dir / "ensemble_validation.parquet",
        )
        save_series_panel(
            ensemble_oos,
            output_dir / "ensemble_oos.parquet",
        )

    frozen_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "protocol_path": str(protocol_path.relative_to(root)),
        "source_runs_dir": str(runs_dir.relative_to(root)),
        "output_dir": str(output_dir.relative_to(root)),
        "primary_selection_metric": protocol["selection"]["primary_metric"],
        "primary_transaction_cost_bps": protocol[
            "primary_transaction_cost_bps"
        ],
        "retrospective_holdout": protocol["retrospective_holdout"],
        "folds": protocol["folds"],
        "stable_criteria": protocol["selection"]["stable_criteria"],
        "selection": selections,
        "guardrails": protocol.get("interpretation_guardrails", []),
    }
    (output_dir / "frozen_selection.json").write_text(
        json.dumps(frozen_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shutil.copy2(protocol_path, output_dir / "phase3_protocol_v2.yaml")
    amendment_path = root / "docs" / "phase3_protocol_amendment.md"
    if amendment_path.exists():
        shutil.copy2(
            amendment_path,
            output_dir / "phase3_protocol_amendment.md",
        )

    build_report(
        output_dir,
        rank_stability,
        config_table,
        selections,
        comparison,
        protocol,
    )

    print(f"\nSaved corrected Phase 3 outputs to: {output_dir}")
    print("\nFrozen selection:")
    for market in MARKETS:
        print(f"\n{market.upper()}")
        print(json.dumps(selections[market], indent=2, ensure_ascii=False))

    print("\nFold rank stability:")
    print(
        rank_stability[
            [
                "market",
                "fold",
                "spearman_train_test_rho",
                "median_absolute_rank_change",
                "train_top_fraction_retention_share",
                "train_winner_test_rank",
            ]
        ].to_string(index=False)
    )

    print("\nRetrospective holdout comparison:")
    print(
        comparison[
            [
                "market",
                "portfolio_type",
                "portfolio_name",
                "construction_is_stable",
                "net_annualized_return",
                "net_sharpe",
                "net_max_drawdown",
                "net_information_ratio_vs_benchmark",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
