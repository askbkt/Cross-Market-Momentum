from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from enhanced_momentum.data_loaders.registry import get_loader, load_market_config


MARKETS = ("us", "moex", "crypto")


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


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return result


def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 3 file not found: {path}")
    panel = pd.read_parquet(path)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    missing = set(MARKETS) - set(panel.columns)
    if missing:
        raise ValueError(f"{path} is missing markets: {sorted(missing)}")
    return panel.loc[:, list(MARKETS)]


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


def load_proxy_returns(
    root: Path,
    protocol: dict[str, Any],
    market: str,
) -> pd.Series:
    config = resolved_market_config(root, protocol, market)
    loader = get_loader(market, config)
    data = loader.load()
    proxy = pd.to_numeric(
        data.market_proxy_returns,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    proxy.index = pd.to_datetime(proxy.index)
    proxy = proxy.sort_index()
    proxy.name = "market_proxy_return"
    if proxy.dropna().empty:
        raise RuntimeError(f"Empty market proxy for {market}")
    return proxy


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


def annualized_return(returns: pd.Series, annualization: int) -> float:
    r = clean_returns(returns)
    if r.empty:
        return math.nan
    growth = float((1.0 + r).prod())
    years = len(r) / annualization
    if growth <= 0.0 or years <= 0.0:
        return math.nan
    return float(growth ** (1.0 / years) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    r = clean_returns(returns)
    if r.empty:
        return math.nan
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def expected_shortfall(returns: pd.Series, probability: float = 0.01) -> float:
    r = clean_returns(returns)
    if r.empty:
        return math.nan
    threshold = float(r.quantile(probability))
    tail = r[r <= threshold]
    return float(tail.mean()) if not tail.empty else math.nan


def compute_metrics(
    returns: pd.Series,
    *,
    annualization: int,
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
            "worst_day": math.nan,
            "q01_daily_return": math.nan,
            "expected_shortfall_1pct": math.nan,
        }

    std = float(r.std(ddof=1)) if len(r) > 1 else math.nan
    volatility = (
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

    return {
        "n_days": int(len(r)),
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "total_return": float((1.0 + r).prod() - 1.0),
        "annualized_return": ann_return,
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": drawdown,
        "calmar": float(calmar),
        "worst_day": float(r.min()),
        "q01_daily_return": float(r.quantile(0.01)),
        "expected_shortfall_1pct": expected_shortfall(r, 0.01),
    }


def build_features(
    base_return: pd.Series,
    proxy_return: pd.Series,
    rolling_window: int,
) -> pd.DataFrame:
    base = clean_returns(base_return).rename("base_return")
    aligned = base.to_frame()
    aligned["market_proxy_return"] = pd.to_numeric(
        proxy_return.reindex(base.index),
        errors="coerce",
    )
    aligned = aligned.sort_index()
    aligned["portfolio_vol"] = (
        aligned["base_return"]
        .rolling(rolling_window, min_periods=rolling_window)
        .std(ddof=1)
        .shift(1)
    )
    aligned["market_vol"] = (
        aligned["market_proxy_return"]
        .rolling(rolling_window, min_periods=rolling_window)
        .std(ddof=1)
        .shift(1)
    )
    return aligned


def calibrate_thresholds(
    validation_features: pd.DataFrame,
    quantile: float,
) -> dict[str, float]:
    portfolio_sample = validation_features["portfolio_vol"].dropna()
    market_sample = validation_features["market_vol"].dropna()
    if len(portfolio_sample) < 100 or len(market_sample) < 100:
        raise RuntimeError(
            "Insufficient validation observations for protection thresholds: "
            f"portfolio={len(portfolio_sample)}, market={len(market_sample)}"
        )
    return {
        "portfolio_vol_q90": float(portfolio_sample.quantile(quantile)),
        "market_vol_q90": float(market_sample.quantile(quantile)),
    }


def risk_off_signal(
    features: pd.DataFrame,
    *,
    variant: str,
    thresholds: dict[str, float],
) -> pd.Series:
    portfolio_high = features["portfolio_vol"] > thresholds["portfolio_vol_q90"]
    market_high = features["market_vol"] > thresholds["market_vol_q90"]

    if variant == "no_protection":
        signal = pd.Series(False, index=features.index)
    elif variant == "portfolio_vol_q90":
        signal = portfolio_high
    elif variant == "market_vol_q90":
        signal = market_high
    elif variant == "combo_and_q90":
        signal = portfolio_high & market_high
    else:
        raise ValueError(f"Unknown protection variant: {variant}")
    return signal.fillna(False).astype(bool)


def apply_protection(
    features: pd.DataFrame,
    *,
    variant: str,
    thresholds: dict[str, float],
    switching_tc_bps: float,
    normal_exposure: float,
    risk_off_exposure: float,
) -> pd.DataFrame:
    result = features.copy()
    risk_off = risk_off_signal(
        result,
        variant=variant,
        thresholds=thresholds,
    )
    exposure = pd.Series(
        np.where(risk_off, risk_off_exposure, normal_exposure),
        index=result.index,
        dtype=float,
    )

    # Unified-engine convention: one-way turnover is 0.5 × L1 weight change.
    # Scaling a gross-one portfolio from exposure 1 to 0 therefore creates 0.5
    # units of switching turnover. The first observed state is not charged as a
    # transition from an unobserved pre-sample position.
    exposure_change = exposure.diff().abs().fillna(0.0)
    switching_turnover = 0.5 * exposure_change
    switching_cost = switching_turnover * switching_tc_bps / 10_000.0

    result["variant"] = variant
    result["risk_off"] = risk_off
    result["exposure"] = exposure
    result["exposure_change"] = exposure_change
    result["switching_turnover"] = switching_turnover
    result["switching_cost"] = switching_cost
    result["protected_return_before_switching_tc"] = (
        result["base_return"] * exposure
    )
    result["protected_return"] = (
        result["protected_return_before_switching_tc"] - switching_cost
    )
    return result


def summarize_variant(
    protected: pd.DataFrame,
    *,
    market: str,
    sample: str,
    variant: str,
    annualization: int,
    base_metadata: dict[str, Any],
    switching_tc_bps: float,
) -> dict[str, Any]:
    baseline = clean_returns(protected["base_return"])
    strategy = clean_returns(protected["protected_return"])
    baseline_metrics = compute_metrics(
        baseline,
        annualization=annualization,
    )
    protected_metrics = compute_metrics(
        strategy,
        annualization=annualization,
    )

    risk_off = protected["risk_off"].reindex(baseline.index).fillna(False)
    exposure = protected["exposure"].reindex(baseline.index).fillna(1.0)
    switching_turnover = protected["switching_turnover"].reindex(
        baseline.index
    ).fillna(0.0)
    switching_cost = protected["switching_cost"].reindex(
        baseline.index
    ).fillna(0.0)
    risk_off_returns = baseline[risk_off]

    return {
        "market": market,
        "sample": sample,
        "variant": variant,
        "base_strategy_type": base_metadata["base_strategy_type"],
        "base_strategy_is_stable": base_metadata["base_strategy_is_stable"],
        "base_strategy_members": "+".join(base_metadata["base_strategy_members"]),
        "switching_tc_bps": switching_tc_bps,
        "mean_exposure": float(exposure.mean()),
        "risk_off_share": float(risk_off.mean()),
        "n_switches": int((switching_turnover > 0.0).sum()),
        "total_switching_turnover": float(switching_turnover.sum()),
        "switching_tc_drag_arithmetic": float(switching_cost.sum()),
        "missed_positive_return_sum": float(
            risk_off_returns.clip(lower=0.0).sum()
        ),
        "avoided_negative_return_sum": float(
            -risk_off_returns.clip(upper=0.0).sum()
        ),
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        **{f"protected_{key}": value for key, value in protected_metrics.items()},
        "delta_annualized_return": (
            float(protected_metrics["annualized_return"])
            - float(baseline_metrics["annualized_return"])
        ),
        "delta_sharpe": (
            float(protected_metrics["sharpe"])
            - float(baseline_metrics["sharpe"])
        ),
        "delta_sortino": (
            float(protected_metrics["sortino"])
            - float(baseline_metrics["sortino"])
        ),
        "delta_max_drawdown": (
            float(protected_metrics["max_drawdown"])
            - float(baseline_metrics["max_drawdown"])
        ),
        "delta_calmar": (
            float(protected_metrics["calmar"])
            - float(baseline_metrics["calmar"])
        ),
        "delta_expected_shortfall_1pct": (
            float(protected_metrics["expected_shortfall_1pct"])
            - float(baseline_metrics["expected_shortfall_1pct"])
        ),
    }


def threshold_table(
    thresholds_by_market: dict[str, dict[str, float]],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rolling_window = int(protocol["protection"]["rolling_vol_window"])
    quantile = float(protocol["protection"]["volatility_quantile"])
    for market, thresholds in thresholds_by_market.items():
        annualization = int(protocol["annualization_days"][market])
        for threshold_name, value in thresholds.items():
            rows.append(
                {
                    "market": market,
                    "threshold": threshold_name,
                    "value_daily_vol": value,
                    "annualized_value": value * np.sqrt(annualization),
                    "calibration_quantile": quantile,
                    "rolling_window": rolling_window,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["market", "threshold"]
    ).reset_index(drop=True)


def signal_diagnostics(
    validation_features: pd.DataFrame,
    oos_features: pd.DataFrame,
    *,
    market: str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = {
        "portfolio_vol": thresholds["portfolio_vol_q90"],
        "market_vol": thresholds["market_vol_q90"],
    }
    for feature, threshold in mapping.items():
        validation = validation_features[feature].dropna()
        oos = oos_features[feature].dropna()
        exceed = oos > threshold
        rows.append(
            {
                "market": market,
                "feature": feature,
                "threshold": threshold,
                "validation_n": int(len(validation)),
                "validation_exceedance_share": float(
                    (validation > threshold).mean()
                ),
                "oos_n": int(len(oos)),
                "oos_missing_feature_days": int(len(oos_features) - len(oos)),
                "oos_min": float(oos.min()),
                "oos_median": float(oos.median()),
                "oos_q90": float(oos.quantile(0.90)),
                "oos_max": float(oos.max()),
                "oos_exceedance_days": int(exceed.sum()),
                "oos_exceedance_share": float(exceed.mean()),
                "first_oos_exceedance": (
                    str(exceed[exceed].index[0].date())
                    if exceed.any()
                    else "did_not_fire"
                ),
            }
        )
    return rows


def crisis_window_results(
    protected_by_variant: dict[str, pd.DataFrame],
    *,
    market: str,
    protocol: dict[str, Any],
    base_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    annualization = int(protocol["annualization_days"][market])
    windows = protocol["protection"].get("crisis_windows", {}).get(market, [])
    for window in windows:
        start = str(window["start"])
        end = str(window["end"])
        for variant, full_protected in protected_by_variant.items():
            window_frame = full_protected.loc[
                pd.Timestamp(start):pd.Timestamp(end)
            ]
            if window_frame.empty:
                continue
            summary = summarize_variant(
                window_frame,
                market=market,
                sample="validation_crisis_diagnostic",
                variant=variant,
                annualization=annualization,
                base_metadata=base_metadata,
                switching_tc_bps=float(
                    protocol["primary_transaction_cost_bps"][market]
                ),
            )
            summary.update(
                {
                    "crisis_window": str(window["name"]),
                    "crisis_start": start,
                    "crisis_end": end,
                    "evidence_role": "in_sample_validation_diagnostic_only",
                }
            )
            rows.append(summary)
    return rows


def safe_markdown(frame: pd.DataFrame, **kwargs: Any) -> str:
    try:
        return frame.to_markdown(**kwargs)
    except ImportError:
        return frame.to_string(index=kwargs.get("index", False))


def build_report(
    output_dir: Path,
    thresholds: pd.DataFrame,
    results: pd.DataFrame,
    frozen_selection: dict[str, Any],
) -> None:
    lines = [
        "# Phase 3 v2 — Frozen Regime-Aware Protection",
        "",
        "## Design",
        "",
        "- The base series is the corrected frozen Phase 3 construction for each market.",
        "- When no stable ensemble exists, the report explicitly labels the base as a stable single or unstable validation reference.",
        "- Portfolio and market volatility use a 63-observation rolling standard deviation shifted by one observation.",
        "- q90 thresholds are calibrated only on pre-holdout history and mechanically applied to the 2023–2024 retrospective holdout.",
        "- No protection winner is selected from holdout results; all frozen variants remain in the comparison.",
        "- Base returns already include primary strategy transaction costs. Switching costs are incremental and use 0.5 × absolute exposure change.",
        "",
        "## Base constructions",
        "",
    ]
    for market in MARKETS:
        selection = frozen_selection["selection"][market]
        lines.extend(
            [
                f"- {market.upper()}: `{selection['base_strategy_type']}`; "
                f"stable={selection['base_strategy_is_stable']}; "
                f"members={selection['base_strategy_members']}",
            ]
        )

    lines.extend(["", "## Frozen thresholds", ""])
    lines.append(safe_markdown(thresholds, index=False, floatfmt=".6f"))
    lines.extend(["", "## Retrospective holdout comparison", ""])
    oos = results[results["sample"] == "retrospective_holdout"].copy()
    columns = [
        "market",
        "base_strategy_type",
        "base_strategy_is_stable",
        "variant",
        "protected_annualized_return",
        "protected_sharpe",
        "protected_max_drawdown",
        "protected_calmar",
        "protected_expected_shortfall_1pct",
        "mean_exposure",
        "risk_off_share",
        "n_switches",
        "switching_tc_drag_arithmetic",
        "delta_annualized_return",
        "delta_sharpe",
        "delta_max_drawdown",
        "delta_calmar",
    ]
    lines.append(
        safe_markdown(
            oos[columns].sort_values(["market", "variant"]),
            index=False,
            floatfmt=".4f",
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Protection is evaluated as a risk-management mechanism, not as an alpha-search loop.",
            "- A rule may be useful if it reduces drawdown or tail loss at an explicitly reported cost in exposure, switching and missed upside.",
            "- A rule that does not fire, fires too often, or worsens results is a valid negative finding.",
            "- Historical crisis-window tables are validation diagnostics and are not independent holdout evidence.",
            "- No new threshold, window or logical variant may be introduced because of these holdout results.",
            "",
        ]
    )
    (output_dir / "protection_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="config/phase3_protocol_v2.yaml",
    )
    parser.add_argument("--phase3-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = repo_root()
    protocol_path = root / args.protocol
    protocol = load_yaml(protocol_path)
    phase3_dir = root / str(args.phase3_dir or protocol["output_dir"])
    output_dir = root / str(
        args.output_dir or f"{protocol['output_dir']}/protection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_selection_path = phase3_dir / "frozen_selection.json"
    if not frozen_selection_path.exists():
        raise FileNotFoundError(
            "Run the corrected walk_forward_runner before protection: "
            f"{frozen_selection_path}"
        )
    frozen_selection = json.loads(
        frozen_selection_path.read_text(encoding="utf-8")
    )

    validation_panel = read_panel(phase3_dir / "base_validation.parquet")
    oos_panel = read_panel(phase3_dir / "base_oos.parquet")

    protection_config = protocol["protection"]
    rolling_window = int(protection_config["rolling_vol_window"])
    quantile = float(protection_config["volatility_quantile"])
    variants = [str(value) for value in protection_config["variants"]]
    normal_exposure = float(protection_config["normal_exposure"])
    risk_off_exposure = float(
        protection_config["binary_risk_off_exposure"]
    )

    thresholds_by_market: dict[str, dict[str, float]] = {}
    result_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    crisis_rows: list[dict[str, Any]] = []

    for market in MARKETS:
        print(f"Loading cached market proxy for {market}...")
        proxy = load_proxy_returns(root, protocol, market)
        validation_return = validation_panel[market].dropna()
        oos_return = oos_panel[market].dropna()
        base_metadata = frozen_selection["selection"][market]

        combined_return = pd.concat(
            [validation_return, oos_return]
        ).sort_index()
        combined_return = combined_return[
            ~combined_return.index.duplicated(keep="first")
        ]
        full_features = build_features(
            combined_return,
            proxy,
            rolling_window,
        )
        validation_features = full_features.reindex(
            validation_return.index
        ).dropna(subset=["base_return"])
        oos_features = full_features.reindex(oos_return.index).dropna(
            subset=["base_return", "market_proxy_return"]
        )

        thresholds = calibrate_thresholds(
            validation_features,
            quantile,
        )
        thresholds_by_market[market] = thresholds
        diagnostic_rows.extend(
            signal_diagnostics(
                validation_features,
                oos_features,
                market=market,
                thresholds=thresholds,
            )
        )

        protected_by_variant: dict[str, pd.DataFrame] = {}
        for variant in variants:
            full_protected = apply_protection(
                full_features,
                variant=variant,
                thresholds=thresholds,
                switching_tc_bps=float(
                    protocol["primary_transaction_cost_bps"][market]
                ),
                normal_exposure=normal_exposure,
                risk_off_exposure=risk_off_exposure,
            )
            protected_by_variant[variant] = full_protected

            samples = {
                "validation": full_protected.reindex(validation_return.index),
                "retrospective_holdout": full_protected.reindex(oos_return.index),
            }
            for sample, protected in samples.items():
                protected = protected.dropna(
                    subset=["base_return"]
                )
                result_rows.append(
                    summarize_variant(
                        protected,
                        market=market,
                        sample=sample,
                        variant=variant,
                        annualization=int(
                            protocol["annualization_days"][market]
                        ),
                        base_metadata=base_metadata,
                        switching_tc_bps=float(
                            protocol["primary_transaction_cost_bps"][market]
                        ),
                    )
                )
                daily = protected[
                    [
                        "base_return",
                        "market_proxy_return",
                        "portfolio_vol",
                        "market_vol",
                        "risk_off",
                        "exposure",
                        "exposure_change",
                        "switching_turnover",
                        "switching_cost",
                        "protected_return_before_switching_tc",
                        "protected_return",
                    ]
                ].copy()
                daily.insert(0, "base_strategy_type", base_metadata["base_strategy_type"])
                daily.insert(0, "variant", variant)
                daily.insert(0, "sample", sample)
                daily.insert(0, "market", market)
                daily.index.name = "date"
                daily_rows.append(daily.reset_index())

        crisis_rows.extend(
            crisis_window_results(
                protected_by_variant,
                market=market,
                protocol=protocol,
                base_metadata=base_metadata,
            )
        )

    results = pd.DataFrame(result_rows).sort_values(
        ["sample", "market", "variant"]
    ).reset_index(drop=True)
    thresholds = threshold_table(thresholds_by_market, protocol)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["market", "feature"]
    ).reset_index(drop=True)
    daily = pd.concat(daily_rows, ignore_index=True).sort_values(
        ["sample", "market", "variant", "date"]
    ).reset_index(drop=True)
    crisis = pd.DataFrame(crisis_rows)
    if not crisis.empty:
        crisis = crisis.sort_values(
            ["market", "crisis_window", "variant"]
        ).reset_index(drop=True)

    results.to_csv(output_dir / "protection_results.csv", index=False)
    thresholds.to_csv(output_dir / "frozen_thresholds.csv", index=False)
    diagnostics.to_csv(output_dir / "signal_diagnostics.csv", index=False)
    daily.to_parquet(output_dir / "protection_daily.parquet", index=False)
    crisis.to_csv(output_dir / "crisis_window_results.csv", index=False)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "protocol_path": str(protocol_path.relative_to(root)),
        "phase3_dir": str(phase3_dir.relative_to(root)),
        "output_dir": str(output_dir.relative_to(root)),
        "base_selection": frozen_selection["selection"],
        "rolling_vol_window": rolling_window,
        "volatility_quantile": quantile,
        "variants": variants,
        "normal_exposure": normal_exposure,
        "risk_off_exposure": risk_off_exposure,
        "switching_turnover_convention": "0.5 * absolute exposure change",
        "primary_transaction_cost_bps": protocol[
            "primary_transaction_cost_bps"
        ],
        "thresholds": thresholds_by_market,
        "guardrails": [
            "Thresholds calibrated on validation only.",
            "Volatility features shifted by one observation.",
            "OOS features warm-started with pre-OOS history only.",
            "Exposure state and switching costs carry across the boundary.",
            "No holdout-based variant selection.",
            "All frozen variants retained.",
            "Crisis-window results are validation diagnostics only.",
        ],
    }
    (output_dir / "protection_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(protocol_path, output_dir / "phase3_protocol_v2.yaml")
    build_report(
        output_dir,
        thresholds,
        results,
        frozen_selection,
    )

    print(f"\nSaved corrected protection outputs to: {output_dir}")
    print("\nFrozen thresholds:")
    print(thresholds.to_string(index=False))
    print("\nSignal diagnostics:")
    print(diagnostics.to_string(index=False))
    print("\nRetrospective holdout protection comparison:")
    columns = [
        "market",
        "base_strategy_type",
        "base_strategy_is_stable",
        "variant",
        "protected_annualized_return",
        "protected_sharpe",
        "protected_max_drawdown",
        "protected_calmar",
        "protected_expected_shortfall_1pct",
        "mean_exposure",
        "risk_off_share",
        "n_switches",
        "switching_tc_drag_arithmetic",
        "delta_max_drawdown",
    ]
    print(
        results.loc[
            results["sample"] == "retrospective_holdout",
            columns,
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
