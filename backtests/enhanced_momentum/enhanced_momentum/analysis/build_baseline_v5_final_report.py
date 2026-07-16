from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MARKETS = ("us", "moex", "crypto")
LOOKBACK_ORDER = ("6M", "12M", "24M")
SKIP_ORDER = ("0M", "1M", "3M")
QUANTILE_ORDER = (0.10, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(
        description=(
            "Build final tables, robustness maps, figures, and a markdown "
            "summary for baseline_grid_v5."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root / "results" / "baseline_grid_v5" / "summary.csv",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=project_root / "results" / "baseline_grid_v5" / "runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "results"
            / "baseline_grid_v5"
            / "final_analysis"
        ),
    )
    return parser.parse_args()


def validate_summary(summary: pd.DataFrame) -> None:
    required = {
        "run_id",
        "market",
        "lookback_label",
        "skip_label",
        "quantile",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "annualized_turnover",
        "min_n_long",
        "min_n_short",
        "min_daily_gross_exposure",
        "max_daily_gross_exposure",
        "max_abs_daily_net_exposure",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            "Summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    duplicated = summary["run_id"].duplicated()
    if duplicated.any():
        duplicates = summary.loc[duplicated, "run_id"].tolist()
        raise ValueError(f"Duplicate run_id values: {duplicates}")

    found_markets = set(summary["market"].astype(str))
    expected_markets = set(DEFAULT_MARKETS)
    if found_markets != expected_markets:
        raise ValueError(
            f"Expected markets {sorted(expected_markets)}, "
            f"found {sorted(found_markets)}."
        )

    counts = summary["market"].value_counts()
    if not (counts == 27).all():
        raise ValueError(
            "Expected exactly 27 configurations per market; got:\n"
            + counts.to_string()
        )

    if summary.isna().any().any():
        columns = summary.columns[summary.isna().any()].tolist()
        raise ValueError(
            "Summary contains missing values in columns: "
            + ", ".join(columns)
        )


def collect_zero_target_flags(
    summary: pd.DataFrame,
    runs_dir: Path,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for run_id in summary["run_id"]:
        diag_path = runs_dir / str(run_id) / "rebalance_diagnostics.parquet"

        if not diag_path.exists():
            records.append(
                {
                    "run_id": run_id,
                    "n_zero_target_rebalances": np.nan,
                    "zero_target_effective_dates": "",
                }
            )
            continue

        diag = pd.read_parquet(diag_path)
        zero = diag[
            (diag["n_long"] == 0)
            | (diag["n_short"] == 0)
        ].copy()

        effective_dates = (
            pd.to_datetime(zero["effective_date"])
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        records.append(
            {
                "run_id": run_id,
                "n_zero_target_rebalances": int(len(zero)),
                "zero_target_effective_dates": ",".join(effective_dates),
            }
        )

    return pd.DataFrame(records)


def build_market_overview(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for market, group in summary.groupby("market", sort=False):
        best = group.loc[group["sharpe"].idxmax()]
        rows.append(
            {
                "market": market,
                "n_runs": int(len(group)),
                "n_positive_sharpe": int((group["sharpe"] > 0).sum()),
                "n_positive_annualized_return": int(
                    (group["annualized_return"] > 0).sum()
                ),
                "mean_sharpe": float(group["sharpe"].mean()),
                "median_sharpe": float(group["sharpe"].median()),
                "min_sharpe": float(group["sharpe"].min()),
                "max_sharpe": float(group["sharpe"].max()),
                "mean_annualized_return": float(
                    group["annualized_return"].mean()
                ),
                "mean_max_drawdown": float(
                    group["max_drawdown"].mean()
                ),
                "best_run_id": str(best["run_id"]),
                "best_run_sharpe": float(best["sharpe"]),
                "best_run_annualized_return": float(
                    best["annualized_return"]
                ),
                "best_run_max_drawdown": float(
                    best["max_drawdown"]
                ),
            }
        )

    result = pd.DataFrame(rows)
    result["market"] = pd.Categorical(
        result["market"],
        categories=list(DEFAULT_MARKETS),
        ordered=True,
    )
    return result.sort_values("market").reset_index(drop=True)


def build_parameter_effect(
    summary: pd.DataFrame,
    parameter: str,
) -> pd.DataFrame:
    result = (
        summary
        .groupby(["market", parameter], observed=True)
        .agg(
            n_runs=("run_id", "size"),
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            min_sharpe=("sharpe", "min"),
            max_sharpe=("sharpe", "max"),
            n_positive_sharpe=("sharpe", lambda x: int((x > 0).sum())),
            mean_annualized_return=("annualized_return", "mean"),
            mean_annualized_vol=("annualized_vol", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_annualized_turnover=("annualized_turnover", "mean"),
        )
        .reset_index()
    )
    return result


def build_cross_market_robustness(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        summary
        .groupby(
            ["lookback_label", "skip_label", "quantile"],
            observed=True,
        )
        .agg(
            n_markets=("market", "nunique"),
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            worst_market_sharpe=("sharpe", "min"),
            best_market_sharpe=("sharpe", "max"),
            std_sharpe=("sharpe", "std"),
            mean_annualized_return=("annualized_return", "mean"),
            worst_annualized_return=("annualized_return", "min"),
            mean_max_drawdown=("max_drawdown", "mean"),
            worst_max_drawdown=("max_drawdown", "min"),
            mean_annualized_turnover=("annualized_turnover", "mean"),
            n_positive_markets=("sharpe", lambda x: int((x > 0).sum())),
            n_caveated_runs=(
                "n_zero_target_rebalances",
                lambda x: int((x.fillna(0) > 0).sum()),
            ),
        )
        .reset_index()
    )

    result["all_markets_positive"] = (
        result["n_positive_markets"] == result["n_markets"]
    )

    result = result.sort_values(
        [
            "all_markets_positive",
            "worst_market_sharpe",
            "mean_sharpe",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    result.insert(
        0,
        "robustness_rank",
        np.arange(1, len(result) + 1),
    )
    return result


def build_top_configs_by_market(
    summary: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    return (
        summary
        .sort_values(
            ["market", "sharpe"],
            ascending=[True, False],
        )
        .groupby("market", group_keys=False)
        .head(n)
        [
            [
                "market",
                "run_id",
                "lookback_label",
                "skip_label",
                "quantile",
                "sharpe",
                "annualized_return",
                "annualized_vol",
                "max_drawdown",
                "annualized_turnover",
                "min_n_long",
                "min_n_short",
                "n_zero_target_rebalances",
            ]
        ]
        .reset_index(drop=True)
    )


def save_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    value_format: str = ".2f",
) -> None:
    ordered = pivot.reindex(
        index=list(LOOKBACK_ORDER),
        columns=list(SKIP_ORDER),
    )

    values = ordered.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(values, aspect="auto")

    ax.set_xticks(np.arange(len(ordered.columns)))
    ax.set_xticklabels(ordered.columns)
    ax.set_yticks(np.arange(len(ordered.index)))
    ax.set_yticklabels(ordered.index)
    ax.set_xlabel("Skip period")
    ax.set_ylabel("Lookback")
    ax.set_title(title)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    format(value, value_format),
                    ha="center",
                    va="center",
                )

    fig.colorbar(image, ax=ax, label="Sharpe ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_figures(
    summary: pd.DataFrame,
    cross_market: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    for market in DEFAULT_MARKETS:
        market_data = summary[summary["market"] == market]

        for quantile in QUANTILE_ORDER:
            subset = market_data[
                np.isclose(market_data["quantile"], quantile)
            ]
            pivot = subset.pivot(
                index="lookback_label",
                columns="skip_label",
                values="sharpe",
            )
            save_heatmap(
                pivot,
                title=(
                    f"{market.upper()} Sharpe robustness, "
                    f"q={int(quantile * 100)}%"
                ),
                output_path=(
                    figures_dir
                    / f"sharpe_heatmap_{market}_q{int(quantile * 100)}.png"
                ),
            )

    for quantile in QUANTILE_ORDER:
        subset = cross_market[
            np.isclose(cross_market["quantile"], quantile)
        ]
        pivot = subset.pivot(
            index="lookback_label",
            columns="skip_label",
            values="worst_market_sharpe",
        )
        save_heatmap(
            pivot,
            title=(
                "Worst-market Sharpe across US, MOEX, Crypto, "
                f"q={int(quantile * 100)}%"
            ),
            output_path=(
                figures_dir
                / f"worst_market_sharpe_q{int(quantile * 100)}.png"
            ),
        )

    lookback = (
        summary
        .groupby(["market", "lookback_label"], observed=True)["sharpe"]
        .mean()
        .unstack("market")
        .reindex(list(LOOKBACK_ORDER))
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for market in DEFAULT_MARKETS:
        ax.plot(
            lookback.index,
            lookback[market],
            marker="o",
            label=market.upper(),
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Mean Sharpe by momentum lookback")
    ax.set_xlabel("Lookback")
    ax.set_ylabel("Mean Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "mean_sharpe_by_lookback.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for market in DEFAULT_MARKETS:
        group = summary[summary["market"] == market]
        ax.scatter(
            group["max_drawdown"].abs(),
            group["annualized_return"],
            label=market.upper(),
            alpha=0.8,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Annualized return versus maximum drawdown")
    ax.set_xlabel("Absolute maximum drawdown")
    ax.set_ylabel("Annualized return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "return_vs_drawdown.png",
        dpi=180,
    )
    plt.close(fig)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_markdown_report(
    summary: pd.DataFrame,
    market_overview: pd.DataFrame,
    lookback_effect: pd.DataFrame,
    cross_market: pd.DataFrame,
    caveated: pd.DataFrame,
) -> str:
    robust_clean = cross_market[
        cross_market["n_caveated_runs"] == 0
    ]
    best_robust = robust_clean.iloc[0]

    lines = [
        "# Baseline Momentum Grid v5 — Final Analysis",
        "",
        "## Validation status",
        "",
        f"- Runs: {len(summary)}.",
        "- Markets: US, MOEX, Binance spot crypto.",
        "- Portfolio accounting: fixed-notional long and short sleeves.",
        "- Gross exposure: 1.0; target net exposure: 0.0.",
        (
            "- Runs with a zero-target rebalance: "
            f"{int((summary['n_zero_target_rebalances'].fillna(0) > 0).sum())}."
        ),
        "",
        "## Market overview",
        "",
    ]

    for row in market_overview.itertuples(index=False):
        lines.append(
            f"- **{str(row.market).upper()}**: "
            f"{row.n_positive_sharpe}/{row.n_runs} configurations "
            f"with positive Sharpe; mean Sharpe "
            f"{row.mean_sharpe:.3f}; best configuration "
            f"`{row.best_run_id}` with Sharpe "
            f"{row.best_run_sharpe:.3f}, annualized return "
            f"{pct(row.best_run_annualized_return)}, maximum drawdown "
            f"{pct(row.best_run_max_drawdown)}."
        )

    lines.extend(
        [
            "",
            "## Lookback effect",
            "",
        ]
    )

    for market in DEFAULT_MARKETS:
        market_rows = (
            lookback_effect[
                lookback_effect["market"] == market
            ]
            .set_index("lookback_label")
            .reindex(list(LOOKBACK_ORDER))
        )
        values = ", ".join(
            f"{label}: {market_rows.loc[label, 'mean_sharpe']:.3f}"
            for label in LOOKBACK_ORDER
        )
        lines.append(f"- **{market.upper()}** mean Sharpe — {values}.")

    lines.extend(
        [
            "",
            "## Most robust clean cross-market configuration",
            "",
            (
                f"`{best_robust['lookback_label']} / "
                f"{best_robust['skip_label']} / "
                f"q={int(float(best_robust['quantile']) * 100)}%`"
            ),
            "",
            (
                f"- Mean Sharpe: {best_robust['mean_sharpe']:.3f}."
            ),
            (
                "- Worst-market Sharpe: "
                f"{best_robust['worst_market_sharpe']:.3f}."
            ),
            (
                "- Mean annualized return: "
                f"{pct(float(best_robust['mean_annualized_return']))}."
            ),
            (
                "- Mean maximum drawdown: "
                f"{pct(float(best_robust['mean_max_drawdown']))}."
            ),
            "",
            "## MOEX data caveat",
            "",
        ]
    )

    if caveated.empty:
        lines.append("- No zero-target rebalance cases were found.")
    else:
        run_ids = ", ".join(
            f"`{run_id}`"
            for run_id in caveated["run_id"].tolist()
        )
        dates = sorted(
            {
                date
                for text in caveated["zero_target_effective_dates"]
                for date in str(text).split(",")
                if date
            }
        )
        lines.extend(
            [
                (
                    f"- Affected configurations: {run_ids}."
                ),
                (
                    "- Affected effective rebalance date(s): "
                    + ", ".join(dates)
                    + "."
                ),
                (
                    "- Cause: the 12-month, zero-skip signal starts on "
                    "2022-01-07, a sparse MOEX observation with only "
                    "12 market-wide closes versus roughly 257–261 on "
                    "adjacent sessions. None of the 90 assets present at "
                    "the 2023-01-31 decision date had a comparable start "
                    "price, so the portfolio remained in cash until the "
                    "next scheduled rebalance."
                ),
                (
                    "- Treatment: retain the raw runs, flag them in all "
                    "tables, and do not use them as primary evidence for "
                    "cross-market robustness."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Main interpretation",
            "",
            (
                "The baseline supports a cross-market medium-term "
                "momentum effect concentrated around the 6-month "
                "lookback. The 12-month horizon remains positive on "
                "US and MOEX but is weaker and less reliable on crypto. "
                "The 24-month horizon deteriorates across all three "
                "markets and is negative on average."
            ),
            "",
            (
                "The strongest clean robustness region is the "
                "6-month lookback with a 1-month skip and a 20–30% "
                "quantile. These configurations remain positive across "
                "all three markets and are not affected by the MOEX "
                "sparse-session caveat."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    validate_summary(summary)

    flags = collect_zero_target_flags(summary, runs_dir)
    summary = summary.merge(
        flags,
        on="run_id",
        how="left",
        validate="one_to_one",
    )
    summary["has_zero_target_rebalance"] = (
        summary["n_zero_target_rebalances"].fillna(0) > 0
    )

    market_overview = build_market_overview(summary)
    lookback_effect = build_parameter_effect(
        summary,
        "lookback_label",
    )
    skip_effect = build_parameter_effect(
        summary,
        "skip_label",
    )
    quantile_effect = build_parameter_effect(
        summary,
        "quantile",
    )
    cross_market = build_cross_market_robustness(summary)
    top_configs = build_top_configs_by_market(summary)

    caveated = summary[
        summary["has_zero_target_rebalance"]
    ][
        [
            "run_id",
            "market",
            "lookback_label",
            "skip_label",
            "quantile",
            "sharpe",
            "annualized_return",
            "max_drawdown",
            "n_zero_target_rebalances",
            "zero_target_effective_dates",
        ]
    ].copy()

    files = {
        "summary_with_flags.csv": summary,
        "table_market_overview.csv": market_overview,
        "table_lookback_effect.csv": lookback_effect,
        "table_skip_effect.csv": skip_effect,
        "table_quantile_effect.csv": quantile_effect,
        "table_cross_market_robustness.csv": cross_market,
        "table_top_configs_by_market.csv": top_configs,
        "table_caveated_runs.csv": caveated,
    }

    for filename, frame in files.items():
        frame.to_csv(
            output_dir / filename,
            index=False,
        )

    build_figures(
        summary,
        cross_market,
        figures_dir,
    )

    report = build_markdown_report(
        summary,
        market_overview,
        lookback_effect,
        cross_market,
        caveated,
    )
    (output_dir / "baseline_v5_final_report.md").write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 88)
    print("BASELINE GRID V5 FINAL ANALYSIS COMPLETE")
    print("=" * 88)
    print(f"Input summary: {summary_path}")
    print(f"Runs directory: {runs_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("Caveated runs:")
    if caveated.empty:
        print("None")
    else:
        print(
            caveated[
                [
                    "run_id",
                    "n_zero_target_rebalances",
                    "zero_target_effective_dates",
                ]
            ].to_string(index=False)
        )
    print()
    print("Top 10 cross-market robustness configurations:")
    print(
        cross_market[
            [
                "robustness_rank",
                "lookback_label",
                "skip_label",
                "quantile",
                "mean_sharpe",
                "worst_market_sharpe",
                "mean_annualized_return",
                "mean_max_drawdown",
                "n_caveated_runs",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print()
    print("Generated:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
