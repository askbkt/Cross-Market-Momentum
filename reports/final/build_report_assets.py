from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[1]
GENERATED_DIR = REPORT_DIR / "generated"
FIGURES_DIR = GENERATED_DIR / "figures"
TABLES_DIR = GENERATED_DIR / "tables"

COLORS = {
    "us": "#2F5D8A",
    "moex": "#D08C35",
    "crypto": "#8B6AAE",
    "benchmark": "#5B6573",
    "single": "#2F5D8A",
    "ensemble": "#D08C35",
    "zero": "#30343B",
}
MARKET_LABELS = {"us": "US", "moex": "MOEX", "crypto": "Crypto"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_markdown_metric(path: Path, label: str) -> str:
    pattern = re.compile(
        rf"^\|\s*{re.escape(label)}\s*\|\s*(.*?)\s*\|\s*$",
        re.IGNORECASE,
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    raise KeyError(f"Metric {label!r} was not found in {path}")


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def format_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def write_table(
    path: Path,
    headers: list[str],
    rows: list[list[object]],
    alignment: str,
    *,
    font_size: str = r"\small",
) -> None:
    body = [
        font_size,
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(" & ".join(str(cell) for cell in row) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def setup_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#30343B",
            "axes.titlecolor": "#30343B",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": "#5B6573",
            "ytick.color": "#5B6573",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#D9DEE5",
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def build_source_manifest() -> None:
    sources = [
        "docs/DoD_market_momentum_study.md",
        "docs/Roadmap_cross_market_momentum.md",
        "docs/phase3_protocol_amendment.md",
        "config/phase3_protocol_v2.yaml",
        "backtests/enhanced_momentum/reports/data_quality/us_data_quality.md",
        "backtests/enhanced_momentum/reports/data_quality/moex_data_quality.md",
        "backtests/enhanced_momentum/reports/data_quality/crypto_data_quality.md",
        "results/baseline_grid_v5/manifest.json",
        "results/baseline_grid_v5/final_analysis/baseline_v5_final_report.md",
        "results/baseline_grid_v5/final_analysis/table_market_overview.csv",
        "results/baseline_grid_v5/final_analysis/table_lookback_effect.csv",
        "results/phase3/fold_rank_stability.csv",
        "results/phase3/config_stability.csv",
        "results/phase3/frozen_selection.json",
        "results/phase3/benchmark_comparison.csv",
        "results/phase3/tc_sensitivity.csv",
        "results/phase3/oos_nav.parquet",
        "results/phase3/protection/frozen_thresholds.csv",
        "results/phase3/protection/signal_diagnostics.csv",
        "results/phase3/protection/protection_results.csv",
        "results/phase3/protection/crisis_window_results.csv",
        "results/phase3/protection/protection_daily.parquet",
    ]
    rows = []
    for relative in sources:
        path = REPO_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(GENERATED_DIR / "source_manifest.csv", index=False)


def validate_frozen_evidence() -> dict[str, pd.DataFrame]:
    baseline = pd.read_csv(REPO_ROOT / "results/baseline_grid_v5/final_analysis/table_market_overview.csv")
    lookback = pd.read_csv(REPO_ROOT / "results/baseline_grid_v5/final_analysis/table_lookback_effect.csv")
    fold = pd.read_csv(REPO_ROOT / "results/phase3/fold_rank_stability.csv")
    stability = pd.read_csv(REPO_ROOT / "results/phase3/config_stability.csv")
    holdout = pd.read_csv(REPO_ROOT / "results/phase3/benchmark_comparison.csv")
    tc = pd.read_csv(REPO_ROOT / "results/phase3/tc_sensitivity.csv")
    protection = pd.read_csv(REPO_ROOT / "results/phase3/protection/protection_results.csv")
    signals = pd.read_csv(REPO_ROOT / "results/phase3/protection/signal_diagnostics.csv")
    crisis = pd.read_csv(REPO_ROOT / "results/phase3/protection/crisis_window_results.csv")

    assert set(baseline["market"]) == {"us", "moex", "crypto"}
    assert baseline["n_runs"].sum() == 81
    assert len(fold) == 14
    assert len(stability) == 81
    starts = pd.to_datetime(holdout["net_start"])
    ends = pd.to_datetime(holdout["net_end"])
    assert starts.min() >= pd.Timestamp("2023-01-01")
    assert starts.max() <= pd.Timestamp("2023-01-03")
    assert ends.min() >= pd.Timestamp("2024-12-30")
    assert ends.max() <= pd.Timestamp("2024-12-31")
    assert protection.loc[protection["sample"].eq("retrospective_holdout"), "variant"].nunique() == 4
    assert set(crisis["evidence_role"]) == {"in_sample_validation_diagnostic_only"}

    stable_counts = stability.loc[stability["stable_config"].eq(True)].groupby("market").size().to_dict()
    assert stable_counts == {"crypto": 2, "moex": 4, "us": 1}

    return {
        "baseline": baseline,
        "lookback": lookback,
        "fold": fold,
        "stability": stability,
        "holdout": holdout,
        "tc": tc,
        "protection": protection,
        "signals": signals,
        "crisis": crisis,
    }


def build_tables(data: dict[str, pd.DataFrame]) -> None:
    protocol = yaml.safe_load((REPO_ROOT / "config/phase3_protocol_v2.yaml").read_text(encoding="utf-8"))
    frozen = json.loads((REPO_ROOT / "results/phase3/frozen_selection.json").read_text(encoding="utf-8"))
    markets_block = frozen.get("markets", frozen.get("selection", frozen))

    quality_paths = {
        "us": REPO_ROOT / "backtests/enhanced_momentum/reports/data_quality/us_data_quality.md",
        "moex": REPO_ROOT / "backtests/enhanced_momentum/reports/data_quality/moex_data_quality.md",
        "crypto": REPO_ROOT / "backtests/enhanced_momentum/reports/data_quality/crypto_data_quality.md",
    }
    source_labels = {
        "us": "Supervisor Russell 3000 panel",
        "moex": "MOEX ISS, historical TQBR",
        "crypto": "Binance Public Data, USDT pairs",
    }
    data_rows = []
    for market in ["us", "moex", "crypto"]:
        path = quality_paths[market]
        data_rows.append([
            MARKET_LABELS[market], source_labels[market],
            f"{read_markdown_metric(path, 'First date')}--{read_markdown_metric(path, 'Last date')}",
            read_markdown_metric(path, "Total assets (ever)"),
            read_markdown_metric(path, "Mean investable assets"),
            f"{protocol['primary_transaction_cost_bps'][market]:.0f}",
        ])
    write_table(TABLES_DIR / "data_scope.tex", ["Market", "Source and universe", "Panel span", "Assets ever", "Mean investable", "Primary TC (bps)"], data_rows, "llrrrr", font_size=r"\footnotesize")

    baseline_rows = []
    for market in ["us", "moex", "crypto"]:
        row = data["baseline"].loc[data["baseline"]["market"].eq(market)].iloc[0]
        baseline_rows.append([
            MARKET_LABELS[market], f"{int(row.n_positive_sharpe)}/{int(row.n_runs)}",
            format_num(row.mean_sharpe, 3), latex_escape(row.best_run_id),
            format_num(row.best_run_sharpe, 3), format_pct(row.best_run_annualized_return, 2),
            format_pct(row.best_run_max_drawdown, 1),
        ])
    write_table(TABLES_DIR / "baseline_overview.tex", ["Market", "Positive Sharpe", "Mean Sharpe", "Best configuration", "Best Sharpe", "Best ann. return", "Best MaxDD"], baseline_rows, "lrrlrrr", font_size=r"\footnotesize")

    selection_rows = []
    for market in ["us", "moex", "crypto"]:
        info = markets_block[market]
        members = info.get("ensemble_members", [])
        selection_rows.append([
            MARKET_LABELS[market], str(info["n_eligible_configs"]), str(info["n_stable_configs"]),
            latex_escape(info["best_frozen_single"]), latex_escape(info["ensemble_status"]),
            latex_escape(", ".join(members) if members else "none"),
        ])
    write_table(TABLES_DIR / "selection_summary.tex", ["Market", "Eligible", "Stable", "Best frozen single", "Ensemble", "Members"], selection_rows, "lrrlll", font_size=r"\scriptsize")

    type_labels = {"benchmark": "Market proxy", "best_frozen_single": "Best frozen single", "stable_ensemble": "Stable ensemble"}
    holdout_rows = []
    for market in ["us", "moex", "crypto"]:
        for ptype in ["benchmark", "best_frozen_single", "stable_ensemble"]:
            block = data["holdout"].loc[data["holdout"]["market"].eq(market) & data["holdout"]["portfolio_type"].eq(ptype)]
            if block.empty:
                continue
            row = block.iloc[0]
            holdout_rows.append([
                MARKET_LABELS[market], type_labels[ptype], format_pct(row.net_annualized_return, 1),
                format_pct(row.net_annualized_volatility, 1), format_num(row.net_sharpe, 2),
                format_pct(row.net_max_drawdown, 1), format_num(row.net_correlation_to_benchmark, 2),
            ])
    write_table(TABLES_DIR / "holdout_summary.tex", ["Market", "Portfolio", "Ann. return", "Ann. vol", "Sharpe", "MaxDD", "Corr. to proxy"], holdout_rows, "llrrrrr", font_size=r"\footnotesize")

    oos_protection = data["protection"].loc[data["protection"]["sample"].eq("retrospective_holdout")]
    protection_rows = []
    for market in ["us", "moex", "crypto"]:
        for variant in ["portfolio_vol_q90", "market_vol_q90", "combo_and_q90"]:
            row = oos_protection.loc[oos_protection["market"].eq(market) & oos_protection["variant"].eq(variant)].iloc[0]
            protection_rows.append([
                MARKET_LABELS[market], latex_escape(variant), format_pct(row.risk_off_share, 1),
                str(int(row.n_switches)), format_pct(row.delta_annualized_return, 2),
                format_num(row.delta_sharpe, 2), f"{100.0 * row.delta_max_drawdown:.2f} pp",
            ])
    write_table(TABLES_DIR / "protection_summary.tex", ["Market", "Rule", "Risk-off", "Switches", r"$\Delta$ ann. return", r"$\Delta$ Sharpe", r"$\Delta$ MaxDD"], protection_rows, "llrrrrr", font_size=r"\scriptsize")

    folds_rows = []
    for market in ["us", "moex", "crypto"]:
        for fold in protocol["folds"][market]:
            train_span = f"{str(fold['train_start'])[:4]}--{str(fold['train_end'])[:4]}"
            test_span = f"{str(fold['test_start'])[:4]}--{str(fold['test_end'])[:4]}"
            folds_rows.append([
                MARKET_LABELS[market],
                latex_escape(fold["name"]),
                train_span,
                test_span,
            ])
    write_table(TABLES_DIR / "folds.tex", ["Market", "Fold", "Train", "Test"], folds_rows, "llll", font_size=r"\scriptsize")


def plot_baseline_lookback(lookback: pd.DataFrame) -> None:
    order = ["6M", "12M", "24M"]
    markets = ["us", "moex", "crypto"]
    x = np.arange(len(order))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    hatches = ["", "//", ".."]
    for i, market in enumerate(markets):
        block = lookback.loc[lookback["market"].eq(market)].set_index("lookback_label").reindex(order)
        ax.bar(x + (i - 1) * width, block["mean_sharpe"], width, label=MARKET_LABELS[market], color=COLORS[market], edgecolor="#30343B", linewidth=0.5, hatch=hatches[i])
    ax.axhline(0, color=COLORS["zero"], linewidth=0.9)
    ax.set_xticks(x, order)
    ax.set_ylabel("Mean gross Sharpe across nine configurations")
    ax.set_title("Baseline mean Sharpe by momentum lookback")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper right")
    fig.savefig(FIGURES_DIR / "baseline_lookback.pdf")
    plt.close(fig)


def plot_rank_transfer(fold: pd.DataFrame) -> None:
    markets = ["us", "moex", "crypto"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.4), sharey=False)
    for ax, market in zip(axes, markets, strict=True):
        block = fold.loc[fold["market"].eq(market)].sort_values("fold")
        y = np.arange(len(block))
        rho = block["spearman_train_test_rho"].to_numpy()
        ax.hlines(y, 0, rho, color=COLORS[market], linewidth=1.4)
        ax.scatter(rho, y, color=COLORS[market], edgecolor="#30343B", linewidth=0.5, s=36, zorder=3)
        ax.axvline(0, color=COLORS["zero"], linewidth=0.8)
        ax.set_yticks(y, [name.replace(f"{market}_", "") for name in block["fold"]])
        ax.set_xlim(-0.55, 1.0)
        ax.set_title(MARKET_LABELS[market])
        ax.set_xlabel("Spearman rank correlation")
        ax.grid(axis="x")
    axes[0].set_ylabel("Walk-forward fold")
    fig.suptitle("Train-to-test configuration rankings transferred unevenly", y=1.02, fontsize=11)
    fig.savefig(FIGURES_DIR / "rank_transfer.pdf")
    plt.close(fig)


def plot_holdout_performance(holdout: pd.DataFrame) -> None:
    strategies = holdout.loc[~holdout["portfolio_type"].eq("benchmark")].copy()
    strategies["label"] = strategies.apply(lambda row: f"{MARKET_LABELS[row.market]} - " + ("single" if row.portfolio_type == "best_frozen_single" else "ensemble"), axis=1)
    order = [label for label in ["US - single", "MOEX - single", "MOEX - ensemble", "Crypto - single", "Crypto - ensemble"] if label in set(strategies["label"])]
    strategies = strategies.set_index("label").reindex(order).reset_index()
    colors = [COLORS["ensemble"] if "ensemble" in label else COLORS["single"] for label in strategies["label"]]
    y = np.arange(len(strategies))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    axes[0].barh(y, 100 * strategies["net_annualized_return"], color=colors, edgecolor="#30343B", linewidth=0.5)
    axes[1].barh(y, strategies["net_sharpe"], color=colors, edgecolor="#30343B", linewidth=0.5, hatch=["" if "single" in label else "//" for label in strategies["label"]])
    for ax, title, xlabel in [(axes[0], "Net annualized return", "Percent"), (axes[1], "Net Sharpe ratio", "Sharpe")]:
        ax.axvline(0, color=COLORS["zero"], linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
    axes[0].set_yticks(y, strategies["label"])
    axes[0].invert_yaxis()
    fig.suptitle("Frozen momentum constructions in the 2023-2024 retrospective holdout", y=1.02, fontsize=11)
    fig.savefig(FIGURES_DIR / "holdout_performance.pdf")
    plt.close(fig)


def plot_holdout_nav() -> None:
    nav = pd.read_parquet(REPO_ROOT / "results/phase3/oos_nav.parquet")
    nav["date"] = pd.to_datetime(nav["date"])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.7), sharex=True)
    style = {"benchmark": (COLORS["benchmark"], "--", 1.1), "best_frozen_single": (COLORS["single"], "-", 1.4), "stable_ensemble": (COLORS["ensemble"], "-.", 1.4)}
    labels = {"benchmark": "Market proxy", "best_frozen_single": "Best frozen single", "stable_ensemble": "Stable ensemble"}
    for ax, market in zip(axes, ["us", "moex", "crypto"], strict=True):
        block = nav.loc[nav["market"].eq(market)]
        for ptype in ["benchmark", "best_frozen_single", "stable_ensemble"]:
            series = block.loc[block["portfolio_type"].eq(ptype)]
            if series.empty:
                continue
            color, linestyle, linewidth = style[ptype]
            ax.plot(series["date"], series["nav"], label=labels[ptype], color=color, linestyle=linestyle, linewidth=linewidth)
        ax.axhline(1.0, color="#AEB6C1", linewidth=0.7)
        ax.set_ylabel(MARKET_LABELS[market])
        ax.grid(axis="y")
        ax.legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Normalized NAV: frozen momentum portfolios and long-only market proxies", y=0.995, fontsize=11)
    fig.savefig(FIGURES_DIR / "holdout_nav.pdf")
    plt.close(fig)


def plot_tc_sensitivity(tc: pd.DataFrame) -> None:
    block = tc.loc[tc["sample"].eq("retrospective_holdout")]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.5), sharey=True)
    styles = {"base_strategy": (COLORS["ensemble"], "o", "-", "Base construction"), "best_frozen_single": (COLORS["single"], "s", "--", "Best frozen single")}
    for ax, market in zip(axes, ["us", "moex", "crypto"], strict=True):
        market_block = block.loc[block["market"].eq(market)]
        for ptype, (color, marker, linestyle, label) in styles.items():
            series = market_block.loc[market_block["portfolio_type"].eq(ptype)].sort_values("transaction_cost_bps")
            if series.empty:
                continue
            ax.plot(series["transaction_cost_bps"], 100 * series["annualized_return"], color=color, marker=marker, linestyle=linestyle, linewidth=1.3, markersize=4, label=label)
        ax.axhline(0, color=COLORS["zero"], linewidth=0.8)
        ax.set_title(MARKET_LABELS[market])
        ax.set_xlabel("One-way transaction cost (bps)")
        ax.grid(axis="y")
    axes[0].set_ylabel("Net annualized return (%)")
    axes[-1].legend(loc="best")
    fig.suptitle("Transaction costs attenuate but do not reverse the main holdout conclusions", y=1.02, fontsize=11)
    fig.savefig(FIGURES_DIR / "tc_sensitivity.pdf")
    plt.close(fig)


def plot_protection_effects(protection: pd.DataFrame) -> None:
    block = protection.loc[protection["sample"].eq("retrospective_holdout") & ~protection["variant"].eq("no_protection")].copy()
    labels = [(market, variant) for market in ["us", "moex", "crypto"] for variant in ["portfolio_vol_q90", "market_vol_q90", "combo_and_q90"]]
    block["key"] = list(zip(block["market"], block["variant"], strict=False))
    block = block.set_index("key").reindex(labels).reset_index(drop=True)
    display = [f"{MARKET_LABELS[m]}\n{v.replace('_q90', '').replace('_', ' ')}" for m, v in labels]
    x = np.arange(len(block))
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.1), sharex=True)
    axes[0].bar(x, block["delta_sharpe"], color=[COLORS[m] for m, _ in labels], edgecolor="#30343B", linewidth=0.5)
    axes[1].bar(x, 100 * block["delta_max_drawdown"], color=[COLORS[m] for m, _ in labels], edgecolor="#30343B", linewidth=0.5, hatch=["", "//", ".."] * 3)
    axes[0].set_ylabel("Change in Sharpe")
    axes[1].set_ylabel("Change in MaxDD (pp)")
    for ax in axes:
        ax.axhline(0, color=COLORS["zero"], linewidth=0.8)
        ax.grid(axis="y")
    axes[1].set_xticks(x, display, rotation=28, ha="right")
    fig.suptitle("Frozen volatility protection did not improve holdout risk-adjusted performance", y=1.01, fontsize=11)
    fig.savefig(FIGURES_DIR / "protection_effects.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    data = validate_frozen_evidence()
    build_source_manifest()
    build_tables(data)
    plot_baseline_lookback(data["lookback"])
    plot_rank_transfer(data["fold"])
    plot_holdout_performance(data["holdout"])
    plot_holdout_nav()
    plot_tc_sensitivity(data["tc"])
    plot_protection_effects(data["protection"])
    print(f"Generated report assets in {GENERATED_DIR}")


if __name__ == "__main__":
    main()
