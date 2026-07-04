from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:

    p = Path(__file__).resolve()

    for parent in [p.parent, *p.parents]:

        if (parent / ".git").exists():

            return parent

    raise RuntimeError("Cannot locate repo root")

RESULTS_DIR = _repo_root() / "data" / "results_oos"


def _rank_within_window(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{metric}_rank"] = out.groupby("eval_window")[metric].rank(
        ascending=False,
        method="min",
    )
    return out


def _make_analytical_median_rows(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "final_nav",
        "sharpe",
        "ir_benchmark",
        "alpha_benchmark",
        "alpha_benchmark_pvalue",
        "geom_avg_xs_r",
        "max_dd",
        "factor_loadings_low_risk",
        "factor_loadings_momentum",
        "factor_loadings_size",
        "factor_loadings_quality",
        "factor_loadings_value",
        "factor_loadings_spx_rf",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    rows = []
    for eval_window, g in df.groupby("eval_window"):
        row = {
            "run_id": "analytical_median",
            "config_name": "analytical_median_of_7_configs",
            "config_type": "analytical_line",
            "eval_window": eval_window,
            "test_start": g["test_start"].iloc[0] if "test_start" in g.columns else None,
            "test_end": g["test_end"].iloc[0] if "test_end" in g.columns else None,
            "quantile": np.nan,
            "exclude_last_days": np.nan,
            "window_days": np.nan,
            "as_zscore": np.nan,
            "return_type": "n/a",
            "volatility_scaling": np.nan,
            "vol_window_days": np.nan,
        }
        for col in metric_cols:
            row[col] = g[col].median(skipna=True)
        rows.append(row)

    return pd.DataFrame(rows)


def _spearman_is_vs_oos(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Optional comparison if ladder leaderboard exists.
    It maps OOS configs to config_id and compares with in-sample leaderboard if possible.
    """
    ladder_leaderboard = _repo_root() / "data" / "results_ladder" / "leaderboard.csv"
    if not ladder_leaderboard.exists():
        return pd.DataFrame(
            [{"note": "results_ladder/leaderboard.csv not found; skipped Spearman comparison"}]
        )

    lb = pd.read_csv(ladder_leaderboard)

    # leaderboard has config_id as index if saved by prior script; try to recover it
    if "config_id" not in lb.columns:
        first_col = lb.columns[0]
        lb = lb.rename(columns={first_col: "config_id"})

    full = summary[summary["eval_window"] == "full_oos_2020_2023"].copy()

    def cfg_id(row: pd.Series) -> str:
        parts = [
            row.get("quantile"),
            row.get("exclude_last_days"),
            row.get("as_zscore"),
            row.get("window_days"),
            row.get("return_type"),
            row.get("volatility_scaling"),
            row.get("vol_window_days"),
            row.get("rebal_freq"),
            row.get("hedge_freq"),
            row.get("weighting_scheme"),
            row.get("mode"),
        ]
        return "|".join(str(x) for x in parts)

    full["config_id"] = full.apply(cfg_id, axis=1)

    if "median_rank" not in lb.columns:
        return pd.DataFrame([{"note": "median_rank not found in ladder leaderboard"}])

    merged = full.merge(lb[["config_id", "median_rank"]], on="config_id", how="left")
    merged = merged[merged["median_rank"].notna() & merged["ir_benchmark"].notna()].copy()

    if len(merged) < 3:
        return pd.DataFrame(
            [{"note": "not enough matched configs for Spearman", "matched": len(merged)}]
        )

    merged["oos_ir_rank"] = merged["ir_benchmark"].rank(ascending=False, method="min")
    rho = merged[["median_rank", "oos_ir_rank"]].corr(method="spearman").iloc[0, 1]

    return pd.DataFrame(
        [
            {
                "comparison": "in_sample_median_rank_vs_full_oos_ir_rank",
                "spearman_rho": rho,
                "matched_configs": len(merged),
            }
        ]
    )


def main() -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)

    required = {"config_name", "eval_window", "ir_benchmark"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    real = df[df["config_type"] != "analytical_line"].copy()

    real = _rank_within_window(real, "ir_benchmark")
    if "sharpe" in real.columns:
        real = _rank_within_window(real, "sharpe")

    analytical = _make_analytical_median_rows(real)

    combined = pd.concat([real, analytical], ignore_index=True, sort=False)

    # Table 1: config x year with IR + rank
    yearly = combined[combined["eval_window"].isin(["2020", "2021", "2022", "2023"])].copy()

    cols_yearly = [
        "config_name",
        "config_type",
        "eval_window",
        "ir_benchmark",
        "ir_benchmark_rank",
        "sharpe",
        "sharpe_rank",
        "max_dd",
        "geom_avg_xs_r",
    ]
    cols_yearly = [c for c in cols_yearly if c in yearly.columns]

    yearly_out = yearly[cols_yearly].sort_values(
        ["eval_window", "ir_benchmark_rank"],
        na_position="last",
    )
    yearly_out.to_csv(RESULTS_DIR / "oos_config_year_ir_rank.csv", index=False)

    # Rank matrix by year
    rank_matrix = real[real["eval_window"].isin(["2020", "2021", "2022", "2023"])].pivot_table(
        index="config_name",
        columns="eval_window",
        values="ir_benchmark_rank",
        aggfunc="min",
    )
    rank_matrix.to_csv(RESULTS_DIR / "oos_ir_ranks_by_year.csv")

    # Full OOS table
    full = combined[combined["eval_window"] == "full_oos_2020_2023"].copy()
    cols_full = [
        "config_name",
        "config_type",
        "ir_benchmark",
        "ir_benchmark_rank",
        "sharpe",
        "sharpe_rank",
        "final_nav",
        "max_dd",
        "alpha_benchmark",
        "alpha_benchmark_pvalue",
        "geom_avg_xs_r",
        "factor_loadings_low_risk",
        "factor_loadings_momentum",
        "factor_loadings_size",
        "factor_loadings_quality",
        "factor_loadings_value",
        "factor_loadings_spx_rf",
    ]
    cols_full = [c for c in cols_full if c in full.columns]

    full_out = full[cols_full].sort_values("ir_benchmark", ascending=False, na_position="last")
    full_out.to_csv(RESULTS_DIR / "oos_full_results.csv", index=False)

    # Verdict table
    verdict_rows = []
    real_yearly = real[real["eval_window"].isin(["2020", "2021", "2022", "2023"])].copy()
    real_full = real[real["eval_window"] == "full_oos_2020_2023"].copy()

    for config_name, g in real_yearly.groupby("config_name"):
        full_row = real_full[real_full["config_name"] == config_name]

        if full_row.empty:
            continue

        full_ir = float(full_row["ir_benchmark"].iloc[0])
        full_rank = float(full_row["ir_benchmark_rank"].iloc[0])
        full_max_dd = float(full_row["max_dd"].iloc[0]) if "max_dd" in full_row.columns else np.nan

        top3_years = int((g["ir_benchmark_rank"] <= 3).sum())
        worst_year_count = int((g["ir_benchmark_rank"] == g["ir_benchmark_rank"].max()).sum())

        if full_ir > 0 and top3_years >= 2:
            status = "confirmed"
        elif full_ir < 0:
            status = "failed_negative_full_oos_ir"
        else:
            status = "mixed"

        verdict_rows.append(
            {
                "config_name": config_name,
                "config_type": g["config_type"].iloc[0],
                "full_oos_ir": full_ir,
                "full_oos_ir_rank": full_rank,
                "top3_years_by_ir": top3_years,
                "full_oos_max_dd": full_max_dd,
                "status": status,
            }
        )

    verdict = pd.DataFrame(verdict_rows).sort_values(
        ["status", "full_oos_ir_rank"],
        ascending=[True, True],
    )
    verdict.to_csv(RESULTS_DIR / "oos_verdict.csv", index=False)

    spearman = _spearman_is_vs_oos(real)
    spearman.to_csv(RESULTS_DIR / "oos_spearman_check.csv", index=False)

    print("Saved:")
    print("-", RESULTS_DIR / "oos_config_year_ir_rank.csv")
    print("-", RESULTS_DIR / "oos_ir_ranks_by_year.csv")
    print("-", RESULTS_DIR / "oos_full_results.csv")
    print("-", RESULTS_DIR / "oos_verdict.csv")
    print("-", RESULTS_DIR / "oos_spearman_check.csv")

    print("\nFull OOS results:")
    print(full_out.to_string(index=False))

    print("\nVerdict:")
    print(verdict.to_string(index=False))


if __name__ == "__main__":
    main()