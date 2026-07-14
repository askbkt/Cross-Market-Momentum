"""Unified data quality report generator.

Produces comparable markdown reports for US / MOEX / crypto markets:

    reports/data_quality/us_data_quality.md
    reports/data_quality/moex_data_quality.md
    reports/data_quality/crypto_data_quality.md

Plus a machine-readable summary:

    reports/data_quality/summary.csv

Usage:
    PYTHONPATH=. python enhanced_momentum/analysis/data_quality.py --markets moex crypto
    PYTHONPATH=. python enhanced_momentum/analysis/data_quality.py --markets us moex crypto
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from enhanced_momentum.data_loaders.registry import load_market

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MARKETS_DIR = PROJECT_ROOT / "markets"
REPORTS_DIR = (
    PROJECT_ROOT
    / "backtests"
    / "enhanced_momentum"
    / "reports"
    / "data_quality"
)

# Extreme return thresholds per market (daily)
EXTREME_THRESHOLDS = {
    "us": 0.25,
    "moex": 0.30,
    "crypto": 0.50,
}

# Annualization convention by market.
# Crypto trades seven days per week, while US and MOEX use trading-day convention.
ANNUALIZATION_DAYS = {
    "us": 252,
    "moex": 252,
    "crypto": 365,
}


# =====================================================================
# Section builders — each returns (markdown_str, dict_of_key_metrics)
# =====================================================================


def _sec_market_coverage(md_data) -> tuple[str, dict]:
    start, end = md_data.date_range
    n_dates = md_data.n_dates
    n_assets_total = md_data.presence_matrix.shape[1]

    years = pd.Series(md_data.returns.index.year)
    n_years = years.nunique()

    lines = [
        "## 1. Market Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| First date | {start.date()} |",
        f"| Last date | {end.date()} |",
        f"| Trading days | {n_dates:,} |",
        f"| Calendar years | {n_years} |",
        f"| Total assets (ever) | {n_assets_total} |",
        f"| Avg trading days/year | {n_dates / max(n_years, 1):.0f} |",
        "",
    ]
    metrics = {
        "first_date": str(start.date()),
        "last_date": str(end.date()),
        "n_trading_days": n_dates,
        "n_assets_total": n_assets_total,
    }
    return "\n".join(lines), metrics


def _sec_assets_per_year(md_data) -> tuple[str, dict]:
    presence = md_data.presence_matrix
    by_year = presence.sum(axis=1).groupby(presence.index.year)

    stats = pd.DataFrame(
        {
            "mean": by_year.mean().round(0).astype(int),
            "min": by_year.min().astype(int),
            "max": by_year.max().astype(int),
        }
    )

    lines = [
        "## 2. Assets Available Per Year",
        "",
        "| Year | Mean | Min | Max |",
        "|------|------|-----|-----|",
    ]
    for year, row in stats.iterrows():
        lines.append(f"| {year} | {row['mean']} | {row['min']} | {row['max']} |")
    lines.append("")

    overall = presence.sum(axis=1)
    metrics = {
        "avg_assets_per_day": float(overall.mean()),
        "min_assets_per_day": int(overall.min()),
        "max_assets_per_day": int(overall.max()),
    }
    return "\n".join(lines), metrics


def _sec_asset_lifetimes(md_data) -> tuple[str, dict]:
    close = md_data.close

    first_dates = close.apply(lambda col: col.first_valid_index()).dropna()
    last_dates = close.apply(lambda col: col.last_valid_index()).dropna()

    common_assets = first_dates.index.intersection(last_dates.index)
    first_dates = first_dates.loc[common_assets]
    last_dates = last_dates.loc[common_assets]

    calendar_lifetimes = (last_dates - first_dates).dt.days
    history_obs = close[common_assets].notna().sum(axis=0)

    if calendar_lifetimes.empty or history_obs.empty:
        return (
            "## 3. Asset Lifetimes\n\nNo assets with usable price history.\n",
            {
                "median_lifetime_days": None,
                "median_history_observations": None,
                "pct_assets_under_126_obs": None,
                "pct_assets_under_252_obs": None,
            },
        )

    under_126 = history_obs < 126
    under_252 = history_obs < 252

    lines = [
        "## 3. Asset Lifetimes",
        "",
        "### Calendar lifetime",
        "",
        "| Statistic | Calendar days | Years |",
        "|-----------|---------------|-------|",
        f"| Median lifetime | {calendar_lifetimes.median():.0f} | {calendar_lifetimes.median() / 365:.1f} |",
        f"| Mean lifetime | {calendar_lifetimes.mean():.0f} | {calendar_lifetimes.mean() / 365:.1f} |",
        f"| Min lifetime | {calendar_lifetimes.min():.0f} | {calendar_lifetimes.min() / 365:.1f} |",
        f"| Max lifetime | {calendar_lifetimes.max():.0f} | {calendar_lifetimes.max() / 365:.1f} |",
        "",
        "### Valid price observations",
        "",
        "| Statistic | Observations |",
        "|-----------|--------------|",
        f"| Median valid observations | {history_obs.median():.0f} |",
        f"| Mean valid observations | {history_obs.mean():.0f} |",
        f"| Min valid observations | {history_obs.min():.0f} |",
        f"| Max valid observations | {history_obs.max():.0f} |",
        "",
        f"Assets with <126 valid observations: {under_126.sum()} ({under_126.mean():.1%})",
        "",
        f"Assets with <252 valid observations: {under_252.sum()} ({under_252.mean():.1%})",
        "",
    ]
    metrics = {
        "median_lifetime_days": float(calendar_lifetimes.median()),
        "median_history_observations": float(history_obs.median()),
        "pct_assets_under_126_obs": float(under_126.mean()),
        "pct_assets_under_252_obs": float(under_252.mean()),
    }
    return "\n".join(lines), metrics


def _sec_listing_delisting(md_data) -> tuple[str, dict]:
    close = md_data.close
    end_date = close.index.max()

    first_dates = close.apply(lambda col: col.first_valid_index()).dropna()
    last_dates = close.apply(lambda col: col.last_valid_index()).dropna()

    listings_by_year = first_dates.dt.year.value_counts().sort_index()

    # This is an end-of-panel inactivity heuristic, not a formal delisting flag.
    inactive_mask = last_dates < (end_date - pd.Timedelta(days=30))
    exits_by_year = last_dates[inactive_mask].dt.year.value_counts().sort_index()

    all_years = sorted(set(listings_by_year.index) | set(exits_by_year.index))

    lines = [
        "## 4. Listings / Historical Exits by Year",
        "",
        "An asset is classified as a historical exit when its last price observation is more than "
        "30 calendar days before the panel end. This is an inactivity heuristic, not necessarily a "
        "formal exchange delisting event.",
        "",
        "| Year | New listings | Historical exits |",
        "|------|--------------|------------------|",
    ]
    for year in all_years:
        n_list = int(listings_by_year.get(year, 0))
        n_exit = int(exits_by_year.get(year, 0))
        lines.append(f"| {year} | {n_list} | {n_exit} |")

    n_inactive_total = int(inactive_mask.sum())
    n_active = len(last_dates) - n_inactive_total
    pct_inactive = n_inactive_total / max(len(last_dates), 1)

    lines += [
        "",
        f"**Historical exits before panel end: {n_inactive_total}**",
        f"**Assets still observed near panel end: {n_active}**",
        "",
        f"Historical coverage check: {pct_inactive:.1%} of assets have no observations within the "
        "final 30 calendar days of the panel. The presence of inactive historical assets indicates "
        "that the universe is not restricted to end-of-sample survivors.",
        "",
    ]
    metrics = {
        "n_historical_exits": n_inactive_total,
        "n_active_near_panel_end": n_active,
        "pct_historical_exits": float(pct_inactive),
    }
    return "\n".join(lines), metrics


def _sec_missingness(md_data) -> tuple[str, dict]:
    close = md_data.close

    # Missingness INSIDE each asset's active lifetime.
    gaps = []
    for col in close.columns:
        s = close[col]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None or last is None:
            continue

        active = s.loc[first:last]
        n_missing = int(active.isna().sum())
        gaps.append(
            {
                "asset": col,
                "active_days": len(active),
                "missing": n_missing,
                "missing_pct": n_missing / max(len(active), 1),
            }
        )

    gaps_df = pd.DataFrame(gaps)
    if gaps_df.empty:
        return (
            "## 5. Missingness (within active lifetime)\n\nNo assets with usable history.\n",
            {
                "median_missing_pct": None,
                "pct_assets_no_gaps": None,
            },
        )

    zero_gap_mask = gaps_df["missing"] == 0

    lines = [
        "## 5. Missingness (within active lifetime)",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Assets with zero gaps | {zero_gap_mask.sum()} ({zero_gap_mask.mean():.1%}) |",
        f"| Median missing % | {gaps_df['missing_pct'].median():.2%} |",
        f"| Mean missing % | {gaps_df['missing_pct'].mean():.2%} |",
        f"| Worst asset missing % | {gaps_df['missing_pct'].max():.2%} |",
        "",
    ]

    worst = gaps_df.nlargest(5, "missing_pct")
    if not worst.empty and worst["missing_pct"].iloc[0] > 0.01:
        lines += ["Worst 5 assets by gap share:", ""]
        lines += [
            "| Asset | Active days | Missing | % |",
            "|-------|-------------|---------|---|",
        ]
        for _, row in worst.iterrows():
            lines.append(
                f"| {row['asset']} | {int(row['active_days'])} | "
                f"{int(row['missing'])} | {row['missing_pct']:.1%} |"
            )
        lines.append("")

    metrics = {
        "median_missing_pct": float(gaps_df["missing_pct"].median()),
        "pct_assets_no_gaps": float(zero_gap_mask.mean()),
    }
    return "\n".join(lines), metrics


def _sec_return_distribution(md_data, market: str) -> tuple[str, dict]:
    returns = md_data.returns
    flat = returns.to_numpy().ravel()
    flat = flat[~np.isnan(flat)]

    if flat.size == 0:
        return (
            "## 6. Daily Return Distribution (pooled)\n\nNo valid return observations.\n",
            {
                "q001": None,
                "q999": None,
                "median_ann_vol": None,
            },
        )

    quantiles = [0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
    q_vals = np.quantile(flat, quantiles)

    lines = [
        "## 6. Daily Return Distribution (pooled)",
        "",
        f"Total daily return observations: {len(flat):,}",
        "",
        "| Quantile | Return |",
        "|----------|--------|",
    ]
    for q, value in zip(quantiles, q_vals):
        lines.append(f"| {q:.1%} | {value:+.2%} |")

    ann_days = ANNUALIZATION_DAYS.get(market, 252)
    ann_vol_med = np.nanmedian(returns.std()) * np.sqrt(ann_days)

    lines += [
        "",
        f"Annualization convention: {ann_days} days/year",
        "",
        f"Median asset annualized vol: {ann_vol_med:.1%}",
        "",
    ]
    metrics = {
        "q001": float(q_vals[0]),
        "q999": float(q_vals[-1]),
        "median_ann_vol": float(ann_vol_med),
    }
    return "\n".join(lines), metrics


def _sec_extreme_returns(md_data, market: str) -> tuple[str, dict]:
    returns = md_data.returns
    threshold = EXTREME_THRESHOLDS.get(market, 0.25)

    extreme_mask = returns.abs() > threshold
    n_extreme = int(extreme_mask.sum().sum())
    total_obs = int(returns.notna().sum().sum())

    by_year = extreme_mask.sum(axis=1).groupby(returns.index.year).sum()
    median_events = float(by_year.median()) if not by_year.empty else 0.0
    marker_threshold = max(median_events * 3, 5)

    lines = [
        f"## 7. Extreme Daily Returns (|r| > {threshold:.0%})",
        "",
        f"Total: {n_extreme:,} events ({n_extreme / max(total_obs, 1):.3%} of observations)",
        "",
        "| Year | Extreme events |",
        "|------|----------------|",
    ]
    for year, count in by_year.items():
        marker = " ⚠" if count > marker_threshold else ""
        lines.append(f"| {year} | {int(count)}{marker} |")

    lines += [
        "",
        f"⚠ marks years with more than max(3× median, 5) extreme events; current threshold: "
        f">{marker_threshold:.0f} events.",
        "",
    ]
    metrics = {
        "n_extreme_returns": n_extreme,
        "extreme_pct": float(n_extreme / max(total_obs, 1)),
    }
    return "\n".join(lines), metrics


def _sec_volume(md_data) -> tuple[str, dict]:
    if md_data.volume is None:
        return "## 8. Volume\n\nNo volume data available.\n", {"has_volume": False}

    volume = md_data.volume
    vol_type = md_data.metadata.get("volume_type", "unknown")

    daily_totals = volume.sum(axis=1)
    med_asset_vol = volume.median()
    total_median_volume = med_asset_vol.sum()
    top10_share = med_asset_vol.nlargest(10).sum() / max(total_median_volume, 1e-12)

    lines = [
        "## 8. Volume Distribution",
        "",
        f"Volume type: `{vol_type}`",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Median daily total volume | {daily_totals.median():,.0f} |",
        f"| Median per-asset daily volume | {med_asset_vol.median():,.0f} |",
        f"| Top-decile asset median volume | {med_asset_vol.quantile(0.9):,.0f} |",
        f"| Bottom-decile asset median volume | {med_asset_vol.quantile(0.1):,.0f} |",
        f"| Concentration: top-10 assets' share of total | {top10_share:.1%} |",
        "",
    ]
    metrics = {
        "has_volume": True,
        "volume_type": vol_type,
        "top10_volume_share": float(top10_share),
    }
    return "\n".join(lines), metrics


def _sec_presence_coverage(md_data) -> tuple[str, dict]:
    presence = md_data.presence_matrix
    close = md_data.close

    # How much of the close panel does presence actually allow?
    covered = int((presence.astype(bool) & close.notna()).sum().sum())
    total_close = int(close.notna().sum().sum())
    coverage_pct = covered / max(total_close, 1)

    lines = [
        "## 9. Presence Matrix Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Presence rule | {md_data.metadata.get('presence_rule', 'n/a')} |",
        f"| Liquidity lag | {md_data.metadata.get('liquidity_lag', 'n/a')} |",
        f"| Close observations allowed by presence | {covered:,} / {total_close:,} ({coverage_pct:.1%}) |",
        "",
    ]
    metrics = {"presence_coverage_pct": float(coverage_pct)}
    return "\n".join(lines), metrics


def _sec_proxy_coverage(md_data, market: str) -> tuple[str, dict]:
    proxy = md_data.market_proxy_returns
    total = len(proxy)
    n_valid = int(proxy.notna().sum())
    ann_days = ANNUALIZATION_DAYS.get(market, 252)

    ann_vol = proxy.std() * np.sqrt(ann_days) if n_valid > 20 else np.nan
    ann_mean = proxy.mean() * ann_days if n_valid > 20 else np.nan

    lines = [
        "## 10. Market Proxy Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Valid observations | {n_valid:,} / {total:,} ({n_valid / max(total, 1):.1%}) |",
        f"| Annualization convention | {ann_days} days/year |",
        f"| Annualized mean | {ann_mean:.2%} |"
        if not np.isnan(ann_mean)
        else "| Annualized mean | n/a |",
        f"| Annualized vol | {ann_vol:.2%} |"
        if not np.isnan(ann_vol)
        else "| Annualized vol | n/a |",
        "",
    ]
    metrics = {
        "proxy_coverage_pct": float(n_valid / max(total, 1)),
        "proxy_ann_vol": float(ann_vol) if not np.isnan(ann_vol) else None,
    }
    return "\n".join(lines), metrics


def _sec_investable_universe(md_data) -> tuple[str, dict]:
    n_assets = md_data.presence_matrix.sum(axis=1)

    if n_assets.empty:
        return (
            "## 11. Investable Universe Diagnostics\n\nNo presence observations available.\n",
            {
                "median_investable_assets": None,
                "p10_investable_assets": None,
                "days_under_10_assets": None,
                "days_under_20_assets": None,
            },
        )

    median_assets = float(n_assets.median())
    p10_assets = float(n_assets.quantile(0.10))
    days_under_10 = int((n_assets < 10).sum())
    days_under_20 = int((n_assets < 20).sum())

    lines = [
        "## 11. Investable Universe Diagnostics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Historical price assets | {md_data.close.shape[1]} |",
        f"| Mean investable assets | {n_assets.mean():.1f} |",
        f"| Median investable assets | {median_assets:.0f} |",
        f"| 10th percentile universe size | {p10_assets:.0f} |",
        f"| Minimum universe size | {n_assets.min():.0f} |",
        f"| Maximum universe size | {n_assets.max():.0f} |",
        f"| Days with <10 investable assets | {days_under_10:,} |",
        f"| Days with <20 investable assets | {days_under_20:,} |",
        "",
    ]
    metrics = {
        "median_investable_assets": median_assets,
        "p10_investable_assets": p10_assets,
        "days_under_10_assets": days_under_10,
        "days_under_20_assets": days_under_20,
    }
    return "\n".join(lines), metrics


# =====================================================================
# Report generator
# =====================================================================


def generate_report(market: str) -> dict:
    """Generate data quality report for one market. Returns key metrics."""
    logger.info("Loading market: %s", market)
    md_data = load_market(
        market,
        config_dir=MARKETS_DIR,
    )

    sections = [
        _sec_market_coverage(md_data),
        _sec_assets_per_year(md_data),
        _sec_asset_lifetimes(md_data),
        _sec_listing_delisting(md_data),
        _sec_missingness(md_data),
        _sec_return_distribution(md_data, market),
        _sec_extreme_returns(md_data, market),
        _sec_volume(md_data),
        _sec_presence_coverage(md_data),
        _sec_proxy_coverage(md_data, market),
        _sec_investable_universe(md_data),
    ]

    header = [
        f"# Data Quality Report: {market.upper()}",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: {md_data.metadata.get('source', 'n/a')}",
        f"Universe: {md_data.metadata.get('universe', 'n/a')}",
        "",
    ]

    caveats = md_data.metadata.get("caveats", [])
    if caveats:
        header += ["**Known caveats:**", ""]
        header += [f"- {c}" for c in caveats]
        header += [""]

    body_parts = [text for text, _ in sections]
    report = "\n".join(header) + "\n" + "\n".join(body_parts)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{market}_data_quality.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Report written: %s", out_path)

    # Merge all metrics.
    all_metrics = {"market": market}
    for _, metrics in sections:
        all_metrics.update(metrics)
    return all_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate data quality reports")
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["moex", "crypto"],
        choices=["us", "moex", "crypto"],
    )
    args = parser.parse_args()

    summary_rows = []
    for market in args.markets:
        try:
            metrics = generate_report(market)
            summary_rows.append(metrics)
        except Exception:
            logger.exception("Failed to generate report for %s", market)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = REPORTS_DIR / "summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Cross-market summary: %s", summary_path)
        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
