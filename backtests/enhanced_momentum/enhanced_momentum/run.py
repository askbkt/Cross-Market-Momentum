from __future__ import annotations

import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from enhanced_momentum.config.project_experiment_config import ProjectExperimentConfig
from enhanced_momentum.config.project_trading_config import ProjectTradingConfig
from enhanced_momentum.data_loaders.registry import get_loader, load_market_config
from quant_pml.hedge.market_futures_hedge import MarketFuturesHedge
from quant_pml.runner import build_backtest

if TYPE_CHECKING:
    from quant_pml.strategies.base_strategy import BaseStrategy


# =========================
# Paths / caching helpers
# =========================
def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found in parents).")


def _run_id(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


# =========================
# Market config helpers
# =========================
def get_market_config(market: str = "us") -> dict[str, Any]:
    """Load market YAML config from the repository-level markets/ directory."""
    repo_root = _repo_root()
    config_dir = repo_root / "markets"
    return load_market_config(market, config_dir)


# =========================
# Backtest runner
# =========================
def run_backtest(
    strategy: BaseStrategy,
    rebal_freq: str = "ME",
    experiment_cfg: ProjectExperimentConfig | None = None,
    trading_cfg: ProjectTradingConfig | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    market: str = "us",
    plot: bool = True,
    return_runner: bool = False,
) -> pd.DataFrame:
    """Run a backtest for the given market.

    US uses the existing quant-pml runner via USLoader.build_dataset().
    MOEX/crypto loaders are intentionally left as Phase 2 implementations.
    """
    market = market.lower()
    market_config = get_market_config(market)

    if market == "us":
        return _run_backtest_us(
            strategy=strategy,
            rebal_freq=rebal_freq,
            experiment_cfg=experiment_cfg,
            trading_cfg=trading_cfg,
            market_config=market_config,
            start_date=start_date,
            end_date=end_date,
            plot=plot,
            return_runner=return_runner,
        )

    raise NotImplementedError(
        f"Backtest engine for market '{market}' is not implemented yet. "
        "Implement the market loader and engine in Phase 2."
    )


def _run_backtest_us(
    strategy: BaseStrategy,
    rebal_freq: str,
    experiment_cfg: ProjectExperimentConfig | None,
    trading_cfg: ProjectTradingConfig | None,
    market_config: dict[str, Any],
    start_date: pd.Timestamp | str | None,
    end_date: pd.Timestamp | str | None,
    plot: bool,
    return_runner: bool,
) -> pd.DataFrame:
    """US-specific backtest using the existing quant-pml pipeline."""
    loader = get_loader("us", market_config)

    hedger = None
    hedge_instrument = market_config.get("hedge_instrument", "spx_futures")
    if hedge_instrument:
        hedger = MarketFuturesHedge(market_name=market_config.get("market_proxy_col", "spx"))

    strategy_name = strategy.__class__.__name__
    cfg = experiment_cfg if experiment_cfg is not None else loader.build_experiment_config()
    cfg.HEDGE_FREQ = market_config.get("hedge_freq", getattr(cfg, "HEDGE_FREQ", "ME"))

    preprocessor, runner = build_backtest(
        experiment_config=cfg,
        trading_config=trading_cfg if trading_cfg is not None else ProjectTradingConfig(),
        rebal_freq=rebal_freq,
        dataset_builder_fn=loader.build_dataset,
        start=start_date,
        end=end_date,
    )

    res = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=hedger,
    )

    if plot:
        runner.plot_cumulative(strategy_name=strategy_name, include_factors=True)

    metrics = res.to_pandas()

    if return_runner:
        return metrics, runner

    return metrics


def _build_systematic_momentum(params: dict[str, Any]):
    """Build SystematicMomentum while passing only supported constructor args.

    This keeps run.py compatible with both the original coursework strategy
    and future strategy versions that may support extra parameters.
    """
    from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum

    candidate_kwargs = {
        "mode": params["mode"],
        "quantile": params["quantile"],
        "window_days": params["window_days"],
        "exclude_last_days": params["exclude_last_days"],
        "as_zscore": params["as_zscore"],
        "weighting_scheme": params["weighting_scheme"],
        "return_type": params.get("return_type"),
        "volatility_scaling": params.get("volatility_scaling"),
        "vol_window_days": params.get("vol_window_days"),
    }

    signature = inspect.signature(SystematicMomentum)
    accepted = set(signature.parameters)
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted and v is not None}
    return SystematicMomentum(**kwargs)


# =========================
# Main entrypoint
# =========================
if __name__ == "__main__":
    market = "us"
    market_config = get_market_config(market)
    grid = market_config.get("grid", {})

    params: dict[str, Any] = {
        "strategy": "SystematicMomentum",
        "market": market,
        "mode": grid.get("mode", "long_short"),
        "quantile": 0.12,
        "window_days": 252,
        "exclude_last_days": 21,
        "as_zscore": False,
        "weighting_scheme": grid.get("weighting_scheme", "equally_weighted"),
        "return_type": grid.get("return_type", "simple"),
        "volatility_scaling": grid.get("volatility_scaling", True),
        "vol_window_days": grid.get("vol_window_days", 21),
        "rebal_freq": market_config.get("rebal_freq", "ME"),
        "start_date": "2022-01-01",
        "end_date": None,
    }

    repo_root = _repo_root()
    results_dir = repo_root / market_config.get("results_dir", f"results/{market}")
    run_id = _run_id(params)

    out_dir = results_dir / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.parquet"

    meta = {
        "run_id": run_id,
        "market": market,
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    config_path.write_text(
        json.dumps({"params": params, "meta": meta}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if metrics_path.exists():
        print(f"[cache] {run_id} -> {metrics_path}")
        metrics = pd.read_parquet(metrics_path)
    else:
        print(f"[run] {run_id} computing...")

        sys_mom = _build_systematic_momentum(params)

        metrics = run_backtest(
            strategy=sys_mom,
            rebal_freq=params["rebal_freq"],
            market=market,
            start_date=pd.Timestamp(params["start_date"]),
            end_date=pd.Timestamp(params["end_date"]) if params["end_date"] else None,
            plot=True,
        )

        metrics.to_parquet(metrics_path)

    print(metrics)

