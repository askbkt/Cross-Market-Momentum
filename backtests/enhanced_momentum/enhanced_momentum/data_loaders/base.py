from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MarketData:
    """Unified market data container returned by all market loaders.

    Conventions:
    - close / returns / volume / presence_matrix are date x asset panels.
    - market_proxy_returns is a date-indexed return series for benchmark/regime logic.
    - momentum_factor_returns is optional and market-specific.
    """

    close: pd.DataFrame
    returns: pd.DataFrame
    volume: pd.DataFrame | None
    presence_matrix: pd.DataFrame
    market_proxy_returns: pd.Series
    momentum_factor_returns: pd.Series | None = None
    mkt_caps: pd.DataFrame | None = None
    dividends: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def market_proxy(self) -> pd.Series:
        """Alias used by the cross-market roadmap/DoD."""
        return self.market_proxy_returns

    @property
    def momentum_factor(self) -> pd.Series | None:
        """Alias used by the cross-market roadmap/DoD."""
        return self.momentum_factor_returns

    @property
    def close_df(self) -> pd.DataFrame:
        return self.close

    @property
    def returns_df(self) -> pd.DataFrame:
        return self.returns

    @property
    def volume_df(self) -> pd.DataFrame | None:
        return self.volume

    @property
    def n_assets_by_date(self) -> pd.Series:
        return self.presence_matrix.sum(axis=1)

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        idx = self.returns.index if len(self.returns.index) else self.close.index
        return idx.min(), idx.max()

    @property
    def n_dates(self) -> int:
        return len(self.returns.index) if len(self.returns.index) else len(self.close.index)

    def as_tuple(self):
        """Return the exact tuple requested in the roadmap DoD."""
        return (
            self.close,
            self.returns,
            self.volume,
            self.presence_matrix,
            self.market_proxy_returns,
        )

    def summary(self) -> str:
        start, end = self.date_range
        n_assets = self.presence_matrix.shape[1]
        avg_assets = self.n_assets_by_date.mean()
        lines = [
            f"Market: {self.metadata.get('market', 'unknown')}",
            f"Period: {start.date()} -> {end.date()} ({self.n_dates} trading days)",
            f"Assets: {n_assets} total, {avg_assets:.0f} avg available per day",
            f"Has volume: {self.volume is not None}",
            f"Has mkt_caps: {self.mkt_caps is not None}",
            f"Has momentum_factor: {self.momentum_factor_returns is not None}",
        ]
        return "\n".join(lines)


class BaseLoader(ABC):
    """Abstract base class for market data loaders."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.market: str = config.get("market", "unknown")
        self.data_dir: Path = Path(config.get("data_dir", f"data/{self.market}"))

    @abstractmethod
    def fetch(self) -> None:
        """Download / refresh raw data from source. Idempotent."""
        ...

    @abstractmethod
    def load(self) -> MarketData:
        """Load processed data into MarketData. Assumes fetch() was called."""
        ...

    def fetch_and_load(self) -> MarketData:
        """Convenience: fetch then load."""
        self.fetch()
        return self.load()

    def _cache_path(self, name: str) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / f"{name}.parquet"

