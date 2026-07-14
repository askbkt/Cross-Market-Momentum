from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enhanced_momentum.config.project_experiment_config import ProjectExperimentConfig
from enhanced_momentum.data_loaders.base import BaseLoader, MarketData
from quant_pml.dataset.dataset_data import DatasetData

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_source_dir(config: dict[str, Any]) -> Path:
    """Resolve the configured US data directory against the repository root."""
    source = Path(config.get("source_dir", "data/datasets")).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    return source.resolve()

class USLoader(BaseLoader):
    """Loader for the existing US Russell 3000 parquet dataset.

    This loader does not download anything. It wraps the supervisor-provided
    parquet dataset and exposes it through two interfaces:

    1. load() -> MarketData for cross-market analytics and sanity checks.
    2. build_dataset(config) -> DatasetData for quant-pml.runner.build_backtest.
    """

    def fetch(self) -> None:
        source = _resolve_source_dir(self.config)
        required = [
            self.config.get("df_filename", "top3000_data_df.parquet"),
            self.config.get("presence_filename", "top3000_presence_matrix.parquet"),
        ]
        for filename in required:
            path = source / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"US data file not found: {path}. "
                    f"Place the supervisor's parquet files in {source}/"
                )

    def build_experiment_config(self) -> ProjectExperimentConfig:
        """Build the quant-pml experiment config from markets/us.yaml."""
        cfg = ProjectExperimentConfig()

        source_dir = _resolve_source_dir(self.config)
        cfg.PREFIX = self.config.get("prefix", "")
        cfg.PATH_OUTPUT = source_dir
        cfg.DF_FILENAME = self.config.get("df_filename", "top3000_data_df.parquet")
        cfg.DIVIDENDS_FILENAME = self.config.get("dividends_filename", "top3000_dividends.parquet")
        cfg.MKT_CAPS_FILENAME = self.config.get("mkt_caps_filename", "top3000_market_caps.parquet")
        cfg.PRESENCE_MATRIX_FILENAME = self.config.get(
            "presence_filename",
            "top3000_presence_matrix.parquet",
        )
        cfg.VOLUMES_FILENAME = self.config.get("volumes_filename", None)
        cfg.HEDGE_FREQ = self.config.get("hedge_freq", getattr(cfg, "HEDGE_FREQ", "ME"))

        return cfg

    def build_dataset(self, config: Any) -> DatasetData:
        """Adapter for quant-pml.runner.build_backtest.

        build_backtest expects dataset_builder_fn(config) -> DatasetData.
        The original data panel already contains asset price columns plus factor,
        RF, and hedge/proxy columns expected by quant-pml. Therefore this method
        returns the raw data panel, not only the asset-only close panel.
        """
        data, presence, mkt_caps, dividends = self._read_source_files()

        return DatasetData(
            data=data,
            presence_matrix=presence,
            mkt_caps=mkt_caps,
            dividends=dividends,
            volumes=None,
            targets=None,
            macro_features=None,
            asset_features=None,
        )

    def load(self) -> MarketData:
        data, presence, mkt_caps, dividends = self._read_source_files()

        market_proxy_col = self.config.get("market_proxy_col", "spx")
        momentum_col = self.config.get("momentum_factor_col", "momentum")

        market_proxy = (
            pd.to_numeric(data[market_proxy_col], errors="coerce").dropna()
            if market_proxy_col in data.columns
            else pd.Series(index=data.index, dtype=float)
        )
        market_proxy.name = "market_proxy"

        momentum_factor = (
            pd.to_numeric(data[momentum_col], errors="coerce").dropna()
            if momentum_col in data.columns
            else None
        )
        if momentum_factor is not None:
            momentum_factor.name = "momentum_factor"

        asset_cols = self._infer_asset_columns(data, presence)
        close = data.loc[:, asset_cols].copy() if asset_cols else pd.DataFrame(index=data.index)
        returns = close.pct_change(fill_method=None) if not close.empty else pd.DataFrame(index=data.index)

        return MarketData(
            close=close,
            returns=returns,
            volume=None,
            presence_matrix=presence,
            market_proxy_returns=market_proxy,
            momentum_factor_returns=momentum_factor,
            mkt_caps=mkt_caps,
            dividends=dividends,
            metadata={
                "market": "us",
                "source": str(_resolve_source_dir(self.config)),
                "universe": "Russell 3000",
                "description": "Supervisor-provided US equity dataset",
            },
        )

    def _read_source_files(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        source = _resolve_source_dir(self.config)

        df_filename = self.config.get("df_filename", "top3000_data_df.parquet")
        presence_filename = self.config.get("presence_filename", "top3000_presence_matrix.parquet")
        mkt_caps_filename = self.config.get("mkt_caps_filename", "top3000_market_caps.parquet")
        dividends_filename = self.config.get("dividends_filename", "top3000_dividends.parquet")

        data = pd.read_parquet(source / df_filename)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data.columns = data.columns.astype(str)

        presence = pd.read_parquet(source / presence_filename)
        presence.index = pd.to_datetime(presence.index)
        presence = presence.sort_index()
        presence.columns = presence.columns.astype(str)

        mkt_caps = None
        mkt_caps_path = source / mkt_caps_filename
        if mkt_caps_path.exists():
            mkt_caps = pd.read_parquet(mkt_caps_path)
            mkt_caps.index = pd.to_datetime(mkt_caps.index)
            mkt_caps = mkt_caps.sort_index()
            mkt_caps.columns = mkt_caps.columns.astype(str)

        dividends = None
        dividends_path = source / dividends_filename
        if dividends_path.exists():
            dividends = pd.read_parquet(dividends_path)
            dividends.index = pd.to_datetime(dividends.index)
            dividends = dividends.sort_index()
            dividends.columns = dividends.columns.astype(str)

        return data, presence, mkt_caps, dividends

    @staticmethod
    def _infer_asset_columns(data: pd.DataFrame, presence: pd.DataFrame) -> list[str]:
        """Use presence_matrix columns as the authoritative asset universe."""
        presence_cols = [str(c) for c in presence.columns]
        data_cols = set(map(str, data.columns))
        matched = [c for c in presence_cols if c in data_cols]
        if matched:
            return matched

        # Conservative fallback for legacy panels with numeric security IDs.
        return [
            str(c)
            for c in data.columns
            if str(c).isdigit() or isinstance(c, (int, np.integer))
        ]