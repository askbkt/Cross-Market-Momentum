from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from enhanced_momentum.data_loaders.base import BaseLoader, MarketData


_REGISTRY: dict[str, type[BaseLoader]] = {}


def register(name: str):
    """Decorator to register a loader class by market name."""

    def wrapper(cls: type[BaseLoader]):
        _REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def get_loader(market: str, config: dict[str, Any] | None = None) -> BaseLoader:
    """Instantiate a loader for the given market."""
    market = market.lower()
    if config is None:
        config = load_market_config(market)

    if market not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise ValueError(f"Unknown market '{market}'. Available: {available}")

    return _REGISTRY[market](config)


def load_market_config(market: str, config_dir: str | Path = "markets") -> dict[str, Any]:
    """Load market YAML config."""
    path = Path(config_dir) / f"{market.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Market config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Empty market config: {path}")
    return config


def load_market(market: str, config_dir: str | Path = "markets") -> MarketData:
    """One-liner: load config -> instantiate loader -> fetch & load."""
    config = load_market_config(market, config_dir)
    loader = get_loader(market, config)
    return loader.fetch_and_load()


# Auto-register built-in loaders on import.
def _auto_register() -> None:
    from enhanced_momentum.data_loaders.us_loader import USLoader

    _REGISTRY["us"] = USLoader

    try:
        from enhanced_momentum.data_loaders.moex_loader import MOEXLoader

        _REGISTRY["moex"] = MOEXLoader
    except ImportError:
        pass

    try:
        from enhanced_momentum.data_loaders.binance_loader import BinanceLoader

        _REGISTRY["crypto"] = BinanceLoader
    except ImportError:
        pass


_auto_register()
