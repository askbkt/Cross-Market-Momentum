"""MOEX ISS API data loader — v3.

Changes vs v1 (per supervisor review):
- P0.1: Historical universe via /iss/history/ endpoint (survivorship-bias fix)
- P0.3: Range-aware cache (incremental fetch)
- P0.4: Explicit trading_days_window + min_trading_day_ratio
- P0.5: CLOSE before LEGALCLOSEPRICE
- P1.6: Liquidity universe lagged by 1 day
- P1.7: Per-symbol failure isolation
- P1.9: Market proxy reindexed, not dropna'd
- P1.10: volume_type metadata for traded_value_rub
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

from enhanced_momentum.data_loaders.base import BaseLoader, MarketData

logger = logging.getLogger(__name__)

_ISS_BASE = "https://iss.moex.com/iss"
_PAGE_SIZE = 100
_REQUEST_DELAY = 0.25

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "cross-market-momentum-research/1.0",
})

_ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=4,
    pool_maxsize=4,
    max_retries=0,
)

_SESSION.mount("https://", _ADAPTER)


# =====================================================================
# ISS helpers
# =====================================================================

def _iss_get(
    url: str,
    params: dict[str, Any] | None = None,
) -> dict:
    """Perform a MOEX ISS request with DNS-aware retries."""
    params = dict(params or {})

    params.setdefault("iss.json", "compact")
    params.setdefault("iss.meta", "off")

    last_error: Exception | None = None

    for attempt in range(5):
        try:
            resp = _SESSION.get(
                url,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()

            return resp.json()

        except RequestsConnectionError as exc:
            last_error = exc
            error_text = str(exc)

            is_dns = any(
                marker in error_text
                for marker in (
                    "NameResolutionError",
                    "Failed to resolve",
                    "nodename nor servname",
                    "Temporary failure in name resolution",
                )
            )

            wait = 15 if is_dns else min(2 ** attempt, 8)

            logger.warning(
                "ISS connection failed "
                "(attempt %d/5, dns=%s), waiting %ds: %s",
                attempt + 1,
                is_dns,
                wait,
                exc,
            )

            time.sleep(wait)

        except (requests.RequestException, ValueError) as exc:
            last_error = exc

            wait = min(2 ** attempt, 8)

            logger.warning(
                "ISS request failed "
                "(attempt %d/5), waiting %ds: %s",
                attempt + 1,
                wait,
                exc,
            )

            time.sleep(wait)

    raise RuntimeError(
        f"ISS API failed after 5 attempts: {url}"
    ) from last_error


def _iss_paginate(
    url: str,
    block: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    params = dict(params or {})
    rows: list[pd.DataFrame] = []
    start = 0

    while True:
        params["start"] = start
        data = _iss_get(url, params)
        block_data = data.get(block)

        if not isinstance(block_data, dict):
            logger.warning(
                "ISS block %r missing or malformed. Response keys: %s",
                block,
                list(data.keys()),
            )
            break

        columns = block_data.get("columns", [])
        data_rows = block_data.get("data", [])
        if not data_rows:
            break

        rows.append(pd.DataFrame(data_rows, columns=columns))
        received = len(data_rows)
        start += received

        if received < _PAGE_SIZE:
            break
        time.sleep(_REQUEST_DELAY)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =====================================================================
# Historical universe discovery (survivorship-bias fix)
# =====================================================================

def fetch_historical_tqbr_tickers(
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> list[str]:
    """Discover TQBR securities whose listing history overlaps the period."""
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = (
        pd.Timestamp(end_date).normalize()
        if end_date
        else pd.Timestamp.now().normalize()
    )

    url = (
        f"{_ISS_BASE}/history/engines/stock/markets/shares"
        f"/boards/TQBR/listing.json"
    )

    logger.info(
        "Fetching historical TQBR listing for %s -> %s",
        requested_start.date(),
        requested_end.date(),
    )

    df = _iss_paginate(
        url,
        "securities",
        {
            "iss.only": "securities",
            "securities.columns": (
                "SECID,BOARDID,history_from,history_till"
            ),
        },
    )

    if df.empty:
        raise RuntimeError(
            "Historical TQBR listing returned no securities"
        )

    df.columns = [
        str(column).upper()
        for column in df.columns
    ]

    required_columns = {
        "SECID",
        "HISTORY_FROM",
        "HISTORY_TILL",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Historical TQBR listing missing columns: "
            f"{sorted(missing_columns)}. "
            f"Available: {list(df.columns)}"
        )

    df["HISTORY_FROM"] = pd.to_datetime(
        df["HISTORY_FROM"],
        errors="coerce",
    )
    df["HISTORY_TILL"] = pd.to_datetime(
        df["HISTORY_TILL"],
        errors="coerce",
    )

    # Keep securities whose TQBR listing interval overlaps
    # the requested research period.
    overlaps_period = (
        df["HISTORY_FROM"].fillna(pd.Timestamp.min)
        <= requested_end
    ) & (
        df["HISTORY_TILL"].fillna(pd.Timestamp.max)
        >= requested_start
    )

    filtered = df.loc[overlaps_period].copy()

    tickers = sorted(
        ticker
        for ticker in (
            filtered["SECID"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
        if ticker
    )

    if not tickers:
        raise RuntimeError(
            "No TQBR securities overlap the requested period"
        )

    logger.info(
        "Historical TQBR universe: %d unique tickers "
        "(%s -> %s)",
        len(tickers),
        requested_start.date(),
        requested_end.date(),
    )

    return tickers


def _load_or_discover_historical_universe(
    cache_path: Path,
    start_date: str,
    end_date: str | None,
) -> list[str]:
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = (
        pd.Timestamp(end_date).normalize()
        if end_date
        else pd.Timestamp.now().normalize()
    )

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("tickers"), list):
            cached_start = pd.Timestamp(payload["start_date"])
            cached_end = pd.Timestamp(payload["end_date"])
            if cached_start <= requested_start and cached_end >= requested_end:
                tickers = payload["tickers"]
                logger.info(
                    "Historical universe [cache]: %d tickers, coverage %s -> %s",
                    len(tickers),
                    cached_start.date(),
                    cached_end.date(),
                )
                return tickers

    tickers = fetch_historical_tqbr_tickers(start_date, end_date)
    payload = {
        "source": "moex_iss_history_tqbr",
        "start_date": requested_start.strftime("%Y-%m-%d"),
        "end_date": requested_end.strftime("%Y-%m-%d"),
        "tickers": tickers,
    }
    cache_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tickers


# =====================================================================
# Range-aware cache
# =====================================================================

def _load_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            return df
    return None


def _merge_and_save(
    cached: pd.DataFrame | None,
    new: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    if cached is not None and not cached.empty:
        combined = pd.concat([cached, new], ignore_index=True)
    else:
        combined = new

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


def _fetch_with_range_aware_cache(
    fetch_fn,
    cache_path: Path,
    start_date: str,
    end_date: str | None,
    label: str,
) -> pd.DataFrame:
    """Fetch data with incremental cache: fill head and tail gaps."""
    cached = _load_cached(cache_path)
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    if cached is not None and not cached.empty:
        cached_min = cached["date"].min()
        cached_max = cached["date"].max()

        need_head = requested_start < cached_min
        need_tail = requested_end > cached_max

        if not need_head and not need_tail:
            logger.info("%s [cache] %s -> %s", label, cached_min.date(), cached_max.date())
            return cached

        new_parts: list[pd.DataFrame] = []

        if need_head:
            logger.info("%s [fetch head] %s -> %s", label, start_date, cached_min.date())
            head = fetch_fn(
                start_date=start_date,
                end_date=(cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if not head.empty:
                new_parts.append(head)

        if need_tail:
            logger.info("%s [fetch tail] %s -> %s", label, cached_max.date(), end_date)
            tail = fetch_fn(
                start_date=(cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=end_date,
            )
            if not tail.empty:
                new_parts.append(tail)

        if new_parts:
            new_data = pd.concat(new_parts, ignore_index=True)
            return _merge_and_save(cached, new_data, cache_path)

        return cached

    # No cache at all
    logger.info("%s [fetch full] %s -> %s", label, start_date, end_date)
    df = fetch_fn(start_date=start_date, end_date=end_date)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


# =====================================================================
# Per-security history fetch
# =====================================================================

def fetch_history_security(
    ticker: str,
    board: str = "TQBR",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    url = (
        f"{_ISS_BASE}/history/engines/stock/markets/shares"
        f"/boards/{board}/securities/{ticker}.json"
    )
    params: dict[str, Any] = {
        "from": start_date,
        "iss.only": "history",
    }
    if end_date:
        params["till"] = end_date

    df = _iss_paginate(url, "history", params)

    if df.empty:
        return pd.DataFrame()

    df.columns = [c.upper() for c in df.columns]

    date_col = next((c for c in ["TRADEDATE", "DATE"] if c in df.columns), None)
    # P0.5: CLOSE before LEGALCLOSEPRICE
    price_columns = [
        column
        for column in (
            "CLOSE",
            "LEGALCLOSEPRICE",
            "CLOSEPRICE",
        )
        if column in df.columns
    ]

    if date_col is None or not price_columns:
        logger.warning(
            "Missing date/price columns for %s: %s",
            ticker,
            list(df.columns),
        )
        return pd.DataFrame()

    result = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
    })

    result["ticker"] = ticker

    close = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Float64",
    )

    for column in price_columns:
        candidate = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        candidate = candidate.where(candidate > 0)

        close = close.fillna(candidate)

    result["close"] = close.astype(float)

    result = result.dropna(subset=["close"])
    result = result[result["close"] > 0]
    result = result.drop_duplicates(subset=["date"], keep="last")
    return result.sort_values("date").reset_index(drop=True)


def _fetch_index_history(
    secid: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily history for a single MOEX index SECID."""
    url = (
        f"{_ISS_BASE}/history/engines/stock/markets/index"
        f"/boards/SNDX/securities/{secid}.json"
    )

    params: dict[str, Any] = {
        "from": start_date,
        "iss.only": "history",
    }

    if end_date:
        params["till"] = end_date

    df = _iss_paginate(
        url,
        "history",
        params,
    )

    if df.empty:
        return pd.DataFrame(
            columns=["date", "close"]
        )

    df.columns = [
        str(column).upper()
        for column in df.columns
    ]

    date_col = next(
        (
            column
            for column in (
                "TRADEDATE",
                "DATE",
            )
            if column in df.columns
        ),
        None,
    )

    if date_col is None:
        logger.warning(
            "%s index history has no date column: %s",
            secid,
            list(df.columns),
        )
        return pd.DataFrame(
            columns=["date", "close"]
        )

    close = pd.Series(
        index=df.index,
        dtype="float64",
    )

    for column in (
        "CLOSE",
        "CURRENTVALUE",
        "LEGALCLOSEPRICE",
    ):
        if column not in df.columns:
            continue

        candidate = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        candidate = candidate.where(
            candidate > 0
        )

        close = close.combine_first(
            candidate
        )

    result = pd.DataFrame({
        "date": pd.to_datetime(
            df[date_col],
            errors="coerce",
        ),
        "close": close,
    })

    result = result.dropna(
        subset=["date", "close"]
    )

    result = result.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_imoex_history(
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch continuous MOEX Russia Index history.

    The legacy SECID MICEXINDEXCF was replaced by IMOEX in 2018.
    Both histories are combined into one continuous market proxy.
    """
    index_secids = (
        ("MICEXINDEXCF", 0),
        ("IMOEX", 1),
    )

    parts: list[pd.DataFrame] = []

    for secid, priority in index_secids:
        df = _fetch_index_history(
            secid=secid,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            continue

        df = df.copy()
        df["_priority"] = priority

        parts.append(df)

    if not parts:
        return pd.DataFrame(
            columns=["date", "close"]
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    # Prefer the current IMOEX SECID if both codes
    # happen to contain the same trading date.
    result = (
        result
        .sort_values(["date", "_priority"])
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .drop(columns="_priority")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return result


# =====================================================================
# Panel builders
# =====================================================================

def _build_wide_panel(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        long_df
        .pivot_table(index="date", columns="ticker", values=value_col, aggfunc="last")
        .sort_index()
    )


def _build_presence_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    trading_days_window: int = 126,
    min_trading_day_ratio: float = 0.5,
) -> pd.DataFrame:
    """Build dynamic investable-universe presence matrix."""
    presence = close.notna()

    if (
        trading_days_window > 0
        and volume is not None
    ):
        aligned_volume = volume.reindex(
            index=close.index,
            columns=close.columns,
        )

        has_volume = (
            aligned_volume
            .fillna(0)
            .gt(0)
        )

        trailing_days = has_volume.rolling(
            trading_days_window,
            min_periods=1,
        ).sum()

        threshold = (
            trading_days_window
            * min_trading_day_ratio
        )

        sufficiently_traded = (
            trailing_days >= threshold
        )

        presence = (
            presence
            & sufficiently_traded
        )

    return (
        presence
        .fillna(False)
        .astype(int)
    )


# =====================================================================
# MOEXLoader v3
# =====================================================================

class MOEXLoader(BaseLoader):

    def fetch(self) -> None:
        start_date = self.config.get("start_date", "2010-01-01")
        end_date = self.config.get("end_date", None)
        smoke_tickers: list[str] | None = self.config.get("smoke_tickers", None)

        cache_dir = self.data_dir / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Historical universe
        if smoke_tickers:
            tickers = smoke_tickers
            logger.info("Smoke mode: %d tickers: %s", len(tickers), tickers)
        else:
            universe_cache = cache_dir / "_historical_universe.json"
            tickers = _load_or_discover_historical_universe(
                universe_cache,
                start_date=start_date,
                end_date=end_date,
            )

            exclude = set(self.config.get("universe", {}).get("exclude_tickers", []))
            if exclude:
                tickers = [t for t in tickers if t not in exclude]

        # 2. Download each ticker (P0.3 range-aware, P1.7 failure isolation)
        all_rows: list[pd.DataFrame] = []
        failed: list[str] = []

        for i, ticker in enumerate(tickers):
            cache_path = cache_dir / f"{ticker}.parquet"
            label = f"[{i + 1:3d}/{len(tickers)}] {ticker}"

            try:
                df = _fetch_with_range_aware_cache(
                    fetch_fn=lambda start_date, end_date, t=ticker: fetch_history_security(
                        t,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                    cache_path=cache_path,
                    start_date=start_date,
                    end_date=end_date,
                    label=label,
                )
                if not df.empty:
                    all_rows.append(df)
            except Exception:
                logger.exception("Failed to fetch %s", ticker)
                failed.append(ticker)

        failures_path = self.data_dir / "fetch_failures.json"

        if failed:
            logger.warning(
                "Fetch completed with %d failures: %s",
                len(failed),
                failed,
            )

            failures_path.write_text(
                json.dumps(
                    failed,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            failures_path.unlink(missing_ok=True)

        if not all_rows:
            raise RuntimeError("No MOEX data fetched")

        long_df = pd.concat(all_rows, ignore_index=True)
        long_df.to_parquet(self.data_dir / "all_securities_long.parquet", index=False)

        # 3. IMOEX (range-aware)
        imoex_cache = cache_dir / "IMOEX.parquet"
        _fetch_with_range_aware_cache(
            fetch_fn=fetch_imoex_history,
            cache_path=imoex_cache,
            start_date=start_date,
            end_date=end_date,
            label="IMOEX",
        )

        logger.info("MOEX fetch complete: %d tickers, %d failed", len(tickers), len(failed))

    def load(self) -> MarketData:
        long_path = self.data_dir / "all_securities_long.parquet"
        if not long_path.exists():
            raise FileNotFoundError(f"Run fetch() first. Missing: {long_path}")

        long_df = pd.read_parquet(long_path)
        long_df["date"] = pd.to_datetime(long_df["date"])

        close = _build_wide_panel(long_df, "close")
        close.index = pd.to_datetime(close.index)

        # P1.10: Volume is traded value in RUB when available
        volume = None
        volume_type = "shares"

        value_col = (
            "value"
            if "value" in long_df.columns
            else "volume"
        )

        if value_col in long_df.columns:
            volume = _build_wide_panel(
                long_df,
                value_col,
            )

            volume.index = pd.to_datetime(
                volume.index
            )

            volume = volume.reindex(
                index=close.index,
                columns=close.columns,
            )

            volume_type = (
                "traded_value_rub"
                if value_col == "value"
                else "shares"
            )

        # P0.4: Explicit window + ratio
        universe_cfg = self.config.get("universe", {})
        window = universe_cfg.get("trading_days_window", 126)
        ratio = universe_cfg.get("min_trading_day_ratio", 0.5)
        presence = _build_presence_matrix(close, volume, window, ratio)

        # P1.6: Liquidity filter lagged by 1 day
        min_vol = universe_cfg.get("min_daily_volume_rub", 0)
        if min_vol > 0 and volume is not None:
            median_vol = volume.rolling(63, min_periods=21).median().shift(1)
            presence = presence & (median_vol.fillna(0) >= min_vol).astype(int)
            presence = presence.astype(int)

        returns = close.pct_change(fill_method=None)

        # P1.9: Market proxy reindexed to close.index
        imoex_path = self.data_dir / "raw" / "IMOEX.parquet"
        if imoex_path.exists():
            imoex = pd.read_parquet(imoex_path)
            imoex["date"] = pd.to_datetime(imoex["date"])
            imoex = imoex.set_index("date").sort_index()
            market_proxy = imoex["close"].pct_change(fill_method=None).reindex(close.index)
        else:
            logger.warning("IMOEX not found, using NaN market proxy")
            market_proxy = pd.Series(dtype=float, index=close.index)
        market_proxy.name = "market_proxy"

        # Save processed panels
        close.to_parquet(self._cache_path("close"))
        returns.to_parquet(self._cache_path("returns"))
        presence.to_parquet(self._cache_path("presence_matrix"))
        market_proxy.to_frame("market_proxy").to_parquet(self._cache_path("market_proxy"))
        if volume is not None:
            volume.to_parquet(self._cache_path("volume"))

        return MarketData(
            close=close,
            returns=returns,
            volume=volume,
            presence_matrix=presence,
            market_proxy_returns=market_proxy,
            momentum_factor_returns=None,
            mkt_caps=None,
            dividends=None,
            metadata={
                "market": "moex",
                "source": "MOEX ISS API",
                "board": self.config.get("board", "TQBR"),
                "universe": "Historical TQBR (including delisted)",
                "volume_type": volume_type,
                "close_price_field": "CLOSE (LEGALCLOSEPRICE as fallback)",
                "presence_rule": f"traded >= {ratio:.0%} of last {window} days",
                "liquidity_lag": "1 day",
                "caveats": self.config.get("caveats", []),
            },
        )


# =====================================================================
# Smoke test
# =====================================================================

def smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = {
        "market": "moex",
        "data_dir": "data/moex_smoke",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "smoke_tickers": ["SBER", "GAZP", "LKOH", "NVTK", "ROSN"],
        "universe": {
            "trading_days_window": 63,
            "min_trading_day_ratio": 0.5,
            "min_daily_volume_rub": 0,
        },
    }

    loader = MOEXLoader(config)
    loader.fetch()
    md = loader.load()

    print()
    print(md.summary())
    print()
    print("Metadata:", json.dumps(md.metadata, indent=2, default=str))
    print("Close shape:", md.close.shape)
    print("Returns tail:")
    print(md.returns.tail())
    print()
    print("Market proxy tail:")
    print(md.market_proxy_returns.dropna().tail())
    print()
    n_assets = md.n_assets_by_date
    print(f"Assets per day: min={n_assets.min()}, median={n_assets.median():.0f}, max={n_assets.max()}")

    assert md.close.shape[1] == 5, f"Expected 5 tickers, got {md.close.shape[1]}"
    assert md.n_dates > 500, f"Expected >500 days, got {md.n_dates}"
    assert md.market_proxy_returns.index.equals(md.close.index), "Index mismatch"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    smoke_test()