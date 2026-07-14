"""Binance spot daily OHLCV loader — v3.

Key properties:
- Historical symbol discovery from Binance Public Data archive directory
- Historical daily data loaded from archived monthly 1d kline ZIP files
- Includes delisted symbols when archive files exist
- Range-aware raw parquet cache
- Liquidity universe lagged by 1 day
- Per-symbol failure isolation + fetch_failures.json
- HTTP 418/429 handling
- Market proxy aligned to close.index
"""

from __future__ import annotations

from requests.exceptions import ConnectionError as RequestsConnectionError

import io
import json
import logging
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

import pandas as pd
import requests

from enhanced_momentum.data_loaders.base import BaseLoader, MarketData

logger = logging.getLogger(__name__)

_BINANCE_BASE = "https://api.binance.com"
_BINANCE_DATA_BASE = "https://data.binance.vision"
_BINANCE_S3_BASE = (
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
)
_KLINE_LIMIT = 1000
_REQUEST_DELAY = 0.15

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

_STABLECOIN_BASES = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP",
    "USDD", "GUSD", "FRAX", "LUSD", "SUSD", "HUSD", "UST",
    "USTC", "AEUR", "EURI", "EURITE", "PYUSD",
}
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def _request_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 60,
    attempts: int = 5,
) -> requests.Response:
    """HTTP GET with DNS-aware retries."""
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = _SESSION.get(
                url,
                params=params,
                timeout=timeout,
            )

            if response.status_code in {418, 429}:
                wait = int(
                    response.headers.get(
                        "Retry-After",
                        15,
                    )
                )

                logger.warning(
                    "Binance rate limit status=%d, "
                    "waiting %ds",
                    response.status_code,
                    wait,
                )

                time.sleep(wait)
                continue

            return response

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

            wait = 15 if is_dns else min(
                2 ** attempt,
                8,
            )

            logger.warning(
                "Binance connection failed "
                "(attempt %d/%d, dns=%s), "
                "waiting %ds: %s",
                attempt + 1,
                attempts,
                is_dns,
                wait,
                exc,
            )

            time.sleep(wait)

        except requests.RequestException as exc:
            last_error = exc

            wait = min(
                2 ** attempt,
                8,
            )

            logger.warning(
                "Binance request failed "
                "(attempt %d/%d), "
                "waiting %ds: %s",
                attempt + 1,
                attempts,
                wait,
                exc,
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Binance request failed after {attempts} attempts: {url}"
    ) from last_error


# =====================================================================
# API helpers
# =====================================================================

def _ts_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _binance_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{_BINANCE_BASE}{endpoint}"
    for attempt in range(3):
        try:
            resp = _request_with_retry(url, params=params or {}, timeout=30)
            # P1.8: Handle both 429 and 418
            if resp.status_code in {418, 429}:
                wait = int(resp.headers.get("Retry-After", 10))
                logger.warning("Binance rate limit status=%d, waiting %ds", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Binance request failed (attempt %d): %s", attempt + 1, e)
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Binance API failed after 3 attempts: {endpoint}")


# =====================================================================
# Binance Public Data archive helpers / historical universe
# =====================================================================


def _s3_list_common_prefixes(prefix: str) -> list[str]:
    """List directory-like prefixes from Binance Public Data S3 bucket."""
    namespace = {
        "s3": "http://s3.amazonaws.com/doc/2006-03-01/",
    }

    common_prefixes: list[str] = []
    continuation_token: str | None = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "delimiter": "/",
        }

        if continuation_token is not None:
            params["continuation-token"] = continuation_token

        for attempt in range(3):
            try:
                resp = _request_with_retry(
                    _BINANCE_S3_BASE,
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                break
            except (
                requests.RequestException,
                ET.ParseError,
            ) as exc:
                logger.warning(
                    "Binance S3 listing failed "
                    "(attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                "Binance S3 listing failed after 3 attempts: "
                f"{prefix}"
            )

        for element in root.findall(
            "s3:CommonPrefixes/s3:Prefix",
            namespace,
        ):
            if element.text:
                common_prefixes.append(element.text)

        is_truncated = root.findtext(
            "s3:IsTruncated",
            default="false",
            namespaces=namespace,
        )

        if is_truncated.lower() != "true":
            break

        continuation_token = root.findtext(
            "s3:NextContinuationToken",
            default=None,
            namespaces=namespace,
        )

        if not continuation_token:
            raise RuntimeError(
                "Binance S3 response is truncated "
                "but has no continuation token"
            )

        time.sleep(_REQUEST_DELAY)

    return common_prefixes


def _s3_list_object_keys(prefix: str) -> list[str]:
    """List actual object keys under a Binance Public Data S3 prefix."""
    namespace = {
        "s3": "http://s3.amazonaws.com/doc/2006-03-01/",
    }

    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
        }

        if continuation_token is not None:
            params["continuation-token"] = continuation_token

        for attempt in range(3):
            try:
                resp = _request_with_retry(
                    _BINANCE_S3_BASE,
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                break
            except (
                requests.RequestException,
                ET.ParseError,
            ) as exc:
                logger.warning(
                    "Binance S3 object listing failed "
                    "(attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                "Binance S3 object listing failed after 3 attempts: "
                f"{prefix}"
            )

        for element in root.findall(
            "s3:Contents/s3:Key",
            namespace,
        ):
            if element.text:
                keys.append(element.text)

        is_truncated = root.findtext(
            "s3:IsTruncated",
            default="false",
            namespaces=namespace,
        )

        if is_truncated.lower() != "true":
            break

        continuation_token = root.findtext(
            "s3:NextContinuationToken",
            default=None,
            namespaces=namespace,
        )

        if not continuation_token:
            raise RuntimeError(
                "Binance S3 response is truncated "
                "but has no continuation token"
            )

        time.sleep(_REQUEST_DELAY)

    return keys


def _symbol_is_allowed(symbol: str, config: dict[str, Any]) -> bool:
    quote = config.get("quote_asset", "USDT")
    if not symbol.endswith(quote):
        return False

    base = symbol[:-len(quote)]
    universe_cfg = config.get("universe", {})
    exclude_symbols = set(universe_cfg.get("exclude_symbols", []))
    exclude_bases = set(universe_cfg.get("exclude_bases", []))
    exclude_cats = set(universe_cfg.get("exclude_categories", []))

    if symbol in exclude_symbols:
        return False
    if base in exclude_bases:
        return False
    if "stablecoins" in exclude_cats and base in _STABLECOIN_BASES:
        return False
    if "leveraged" in exclude_cats and any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
        return False
    if "wrapped" in exclude_cats and base.startswith("W") and base[1:] in ("BTC", "ETH", "BNB"):
        return False
    return True


def fetch_archive_usdt_symbols(
    config: dict[str, Any],
) -> list[str]:
    """Discover historical USDT symbols from Binance archive."""
    prefix = "data/spot/monthly/klines/"

    archive_prefixes = _s3_list_common_prefixes(prefix)

    candidates: set[str] = set()

    for archive_prefix in archive_prefixes:
        symbol = archive_prefix.rstrip("/").split("/")[-1]

        if re.fullmatch(r"[A-Z0-9]+", symbol):
            candidates.add(symbol)

    symbols = sorted(
        symbol
        for symbol in candidates
        if _symbol_is_allowed(symbol, config)
    )

    if not symbols:
        raise RuntimeError(
            "No historical Binance symbols discovered "
            "from S3 archive"
        )

    logger.info(
        "Binance archive universe: "
        "%d filtered historical symbols",
        len(symbols),
    )

    return symbols


def _load_or_discover_archive_universe(
    cache_path: Path,
    config: dict[str, Any],
) -> list[str]:
    refresh = bool(config.get("refresh_universe", False))
    quote = config.get("quote_asset", "USDT")

    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("source") == "binance_public_data_archive"
            and payload.get("quote_asset") == quote
            and isinstance(payload.get("symbols"), list)
        ):
            symbols = payload["symbols"]
            logger.info("Historical archive universe [cache]: %d symbols", len(symbols))
            return symbols

    symbols = fetch_archive_usdt_symbols(config)
    payload = {
        "source": "binance_public_data_archive",
        "quote_asset": quote,
        "symbols": symbols,
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return symbols

# =====================================================================
# Kline download
# =====================================================================

def fetch_daily_klines(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    start_ms = _ts_ms(start_date)
    end_ms = _ts_ms(end_date) if end_date else int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": _KLINE_LIMIT,
        }

        data = _binance_get("/api/v3/klines", params)
        if not data:
            break

        all_rows.extend(data)
        last_close_time = data[-1][6]
        current_start = last_close_time + 1

        if len(data) < _KLINE_LIMIT:
            break
        time.sleep(_REQUEST_DELAY)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore",
    ])

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")

    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


# =====================================================================
# Archived monthly 1d klines
# =====================================================================

_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "base_volume",
    "close_time", "quote_volume", "trades", "taker_buy_vol",
    "taker_buy_quote_vol", "ignore",
]


def _archive_month_files(
    symbol: str,
    start_date: str,
    end_date: str | None,
) -> list[str]:
    """List existing monthly 1d ZIP archives in the requested range."""
    prefix = (
        f"data/spot/monthly/klines/"
        f"{symbol}/1d/"
    )

    keys = _s3_list_object_keys(prefix)

    pattern = re.compile(
        rf"^{re.escape(prefix)}"
        rf"{re.escape(symbol)}-1d-"
        rf"(\d{{4}})-(\d{{2}})\.zip$"
    )

    requested_start = pd.Timestamp(
        start_date
    ).to_period("M")

    requested_end = (
        pd.Timestamp(end_date)
        if end_date
        else pd.Timestamp.now().normalize()
    ).to_period("M")

    files: list[tuple[pd.Period, str]] = []

    for key in keys:
        match = pattern.fullmatch(key)

        if match is None:
            continue

        period = pd.Period(
            f"{match.group(1)}-{match.group(2)}",
            freq="M",
        )

        if requested_start <= period <= requested_end:
            filename = key.rsplit("/", 1)[-1]
            files.append((period, filename))

    return [
        filename
        for _, filename in sorted(files)
    ]


def _download_archive_zip(symbol: str, filename: str) -> pd.DataFrame:
    url = (
        f"{_BINANCE_DATA_BASE}/data/spot/monthly/klines/"
        f"{symbol}/1d/{filename}"
    )

    for attempt in range(3):
        try:
            resp = _request_with_retry(url, timeout=60)
            if resp.status_code == 404:
                return pd.DataFrame()
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    raise RuntimeError(f"No CSV inside archive: {filename}")
                with zf.open(csv_names[0]) as fh:
                    df = pd.read_csv(fh, header=None, names=_KLINE_COLUMNS)
            return df
        except (requests.RequestException, zipfile.BadZipFile, ValueError, RuntimeError) as e:
            logger.warning(
                "Archive download failed %s (attempt %d): %s",
                filename,
                attempt + 1,
                e,
            )
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Archive download failed after 3 attempts: {filename}")


def _normalize_archive_klines(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    open_time = pd.to_numeric(raw["open_time"], errors="coerce")
    valid_open_time = open_time.dropna()
    if valid_open_time.empty:
        return pd.DataFrame()

    # Binance Public Data switched Spot timestamps to microseconds from 2025-01-01.
    unit = "us" if valid_open_time.median() >= 1e14 else "ms"

    df = pd.DataFrame()
    df["date"] = (
        pd.to_datetime(open_time, unit=unit, utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    df["symbol"] = symbol
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(raw[col], errors="coerce")
    df["volume"] = pd.to_numeric(raw["quote_volume"], errors="coerce")

    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def fetch_archived_daily_klines(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download historical daily klines from Binance Public Data archives."""
    filenames = _archive_month_files(symbol, start_date, end_date)
    if not filenames:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for filename in filenames:
        raw = _download_archive_zip(symbol, filename)
        if not raw.empty:
            normalized = _normalize_archive_klines(raw, symbol)
            if not normalized.empty:
                parts.append(normalized)
        time.sleep(_REQUEST_DELAY)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


# =====================================================================
# Range-aware cache (P0.3)
# =====================================================================

def _load_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            return df
    return None


def _merge_and_save(cached: pd.DataFrame | None, new: pd.DataFrame, path: Path) -> pd.DataFrame:
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
            head = fetch_fn(start_date=start_date,
                           end_date=(cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            if not head.empty:
                new_parts.append(head)
        if need_tail:
            tail = fetch_fn(start_date=(cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                           end_date=end_date)
            if not tail.empty:
                new_parts.append(tail)

        if new_parts:
            return _merge_and_save(cached, pd.concat(new_parts, ignore_index=True), cache_path)
        return cached

    logger.info("%s [fetch full] %s -> %s", label, start_date, end_date)
    df = fetch_fn(start_date=start_date, end_date=end_date)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def _segment_symbol_episodes(
    long_df: pd.DataFrame,
    max_gap_days: int = 7,
) -> pd.DataFrame:
    """Split reused/relisted symbols into separate trading episodes."""
    df = long_df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    gaps = (
        df
        .groupby("symbol")["date"]
        .diff()
        .dt.days
    )

    starts_new_episode = gaps.gt(max_gap_days).fillna(False)

    df["_episode"] = (
        starts_new_episode
        .groupby(df["symbol"])
        .cumsum()
        .astype(int)
        + 1
    )

    episode_counts = (
        df
        .groupby("symbol")["_episode"]
        .max()
    )

    split_symbols = episode_counts[
        episode_counts > 1
    ]

    for symbol, n_episodes in split_symbols.items():
        logger.warning(
            "Symbol %s split into %d trading episodes",
            symbol,
            n_episodes,
        )

    multi_episode_mask = df["symbol"].isin(
        split_symbols.index
    )

    df.loc[multi_episode_mask, "symbol"] = (
        df.loc[multi_episode_mask, "symbol"]
        + "__seg"
        + df.loc[
            multi_episode_mask,
            "_episode",
        ].astype(str)
    )

    return df.drop(columns="_episode")


# =====================================================================
# Panel builders
# =====================================================================

def _build_wide_panel(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        long_df
        .pivot_table(index="date", columns="symbol", values=value_col, aggfunc="last")
        .sort_index()
    )


def _filter_by_volume(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    presence: pd.DataFrame,
    top_n: int = 50,
    min_volume: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # P1.6: Lag by 1 day
    median_vol = volume.rolling(30, min_periods=7).median().shift(1)

    if min_volume > 0:
        presence = presence & (median_vol.fillna(0) >= min_volume).astype(int)
        presence = presence.astype(int)

    if top_n and top_n < close.shape[1]:
        ranks = median_vol.rank(axis=1, ascending=False, method="first")
        presence = presence & (ranks <= top_n).astype(int)
        presence = presence.astype(int)

    return close, volume, presence


# =====================================================================
# BinanceLoader v3
# =====================================================================

class BinanceLoader(BaseLoader):

    def fetch(self) -> None:
        start_date = self.config.get("start_date", "2018-01-01")
        end_date = self.config.get("end_date", None)
        smoke_symbols: list[str] | None = self.config.get("smoke_symbols", None)

        cache_dir = self.data_dir / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if smoke_symbols:
            symbols = smoke_symbols
            logger.info("Smoke mode: %d symbols", len(symbols))
        else:
            universe_cache = cache_dir / "_historical_universe.json"
            symbols = _load_or_discover_archive_universe(universe_cache, self.config)

        # P1.7: Per-symbol failure isolation
        all_rows: list[pd.DataFrame] = []
        failed: list[str] = []

        for i, sym in enumerate(symbols):
            cache_path = cache_dir / f"{sym}.parquet"
            label = f"[{i + 1:3d}/{len(symbols)}] {sym}"

            try:
                df = _fetch_with_range_aware_cache(
                    fetch_fn=lambda start_date, end_date, sy=sym: fetch_archived_daily_klines(
                        sy,
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
                logger.exception("Failed to fetch %s", sym)
                failed.append(sym)

        if failed:
            logger.warning("Fetch completed with %d failures: %s", len(failed), failed)
            (self.data_dir / "fetch_failures.json").write_text(json.dumps(failed))

        if not all_rows:
            raise RuntimeError("No crypto data fetched")

        long_df = pd.concat(all_rows, ignore_index=True)
        long_df.to_parquet(self.data_dir / "all_symbols_long.parquet", index=False)

        logger.info("Crypto fetch complete: %d symbols, %d failed", len(symbols), len(failed))

    def load(self) -> MarketData:
        long_path = self.data_dir / "all_symbols_long.parquet"
        if not long_path.exists():
            raise FileNotFoundError(f"Run fetch() first. Missing: {long_path}")

        long_df = pd.read_parquet(long_path)
        long_df["date"] = pd.to_datetime(long_df["date"])

        universe_cfg = self.config.get("universe", {})

        segment_gap_days = int(universe_cfg.get("segment_gap_days", 7))

        long_df = _segment_symbol_episodes(
            long_df,
            max_gap_days=segment_gap_days,
        )

        close = _build_wide_panel(long_df, "close")
        close.index = pd.to_datetime(close.index)

        volume = _build_wide_panel(long_df, "volume")
        volume.index = pd.to_datetime(volume.index)

        presence = close.notna().astype(int)

        top_n = universe_cfg.get("top_n", 50)
        min_vol = universe_cfg.get("min_daily_volume_usd", 0)
        close, volume, presence = _filter_by_volume(close, volume, presence, top_n, min_vol)

        returns = close.pct_change(fill_method=None)

        # P1.9: Market proxy reindexed
        market_proxy_sym = self.config.get("market_proxy", "BTCUSDT")
        if market_proxy_sym in close.columns:
            market_proxy = close[market_proxy_sym].pct_change(fill_method=None).reindex(close.index)
        else:
            market_proxy = pd.Series(dtype=float, index=close.index)
        market_proxy.name = "market_proxy"

        close.to_parquet(self._cache_path("close"))
        returns.to_parquet(self._cache_path("returns"))
        volume.to_parquet(self._cache_path("volume"))
        presence.to_parquet(self._cache_path("presence_matrix"))
        market_proxy.to_frame("market_proxy").to_parquet(self._cache_path("market_proxy"))

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
                "market": "crypto",
                "source": "Binance Public Data monthly 1d kline archives",
                "quote_asset": self.config.get("quote_asset", "USDT"),
                "universe": f"Archived historical USDT pairs, top {top_n} by lagged volume",
                "volume_type": "quote_volume_usd",
                "liquidity_lag": "1 day",
                "includes_delisted": True,
                "historical_universe_source": "data.binance.vision monthly Spot kline directory",
                "caveats": self.config.get("caveats", []),
                "symbol_identity_rule": (
                    f"split after >{segment_gap_days} "
                    "calendar-day trading gap"
                ),
            },
        )


# =====================================================================
# Smoke test
# =====================================================================

def smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = {
        "market": "crypto",
        "data_dir": "data/crypto_smoke",
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
        "smoke_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        "market_proxy": "BTCUSDT",
        "universe": {"top_n": 50, "min_daily_volume_usd": 0,
                     "exclude_categories": [], "exclude_symbols": []},
    }

    loader = BinanceLoader(config)
    loader.fetch()
    md = loader.load()

    print()
    print(md.summary())
    print("Metadata:", json.dumps(md.metadata, indent=2, default=str))
    print("Close shape:", md.close.shape)
    print("Market proxy index == close index:", md.market_proxy_returns.index.equals(md.close.index))

    assert md.close.shape[1] == 5
    assert md.n_dates > 500
    assert md.market_proxy_returns.index.equals(md.close.index), "P1.9: index mismatch"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    smoke_test()