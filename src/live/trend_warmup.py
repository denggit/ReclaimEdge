"""Live startup warmup: pre-feed historical BOLL bands into TrendBreakoutAssessor.

This module fetches the most recent closed 15m candles via the exchange-agnostic
``MarketDataClientPort``, computes BOLL(20, 2.0) bands from their close prices,
constructs ``BandSnapshot`` objects, and feeds them into the strategy's trend
compression history ring buffer.  After warmup the strategy can detect trend
compression immediately, without waiting for up to 96 live 15m candles.

No trading decision is made here.  No exchange-specific types are referenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from src.data_feed.market_data_client_port import MarketDataClientPort
from src.strategies.regime.types import BandSnapshot, CompressionEpisode
from src.utils.log import get_logger

logger = get_logger(__name__)

# ── Minimum candles required before we can compute a BOLL band ──────────
_MIN_CLOSED_CANDLES_FOR_BOLL = 20


@dataclass(frozen=True)
class TrendWarmupResult:
    """Outcome of a trend compression warmup attempt."""

    success: bool
    """Whether warmup completed successfully (bands were fed)."""

    closed_candles: int
    """Number of closed candles fetched."""

    band_snapshots: int
    """Number of BOLL band snapshots computed and fed to the strategy."""

    compression_valid: bool = False
    """Whether a valid compression episode was detected after warmup."""

    compressed_candle_count: int = 0
    """Number of candles in the detected compression episode (0 if none)."""

    valid_until_ts_ms: int = 0
    """Timestamp until which the compression episode remains valid."""

    latest_band_source: str = "historical"
    """Source label for the most recent band snapshot fed."""

    reason: str = ""
    """Human-readable reason when warmup is skipped or fails."""


async def warmup_trend_compression_history(
    *,
    strategy,
    candle_client: MarketDataClientPort,
    symbol: str,
    interval: str = "15m",
    limit: int = 100,
    boll_window: int = 20,
    boll_std_multiplier: float = 2.0,
    now_ms: int,
) -> TrendWarmupResult:
    """Fetch closed candles, compute BOLL bands, and feed them as historical
    ``BandSnapshot`` objects into the strategy's trend compression history.

    Parameters
    ----------
    strategy:
        The active ``BollCvdReclaimStrategy`` instance.  Must have
        ``feed_trend_band_snapshot(band)`` and expose
        ``trend_assessor`` (the ``TrendBreakoutAssessor``).
    candle_client:
        Exchange-agnostic market data port implementing ``fetch_recent_klines``.
    symbol:
        Instrument symbol, used for log messages only.
    interval:
        K-line interval, used for log messages only.
    limit:
        Max number of candles to fetch from the REST endpoint.
    boll_window:
        BOLL rolling window size (default 20).
    boll_std_multiplier:
        Standard-deviation multiplier for BOLL bands (default 2.0).
    now_ms:
        Current unix timestamp in milliseconds.  Used to filter unclosed candles
        and to trigger compression detection.

    Returns
    -------
    TrendWarmupResult
        Structured outcome suitable for logging and diagnostics.
    """
    # ── 1. Fetch historical candles ─────────────────────────────────────
    try:
        raw_candles = await candle_client.fetch_recent_klines(limit=limit)
    except Exception as exc:
        logger.warning(
            "TREND_COMPRESSION_WARMUP_FAILED | reason=fetch_error | "
            "symbol=%s error=%s",
            symbol,
            exc,
        )
        return TrendWarmupResult(
            success=False,
            closed_candles=0,
            band_snapshots=0,
            reason=f"fetch_error: {exc}",
        )

    if not raw_candles:
        logger.warning(
            "TREND_COMPRESSION_WARMUP_SKIPPED | "
            "reason=no_candles_returned | symbol=%s",
            symbol,
        )
        return TrendWarmupResult(
            success=False,
            closed_candles=0,
            band_snapshots=0,
            reason="no_candles_returned",
        )

    # ── 2. Keep only closed candles ─────────────────────────────────────
    # A candle is closed when its close_time_ms is in the past.
    # Sort ascending by close_time_ms so oldest come first.
    sorted_candles = sorted(raw_candles, key=lambda c: c.close_time_ms)
    closed_candles = [
        c for c in sorted_candles
        if c.is_closed and c.close_time_ms <= now_ms
    ]

    if len(closed_candles) < _MIN_CLOSED_CANDLES_FOR_BOLL:
        logger.warning(
            "TREND_COMPRESSION_WARMUP_SKIPPED | "
            "reason=not_enough_closed_candles | "
            "symbol=%s closed_candles=%d required=%d",
            symbol,
            len(closed_candles),
            _MIN_CLOSED_CANDLES_FOR_BOLL,
        )
        return TrendWarmupResult(
            success=False,
            closed_candles=len(closed_candles),
            band_snapshots=0,
            reason="not_enough_closed_candles",
        )

    # ── 3. Compute rolling BOLL bands from close prices ──────────────────
    close_prices = [float(c.close_price) for c in closed_candles]

    band_snapshots: list[BandSnapshot] = []
    for i in range(boll_window - 1, len(close_prices)):
        window_closes = close_prices[i - boll_window + 1 : i + 1]
        middle = mean(window_closes)
        std = pstdev(window_closes)
        upper = middle + boll_std_multiplier * std
        lower = middle - boll_std_multiplier * std
        band_snapshots.append(
            BandSnapshot(
                upper=upper,
                middle=middle,
                lower=lower,
                candle_ts_ms=closed_candles[i].close_time_ms,
                source="historical",
            )
        )

    # ── 4. Feed each band snapshot into the strategy ────────────────────
    for band in band_snapshots:
        strategy.feed_trend_band_snapshot(band)

    # ── 5. Check compression state ──────────────────────────────────────
    assessor = getattr(strategy, "trend_assessor", None)
    compression_valid = False
    compressed_candle_count = 0
    valid_until_ts_ms = 0
    reason = "no_recent_compression"

    if assessor is not None:
        episode = assessor.detect_compression(now_ms)
        if episode is not None:
            compression_valid = True
            compressed_candle_count = episode.compressed_candle_count
            valid_until_ts_ms = episode.valid_until_ts_ms
            reason = "compression_detected"

    # ── 6. Emit summary log ─────────────────────────────────────────────
    if compression_valid:
        logger.warning(
            "TREND_COMPRESSION_WARMUP_DONE | "
            "symbol=%s interval=%s closed_candles=%d band_snapshots=%d "
            "compression_valid=true compressed_candle_count=%d "
            "valid_until_ts_ms=%d latest_band_source=historical",
            symbol,
            interval,
            len(closed_candles),
            len(band_snapshots),
            compressed_candle_count,
            valid_until_ts_ms,
        )
    else:
        logger.warning(
            "TREND_COMPRESSION_WARMUP_DONE | "
            "symbol=%s interval=%s closed_candles=%d band_snapshots=%d "
            "compression_valid=false reason=%s latest_band_source=historical",
            symbol,
            interval,
            len(closed_candles),
            len(band_snapshots),
            reason,
        )

    return TrendWarmupResult(
        success=True,
        closed_candles=len(closed_candles),
        band_snapshots=len(band_snapshots),
        compression_valid=compression_valid,
        compressed_candle_count=compressed_candle_count,
        valid_until_ts_ms=valid_until_ts_ms,
        latest_band_source="historical",
        reason=reason,
    )
