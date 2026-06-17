"""Tests for trend compression warmup at live startup.

Covers:
1. 100 historical closed candles → >=80 band snapshots fed
2. Unclosed candle excluded from warmup
3. 8 narrow-band candles → compression_valid=true
4. Insufficient closed candles → TREND_COMPRESSION_WARMUP_SKIPPED
5. No exchange-specific names in the warmup module
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.data_feed.market_data_client_port import CandleSnapshot
from src.live.trend_warmup import (
    TrendWarmupResult,
    warmup_trend_compression_history,
)
from src.strategies.regime.types import BandSnapshot, CompressionEpisode

# ======================================================================
# Helpers
# ======================================================================


def _closed_candle(
    open_time_ms: int,
    close_price: float,
    *,
    open_price: float | None = None,
    high_price: float | None = None,
    low_price: float | None = None,
) -> CandleSnapshot:
    """Build a closed CandleSnapshot with minimal but valid fields."""
    return CandleSnapshot(
        open_time_ms=open_time_ms,
        close_time_ms=open_time_ms + 900_000,  # 15m later
        open_price=Decimal(str(open_price if open_price is not None else close_price)),
        high_price=Decimal(str(high_price if high_price is not None else close_price + 1)),
        low_price=Decimal(str(low_price if low_price is not None else close_price - 1)),
        close_price=Decimal(str(close_price)),
        volume=Decimal("10.0"),
        is_closed=True,
    )


def _unclosed_candle(
    open_time_ms: int,
    close_price: float,
) -> CandleSnapshot:
    """Build an in-progress (unclosed) CandleSnapshot."""
    return CandleSnapshot(
        open_time_ms=open_time_ms,
        close_time_ms=open_time_ms + 900_000,
        open_price=Decimal(str(close_price)),
        high_price=Decimal(str(close_price + 1)),
        low_price=Decimal(str(close_price - 1)),
        close_price=Decimal(str(close_price)),
        volume=Decimal("5.0"),
        is_closed=False,
    )


_FakeCompressionEpisode = CompressionEpisode


# ======================================================================
# Fake strategy for testing warmup
# ======================================================================


class _FakeAssessor:
    """Minimal fake TrendBreakoutAssessor that records fed bands."""

    def __init__(self) -> None:
        self.bands: list[BandSnapshot] = []
        self._compression_result: CompressionEpisode | None = None

    def feed_band(self, band: BandSnapshot) -> None:
        self.bands.append(band)

    def set_compression_result(self, episode: CompressionEpisode | None) -> None:
        self._compression_result = episode

    def detect_compression(self, ts_ms: int) -> CompressionEpisode | None:
        return self._compression_result


class _FakeStrategy:
    """Fake strategy that records bands fed via feed_trend_band_snapshot."""

    def __init__(self, assessor: _FakeAssessor | None = None) -> None:
        self._assessor = assessor or _FakeAssessor()
        self.trend_assessor = self._assessor

    def feed_trend_band_snapshot(self, band: BandSnapshot) -> None:
        self._assessor.feed_band(band)


class _FakeCandleClient:
    """Fake MarketDataClientPort that returns pre-configured candles."""

    def __init__(self, candles: list[CandleSnapshot]) -> None:
        self._candles = candles

    async def fetch_recent_klines(self, *, limit: int) -> list[CandleSnapshot]:
        return self._candles[:limit]


# ======================================================================
# Test: 100 closed candles → band snapshots
# ======================================================================


class TestWarmupWithFullCandles:
    """Warmup with 100 closed 15m candles should produce >=80 band snapshots."""

    def test_100_closed_candles_produces_band_snapshots(self):
        """100 closed candles → band_snapshots >= 80."""
        candles = [
            _closed_candle(i * 900_000, 3000.0 + (i % 10) * 10)
            for i in range(100)
        ]
        client = _FakeCandleClient(candles)
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is True
        assert result.closed_candles == 100
        # BOLL window is 20 → band snapshots = 100 - 20 + 1 = 81
        assert result.band_snapshots >= 80
        assert len(assessor.bands) == result.band_snapshots

    def test_band_snapshots_have_historical_source(self):
        """All fed bands must carry source='historical'."""
        candles = [
            _closed_candle(i * 900_000, 3000.0 + i)
            for i in range(30)
        ]
        client = _FakeCandleClient(candles)
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        _run_warmup(strategy, client)

        for band in assessor.bands:
            assert band.source == "historical", (
                f"Band source must be 'historical', got {band.source!r}"
            )


# ======================================================================
# Test: Unclosed candle excluded
# ======================================================================


class TestUnclosedCandleExcluded:
    """The last (in-progress) candle must never enter warmup band history."""

    def test_unclosed_candle_excluded(self):
        """Last candle is unclosed → not included in band snapshots."""
        closed = [
            _closed_candle(i * 900_000, 3000.0 + i * 5)
            for i in range(25)
        ]
        # Append an unclosed candle chronologically after all closed ones
        unclosed = _unclosed_candle(25 * 900_000, 3150.0)
        all_candles = closed + [unclosed]

        client = _FakeCandleClient(all_candles)
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        # Only 25 closed candles → bands = 25 - 20 + 1 = 6
        assert result.closed_candles == 25
        assert result.band_snapshots == 6
        # The unclosed candle's close_time_ms should NOT appear in any band
        unclosed_ts = unclosed.close_time_ms
        for band in assessor.bands:
            assert band.candle_ts_ms != unclosed_ts, (
                "Unclosed candle timestamp must not appear in warmup bands"
            )

    def test_unclosed_candle_in_future_excluded(self):
        """Candle with close_time_ms > now_ms is excluded even if is_closed=True."""
        now_ms = 100 * 900_000
        closed = [
            _closed_candle(i * 900_000, 3000.0 + i * 5)
            for i in range(30)
        ]
        # A "closed" candle whose close_time is in the future
        future_candle = CandleSnapshot(
            open_time_ms=105 * 900_000,
            close_time_ms=105 * 900_000 + 900_000,  # > now_ms
            open_price=Decimal("3200"),
            high_price=Decimal("3210"),
            low_price=Decimal("3190"),
            close_price=Decimal("3200"),
            volume=Decimal("10.0"),
            is_closed=True,  # marked closed but close_time is future
        )
        all_candles = closed + [future_candle]

        client = _FakeCandleClient(all_candles)
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client, now_ms=now_ms)

        # The future candle should be excluded
        assert result.closed_candles == 30
        future_ts = future_candle.close_time_ms
        for band in assessor.bands:
            assert band.candle_ts_ms != future_ts, (
                "Future candle must not enter warmup bands"
            )


# ======================================================================
# Test: Compression detection after warmup
# ======================================================================


class TestCompressionDetectionAfterWarmup:
    """After feeding narrow-band candles, compression_valid should be true."""

    def test_narrow_bands_yield_compression_valid(self):
        """8 consecutive narrow BOLL bands → compression_valid=true."""
        # Generate 30 candles with very tight close prices
        candles = [
            _closed_candle(i * 900_000, 3000.0 + (i % 3) * 0.5)
            for i in range(30)
        ]
        client = _FakeCandleClient(candles)
        assessor = _FakeAssessor()

        # Set up a fake compression episode for the assessor to return
        assessor.set_compression_result(
            CompressionEpisode(
                start_ts_ms=20 * 900_000,
                end_ts_ms=29 * 900_000,
                valid_until_ts_ms=(29 * 900_000) + 7200_000,
                compressed_candle_count=8,
                min_outer_distance_pct=0.001,
                avg_outer_distance_pct=0.002,
                upper_at_end=3010.0,
                middle_at_end=3005.0,
                lower_at_end=3000.0,
                highest_band_upper=3015.0,
                lowest_band_lower=2995.0,
            )
        )
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is True
        assert result.compression_valid is True
        assert result.compressed_candle_count == 8
        assert result.valid_until_ts_ms > 0

    def test_no_compression_when_bands_are_wide(self):
        """Wide BOLL bands → compression_valid=false."""
        candles = [
            _closed_candle(i * 900_000, 3000.0 + i * 50)
            for i in range(30)
        ]
        client = _FakeCandleClient(candles)
        assessor = _FakeAssessor()
        assessor.set_compression_result(None)  # No compression
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is True
        assert result.compression_valid is False
        assert result.reason == "no_recent_compression"


# ======================================================================
# Test: Insufficient candles → skipped
# ======================================================================


class TestInsufficientCandlesSkipped:
    """When there are fewer than 20 closed candles, warmup is skipped."""

    def test_too_few_candles_skipped(self):
        """Only 10 closed candles → TREND_COMPRESSION_WARMUP_SKIPPED."""
        candles = [
            _closed_candle(i * 900_000, 3000.0 + i)
            for i in range(10)
        ]
        client = _FakeCandleClient(candles)
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is False
        assert result.closed_candles == 10
        assert result.band_snapshots == 0
        assert result.reason == "not_enough_closed_candles"
        # No bands should have been fed
        assert len(assessor.bands) == 0

    def test_exactly_20_candles_succeeds(self):
        """Exactly 20 closed candles → 1 band snapshot, success."""
        candles = [
            _closed_candle(i * 900_000, 3000.0 + i)
            for i in range(20)
        ]
        client = _FakeCandleClient(candles)
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is True
        assert result.closed_candles == 20
        assert result.band_snapshots == 1  # 20 - 20 + 1

    def test_zero_candles_skipped(self):
        """Empty candle list → skipped."""
        client = _FakeCandleClient([])
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is False
        assert result.reason == "no_candles_returned"


# ======================================================================
# Test: Fetch error → failed but does not crash
# ======================================================================


class TestFetchErrorDoesNotCrash:
    """When fetch_recent_klines raises, warmup logs failure and returns."""

    def test_fetch_error_returns_failed_result(self):
        """Exception in candle client → TrendWarmupResult(success=False)."""

        class _FailingClient:
            async def fetch_recent_klines(self, *, limit: int):
                raise RuntimeError("network timeout")

        client = _FailingClient()
        assessor = _FakeAssessor()
        strategy = _FakeStrategy(assessor)

        result = _run_warmup(strategy, client)

        assert result.success is False
        assert "fetch_error" in result.reason
        # Strategy state unchanged
        assert len(assessor.bands) == 0


# ======================================================================
# Test: No exchange-specific names
# ======================================================================


class TestNoExchangeNamesInWarmup:
    """The warmup module must not reference any exchange-specific names."""

    _FORBIDDEN = {"okx", "OKX", "binance", "Binance", "BINANCE"}

    def test_warmup_module_has_no_exchange_names(self):
        """Source scan: warmup module must not contain exchange names."""
        warmup_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "live"
            / "trend_warmup.py"
        )
        assert warmup_path.exists(), f"Warmup module not found at {warmup_path}"
        text = warmup_path.read_text(encoding="utf-8")

        for forbidden in self._FORBIDDEN:
            assert forbidden not in text, (
                f"Warmup module must not contain exchange name: {forbidden!r}"
            )


# ======================================================================
# Test: Strategy public wrapper
# ======================================================================


class TestStrategyFeedTrendBandSnapshot:
    """The strategy's feed_trend_band_snapshot must delegate to the assessor."""

    def test_feed_trend_band_snapshot_delegates_to_assessor(self):
        """Calling strategy.feed_trend_band_snapshot feeds the assessor."""
        from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
        )

        sizer = SimplePositionSizer(SimplePositionSizerConfig(
            dry_run_equity_usdt=1000.0,
            trade_risk_pct=0.003,
            leverage=20.0,
        ))
        config = BollCvdReclaimStrategyConfig(trend_breakout_enabled=True)
        strategy = BollCvdReclaimStrategy(config, sizer)

        band = BandSnapshot(
            upper=3100.0,
            middle=3000.0,
            lower=2900.0,
            candle_ts_ms=1000000,
            source="historical",
        )
        strategy.feed_trend_band_snapshot(band)

        # After feeding, the assessor should have the band
        assessor = strategy.trend_assessor
        assert assessor is not None
        assert len(assessor._band_history) == 1
        fed = assessor._band_history[0]
        assert fed.upper == 3100.0
        assert fed.middle == 3000.0
        assert fed.lower == 2900.0
        assert fed.source == "historical"

    def test_feed_trend_band_snapshot_does_not_crash_when_disabled(self):
        """When trend_breakout_enabled=False, feed_trend_band_snapshot is a no-op."""
        from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
        )

        sizer = SimplePositionSizer(SimplePositionSizerConfig(
            dry_run_equity_usdt=1000.0,
            trade_risk_pct=0.003,
            leverage=20.0,
        ))
        config = BollCvdReclaimStrategyConfig(trend_breakout_enabled=False)
        strategy = BollCvdReclaimStrategy(config, sizer)

        band = BandSnapshot(
            upper=3100.0, middle=3000.0, lower=2900.0,
            candle_ts_ms=1000000, source="historical",
        )
        # Should not crash
        strategy.feed_trend_band_snapshot(band)
        assert strategy.trend_assessor is None


# ======================================================================
# Helper: run warmup synchronously
# ======================================================================


def _run_warmup(
    strategy: _FakeStrategy,
    client,
    *,
    now_ms: int | None = None,
) -> TrendWarmupResult:
    """Run warmup synchronously by driving the async function in a real loop."""
    import asyncio

    if now_ms is None:
        # Default to well after all candles
        now_ms = 200 * 900_000

    async def _go() -> TrendWarmupResult:
        return await warmup_trend_compression_history(
            strategy=strategy,
            candle_client=client,
            symbol="ETH-USDT-SWAP",
            interval="15m",
            limit=100,
            boll_window=20,
            boll_std_multiplier=2.0,
            now_ms=now_ms,
        )

    return asyncio.run(_go())
