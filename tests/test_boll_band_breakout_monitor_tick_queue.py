from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
import types
import unittest

if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    Candle,
    MarketTickEvent,
    OkxPublicMarketClient,
    TradeTick,
)


def candles(count: int = 20) -> list[Candle]:
    return [
        Candle(
            ts_ms=index * 60_000,
            open=100.0 + index,
            high=100.5 + index,
            low=99.5 + index,
            close=100.0 + index,
            volume=1.0,
            confirmed=True,
        )
        for index in range(count)
    ]


class FakeCandleClient:
    def __init__(self, outcomes: list[object]):
        self.outcomes = outcomes

    async def fetch_candles(self, include_live: bool) -> list[Candle]:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


class BollBandBreakoutMonitorTickQueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_tick_event_consumer_processes_ticks_in_input_order_without_concurrency(self) -> None:
        processed: list[int] = []
        active_handlers = 0
        max_active_handlers = 0

        async def handler(event: MarketTickEvent) -> None:
            nonlocal active_handlers, max_active_handlers
            active_handlers += 1
            max_active_handlers = max(max_active_handlers, active_handlers)
            await asyncio.sleep(0)
            processed.append(event.tick.ts_ms)
            active_handlers -= 1

        monitor = BollBandBreakoutMonitor(
            BollBandBreakoutMonitorConfig(),
            tick_handlers=[handler],
        )
        monitor._running = True
        consumer = asyncio.create_task(monitor._tick_event_consumer_loop())
        ticks = [
            TradeTick("ETH-USDT-SWAP", 100.0, 1.0, "buy", 1_000),
            TradeTick("ETH-USDT-SWAP", 99.9, 1.0, "sell", 1_001),
            TradeTick("ETH-USDT-SWAP", 99.8, 1.0, "sell", 1_002),
        ]

        for tick in ticks:
            await monitor._queue_tick_event(MarketTickEvent(tick=tick, boll=None))
        await asyncio.wait_for(monitor._tick_event_queue.join(), timeout=1)

        monitor._running = False
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer

        self.assertEqual(processed, [tick.ts_ms for tick in ticks])
        self.assertEqual(max_active_handlers, 1)

    async def test_candle_sync_timeout_does_not_crash_and_logs_low_frequency(self) -> None:
        monitor = BollBandBreakoutMonitor(BollBandBreakoutMonitorConfig(candle_poll_seconds=1))
        monitor.client = FakeCandleClient([TimeoutError("timeout"), TimeoutError("timeout"), TimeoutError("timeout")])  # type: ignore[assignment]
        now_ms = 1_000_000
        monitor._now_ms = lambda: now_ms  # type: ignore[method-assign]
        monitor._candle_sync_started_ts_ms = now_ms
        monitor._candle_sync_error_log_interval_seconds = 60

        with self.assertLogs("src.monitors.boll_band_breakout_monitor", level="WARNING") as logs:
            first_sleep = await monitor._run_candle_sync_once()
            now_ms += 1_000
            second_sleep = await monitor._run_candle_sync_once()
            now_ms += 61_000
            third_sleep = await monitor._run_candle_sync_once()

        output = "\n".join(logs.output)
        self.assertEqual(monitor._candle_sync_consecutive_failures, 3)
        self.assertEqual(output.count("CANDLE_SYNC_FAILED"), 2)
        self.assertNotIn("Failed to sync candles from OKX REST", output)
        self.assertIsNotNone(logs.records[0].exc_info)
        self.assertFalse(logs.records[-1].exc_info)
        self.assertEqual(first_sleep, 6.0)
        self.assertEqual(second_sleep, 11.0)
        self.assertEqual(third_sleep, 16.0)

    async def test_candle_sync_stale_logs_error_after_warn_threshold(self) -> None:
        monitor = BollBandBreakoutMonitor(BollBandBreakoutMonitorConfig(candle_poll_seconds=1))
        monitor.client = FakeCandleClient([TimeoutError("timeout")])  # type: ignore[assignment]
        now_ms = 1_000_000
        monitor._now_ms = lambda: now_ms  # type: ignore[method-assign]
        monitor._candle_sync_started_ts_ms = now_ms - 181_000

        with self.assertLogs("src.monitors.boll_band_breakout_monitor", level="ERROR") as logs:
            await monitor._run_candle_sync_once()

        self.assertIn("CANDLE_SYNC_STALE", "\n".join(logs.output))
        self.assertIn("risk=live_boll_may_be_stale", "\n".join(logs.output))
        self.assertIsNotNone(logs.records[0].exc_info)

    async def test_candle_sync_recovery_resets_failures_and_logs_recovered(self) -> None:
        monitor = BollBandBreakoutMonitor(BollBandBreakoutMonitorConfig(candle_poll_seconds=1, boll_window=20))
        monitor.client = FakeCandleClient([TimeoutError("timeout"), candles(20)])  # type: ignore[assignment]
        now_ms = 1_000_000
        monitor._now_ms = lambda: now_ms  # type: ignore[method-assign]
        monitor._candle_sync_started_ts_ms = now_ms
        with self.assertLogs("src.monitors.boll_band_breakout_monitor", level="WARNING"):
            await monitor._run_candle_sync_once()
        now_ms += 5_000

        with self.assertLogs("src.monitors.boll_band_breakout_monitor", level="INFO") as logs:
            await monitor._run_candle_sync_once()

        self.assertEqual(monitor._candle_sync_consecutive_failures, 0)
        self.assertEqual(monitor._last_successful_candle_sync_ts_ms, now_ms)
        self.assertIn("CANDLE_SYNC_RECOVERED", "\n".join(logs.output))

    async def test_okx_public_market_client_close_closes_reused_session(self) -> None:
        client = OkxPublicMarketClient(BollBandBreakoutMonitorConfig())

        await client.start()
        session = client._session
        self.assertIsNotNone(session)
        await client.close()

        self.assertIsNone(client._session)
        self.assertTrue(session.closed)

    async def test_live_candle_from_tick_still_updates_without_rest_success(self) -> None:
        monitor = BollBandBreakoutMonitor(BollBandBreakoutMonitorConfig(bar="1m", candle_limit=100))
        await monitor._update_live_candle_from_tick(100.0, 60_000)
        await monitor._update_live_candle_from_tick(99.0, 61_000)
        await monitor._update_live_candle_from_tick(101.0, 62_000)

        self.assertEqual(len(monitor._candles), 1)
        candle = monitor._candles[0]
        self.assertEqual(candle.ts_ms, 60_000)
        self.assertEqual(candle.open, 100.0)
        self.assertEqual(candle.low, 99.0)
        self.assertEqual(candle.high, 101.0)
        self.assertEqual(candle.close, 101.0)


if __name__ == "__main__":
    unittest.main()
