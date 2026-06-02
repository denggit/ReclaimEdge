from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch
from decimal import Decimal

if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

from scripts.run_boll_cvd_live import (  # noqa: E402
    AccountSnapshot,
    ExecutionState,
    TradeCommand,
    account_position_sync_worker,
    execution_worker,
    next_weekly_summary_time,
    strategy_tick_worker,
)
from src.execution.trader import LiveTradeResult, PositionSnapshot  # noqa: E402
from src.indicators.cvd_tracker import CvdSnapshot  # noqa: E402
from src.monitors.boll_band_breakout_monitor import BollSnapshot, MarketTickEvent, TradeTick  # noqa: E402
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig, StrategyPositionState, TradeIntent  # noqa: E402
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402


def flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


def boll() -> BollSnapshot:
    return BollSnapshot("ETH-USDT-SWAP", 0, 100.0, 100.0, 101.0, 99.0, 0.01, 0.01, True, True)


def tick(ts_ms: int) -> MarketTickEvent:
    return MarketTickEvent(TradeTick("ETH-USDT-SWAP", 100.0, 1.0, "buy", ts_ms), boll())


def cvd_snapshot(ts_ms: int) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=ts_ms,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=1.0,
        total_cvd=1.0,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_volume=1.0,
        sell_volume=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=False,
        no_new_high=False,
        window_low=100.0,
        window_high=100.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.0,
        baseline_range_pct=0.0,
        burst_move_ratio=0.0,
        burst_volume=0.0,
        baseline_volume=0.0,
        burst_volume_ratio=0.0,
        up_burst=False,
        down_burst=False,
    )


def intent(ts_ms: int, intent_type: str = "OPEN_LONG") -> TradeIntent:
    return TradeIntent(
        intent_type=intent_type,  # type: ignore[arg-type]
        side="LONG",
        price=100.0,
        layer_index=1,
        tp_price=101.0,
        reason="test",
        size=PositionSize(1.0, 50.0, 0.5, 1, 1.0),
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        boll_upper=101.0,
        boll_middle=100.0,
        boll_lower=99.0,
        ts_ms=ts_ms,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
    )


class FakeCvd:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def update(self, *, side: str, size: float, price: float, ts_ms: int) -> CvdSnapshot:
        self.calls.append(ts_ms)
        return cvd_snapshot(ts_ms)


class FakeStrategy:
    def __init__(self, intents: list[TradeIntent] | None = None, processed: asyncio.Event | None = None) -> None:
        self.state = StrategyPositionState()
        self.intents = intents or []
        self.processed_ts: list[int] = []
        self.processed = processed

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        self.processed_ts.append(ts_ms)
        if self.processed is not None:
            self.processed.set()
        return [self.intents.pop(0)] if self.intents else []


class FakeJournal:
    def __init__(self) -> None:
        self.entries: list[int] = []

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"

    def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.entries.append(kwargs["intent"].ts_ms)

    def record_tp_update(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_error(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_flat(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class FakeStateStore:
    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        pass

    def clear(self) -> None:
        pass


class FakeEmailSender:
    async def send_email_async(self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        return True


class FakeTrader:
    def __init__(self, execute_delay: float = 0.0, position_delay: float = 0.0) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.account_equity_usdt = 100.0
        self.position_contracts = Decimal("0")
        self.execute_delay = execute_delay
        self.position_delay = position_delay
        self.executed: list[int] = []

    async def execute_intent(self, trade_intent: TradeIntent) -> LiveTradeResult:
        if self.execute_delay:
            await asyncio.sleep(self.execute_delay)
        self.executed.append(trade_intent.ts_ms)
        return LiveTradeResult(True, trade_intent.intent_type, f"ord-{trade_intent.ts_ms}", "tp", "1", "101", "ok", True, True)

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        if self.position_delay:
            await asyncio.sleep(self.position_delay)
        return flat_position()

    async def fetch_usdt_equity(self) -> float:
        return 100.0

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": "100"}]}]}

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")


class GuardedLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.held = False

    async def __aenter__(self):
        await self._lock.acquire()
        self.held = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.held = False
        self._lock.release()


class LiveRuntimeWorkerTest(unittest.IsolatedAsyncioTestCase):
    def test_weekly_summary_time(self) -> None:
        real_datetime = dt.datetime

        def fixed_datetime(now_value: dt.datetime):
            class FixedDateTime(real_datetime):
                @classmethod
                def now(cls, tz=None):  # type: ignore[no-untyped-def]
                    if tz is not None:
                        return now_value.astimezone(tz)
                    return now_value

            return FixedDateTime

        cases = [
            (
                dt.datetime(2026, 6, 1, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
                dt.datetime(2026, 6, 1, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
            ),
            (
                dt.datetime(2026, 6, 1, 11, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
                dt.datetime(2026, 6, 8, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
            ),
            (
                dt.datetime(2026, 6, 2, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
                dt.datetime(2026, 6, 8, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))),
            ),
        ]
        for now_value, expected in cases:
            with patch("scripts.run_boll_cvd_live.dt.datetime", fixed_datetime(now_value)):
                self.assertEqual(next_weekly_summary_time(10, 0, weekday=0), expected)

    async def run_strategy_worker_once(self, strategy: FakeStrategy, cvd: FakeCvd, queue: asyncio.Queue[MarketTickEvent]) -> asyncio.Queue[TradeCommand]:
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        worker = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=queue,
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                execution_state=ExecutionState(None, None),
                cvd=cvd,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=999,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        await asyncio.wait_for(queue.join(), timeout=1)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        return execution_queue

    async def test_account_sync_slow_network_does_not_block_strategy_tick_worker(self) -> None:
        processed = asyncio.Event()
        strategy = FakeStrategy(processed=processed)
        cvd = FakeCvd()
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        state_lock = asyncio.Lock()
        account_snapshot = AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1)
        execution_state = ExecutionState(None, None)
        trader = FakeTrader(position_delay=1.0)
        account_task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        strategy_task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=queue,
                execution_queue=asyncio.Queue(maxsize=1000),
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                cvd=cvd,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=999,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        await asyncio.sleep(0.05)
        await queue.put(tick(1_000))

        await asyncio.wait_for(processed.wait(), timeout=0.2)

        account_task.cancel()
        strategy_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await account_task
        with contextlib.suppress(asyncio.CancelledError):
            await strategy_task

    async def test_slow_execution_worker_does_not_block_strategy_ticks(self) -> None:
        strategy = FakeStrategy(intents=[intent(1_000)])
        cvd = FakeCvd()
        strategy_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        state_lock = asyncio.Lock()
        account_snapshot = AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1)
        execution_state = ExecutionState(None, None)
        trader = FakeTrader(execute_delay=1.0)
        strategy_task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=strategy_queue,
                execution_queue=execution_queue,
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                cvd=cvd,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=999,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        execution_task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=state_lock,
                execution_state=execution_state,
                account_snapshot=account_snapshot,
                trader=trader,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
            )
        )
        await strategy_queue.put(tick(1_000))
        await asyncio.wait_for(strategy_queue.join(), timeout=0.2)
        await asyncio.sleep(0.05)
        await strategy_queue.put(tick(1_001))
        await asyncio.wait_for(strategy_queue.join(), timeout=0.2)

        self.assertEqual(cvd.calls, [1_000, 1_001])
        self.assertEqual(strategy.processed_ts, [1_000])

        strategy_task.cancel()
        execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await strategy_task
        with contextlib.suppress(asyncio.CancelledError):
            await execution_task

    async def test_strategy_tick_worker_skips_on_tick_while_execution_pending(self) -> None:
        strategy = FakeStrategy(intents=[intent(1_000)])
        cvd = FakeCvd()
        strategy_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        execution_state = ExecutionState(None, None, pending_order_count=1)
        task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=strategy_queue,
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                execution_state=execution_state,
                cvd=cvd,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=999,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        await strategy_queue.put(tick(1_000))
        await asyncio.wait_for(strategy_queue.join(), timeout=0.2)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(cvd.calls, [1_000])
        self.assertEqual(strategy.processed_ts, [])
        self.assertTrue(execution_queue.empty())

    async def test_account_sync_does_not_reset_flat_strategy_state_while_execution_pending(self) -> None:
        fetched = asyncio.Event()

        class PendingFlatTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                fetched.set()
                return flat_position()

        strategy = FakeStrategy()
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            tp_price=101.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
        )
        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=1)
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                execution_state=execution_state,
                trader=PendingFlatTrader(),  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        await asyncio.wait_for(fetched.wait(), timeout=0.2)
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(strategy.state.side, "LONG")
        self.assertEqual(strategy.state.layers, 1)
        self.assertEqual(execution_state.current_position_id, "pos-1")

    async def test_account_sync_does_not_sync_strategy_cost_while_execution_pending(self) -> None:
        fetched = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("2"), 99.0, 2.0, Decimal("2"))

        class PendingPositionTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                fetched.set()
                return live_position

        strategy = FakeStrategy()
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            tp_price=101.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
        )
        trader = PendingPositionTrader()
        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=1)
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                execution_state=execution_state,
                trader=trader,  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        await asyncio.wait_for(fetched.wait(), timeout=0.2)
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(trader.position_contracts, Decimal("2"))
        self.assertEqual(strategy.state.total_entry_qty, 1.0)
        self.assertEqual(strategy.state.avg_entry_price, 100.0)

    async def test_execution_pending_does_not_produce_second_trade_command(self) -> None:
        strategy = FakeStrategy(intents=[intent(1_000), intent(1_001)])
        cvd = FakeCvd()
        strategy_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        execution_state = ExecutionState(None, None)
        task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=strategy_queue,
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                execution_state=execution_state,
                cvd=cvd,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=999,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        await strategy_queue.put(tick(1_000))
        await asyncio.wait_for(strategy_queue.join(), timeout=0.2)
        await strategy_queue.put(tick(1_001))
        await asyncio.wait_for(strategy_queue.join(), timeout=0.2)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(cvd.calls, [1_000, 1_001])
        self.assertEqual(strategy.processed_ts, [1_000])
        self.assertEqual(execution_queue.qsize(), 1)
        self.assertEqual(execution_state.pending_order_count, 1)

    async def test_strategy_worker_processes_ticks_in_order(self) -> None:
        strategy = FakeStrategy()
        cvd = FakeCvd()
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        for ts_ms in [1_000, 1_001, 1_002]:
            await queue.put(tick(ts_ms))

        await self.run_strategy_worker_once(strategy, cvd, queue)

        self.assertEqual(cvd.calls, [1_000, 1_001, 1_002])

    async def test_execution_worker_processes_commands_in_order(self) -> None:
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        for ts_ms in [1_000, 1_001, 1_002]:
            await execution_queue.put(TradeCommand(intent(ts_ms), StrategyPositionState(), ts_ms, asyncio.get_running_loop().time(), 0, "test"))
        trader = FakeTrader()
        task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                execution_state=ExecutionState(None, None),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                trader=trader,  # type: ignore[arg-type]
                strategy=FakeStrategy(),  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
            )
        )
        await asyncio.wait_for(execution_queue.join(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(trader.executed, [1_000, 1_001, 1_002])

    async def test_execution_worker_skips_stale_add_after_partial_tp_consumed(self) -> None:
        class StaleAddTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.fetch_position_snapshot_calls = 0
                inner_self.execute_intent_called = False

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                inner_self.fetch_position_snapshot_calls += 1
                return PositionSnapshot("LONG", Decimal("5"), 100.0, 0.5, Decimal("5"))

            async def execute_intent(inner_self, trade_intent: TradeIntent) -> LiveTradeResult:
                inner_self.execute_intent_called = True
                raise AssertionError("execute_intent should not be called for stale ADD after partial TP consumed")

        class StrictJournal(FakeJournal):
            def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                raise AssertionError("record_entry should not be called for stale ADD")

            def record_tp_update(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                raise AssertionError("record_tp_update should not be called for stale ADD")

        class RecordingStateStore(FakeStateStore):
            def __init__(inner_self) -> None:
                inner_self.saved_states = []

            def save(inner_self, state) -> None:  # type: ignore[no-untyped-def]
                inner_self.saved_states.append(state)

        class StrictEmailSender(FakeEmailSender):
            async def send_email_async(inner_self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
                raise AssertionError("failure email should not be sent for skipped stale ADD")

        snapshot = StrategyPositionState(
            side="LONG",
            layers=4,
            last_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_price=108.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_consumed=False,
        )
        stale_add_state = StrategyPositionState(
            side="LONG",
            layers=5,
            last_entry_price=90.0,
            total_entry_qty=1.5,
            total_entry_notional=145.0,
            avg_entry_price=96.6667,
            partial_tp_price=106.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_consumed=False,
        )
        add_intent = TradeIntent(
            intent_type="ADD_LONG",
            side="LONG",
            price=90.0,
            layer_index=5,
            tp_price=110.0,
            reason="stale add after partial TP",
            size=PositionSize(1.0, 50.0, 0.5, 5, 1.0),
            fast_cvd=1.0,
            previous_fast_cvd=0.0,
            buy_ratio=1.0,
            sell_ratio=0.0,
            boll_upper=120.0,
            boll_middle=110.0,
            boll_lower=90.0,
            ts_ms=2_000,
            avg_entry_price=100.0,
            breakeven_price=100.1,
            tp_mode="MIDDLE",
            partial_tp_price=106.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_consumed=False,
        )
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        await execution_queue.put(
            TradeCommand(
                add_intent,
                snapshot,
                2_000,
                asyncio.get_running_loop().time(),
                0,
                "stale add after partial TP",
            )
        )
        state_lock = asyncio.Lock()
        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=1)
        account_snapshot = AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1)
        trader = StaleAddTrader()
        state_store = RecordingStateStore()
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = stale_add_state
        task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=state_lock,
                execution_state=execution_state,
                account_snapshot=account_snapshot,
                trader=trader,  # type: ignore[arg-type]
                strategy=strategy,
                journal=StrictJournal(),  # type: ignore[arg-type]
                state_store=state_store,  # type: ignore[arg-type]
                email_sender=StrictEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
            )
        )

        await asyncio.wait_for(execution_queue.join(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(trader.fetch_position_snapshot_calls, 1)
        self.assertFalse(trader.execute_intent_called)
        self.assertEqual(len(state_store.saved_states), 1)
        saved_state = state_store.saved_states[0]
        self.assertTrue(saved_state.partial_tp_consumed)
        self.assertEqual(saved_state.tp_plan, "SINGLE")
        self.assertIsNone(saved_state.partial_tp_price)
        self.assertEqual(saved_state.partial_tp_ratio, 0.0)
        self.assertEqual(saved_state.total_entry_qty, 0.5)
        self.assertEqual(saved_state.avg_entry_price, 100.0)
        self.assertEqual(execution_state.pending_order_count, 0)
        self.assertTrue(strategy.state.partial_tp_consumed)
        self.assertEqual(strategy.state.tp_plan, "SINGLE")
        self.assertIsNone(strategy.state.partial_tp_price)
        self.assertEqual(strategy.state.partial_tp_ratio, 0.0)
        self.assertEqual(strategy.state.total_entry_qty, 0.5)
        self.assertEqual(strategy.state.avg_entry_price, 100.0)

    async def test_account_snapshot_stale_warns_without_blocking_tick(self) -> None:
        processed = asyncio.Event()
        strategy = FakeStrategy(processed=processed)
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=queue,
                execution_queue=asyncio.Queue(maxsize=1000),
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, 0.0, 0, 0),
                execution_state=ExecutionState(None, None),
                cvd=FakeCvd(),  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=0.1,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        with self.assertLogs("scripts.run_boll_cvd_live", level="WARNING") as logs:
            await queue.put(tick(1_000))
            await asyncio.wait_for(processed.wait(), timeout=0.2)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertIn("ACCOUNT_SNAPSHOT_STALE", "\n".join(logs.output))

    async def test_account_sync_network_awaits_do_not_happen_inside_state_lock(self) -> None:
        state_lock = GuardedLock()

        class GuardedTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                self.assertFalse(state_lock.held)
                await asyncio.sleep(0)
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                self.assertFalse(state_lock.held)
                await asyncio.sleep(0)
                return 100.0

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                self.assertFalse(state_lock.held)
                await asyncio.sleep(0)
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": "100"}]}]}

        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=state_lock,  # type: ignore[arg-type]
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                execution_state=ExecutionState(None, None),
                trader=GuardedTrader(),  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=FakeStrategy(),  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=0,
                cash_log_min_delta_usdt=999,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_account_sync_timeout_does_not_clear_strategy_or_snapshot(self) -> None:
        fetched = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1"))

        class TimeoutTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.mark_flat_calls = 0

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                fetched.set()
                raise TimeoutError("private REST timeout")

            def mark_flat(inner_self) -> None:
                inner_self.mark_flat_calls += 1
                super().mark_flat()

        strategy = FakeStrategy()
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            tp_price=101.0,
            total_entry_qty=0.1,
            total_entry_notional=10.0,
            avg_entry_price=100.0,
        )
        account_snapshot = AccountSnapshot(live_position, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1)
        trader = TimeoutTrader()
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=account_snapshot,
                execution_state=ExecutionState("pos-1", 100.0),
                trader=trader,  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        with self.assertLogs("scripts.run_boll_cvd_live", level="WARNING"):
            await asyncio.wait_for(fetched.wait(), timeout=0.2)
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertIs(account_snapshot.position, live_position)
        self.assertEqual(account_snapshot.version, 1)
        self.assertEqual(strategy.state.side, "LONG")
        self.assertEqual(strategy.state.layers, 1)
        self.assertEqual(trader.mark_flat_calls, 0)

    async def test_account_sync_failure_logs_are_throttled(self) -> None:
        fetched = asyncio.Event()

        class FailingTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.calls = 0

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                inner_self.calls += 1
                if inner_self.calls >= 5:
                    fetched.set()
                raise TimeoutError("private REST timeout")

        with patch.dict(os.environ, {"ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS": "60", "ACCOUNT_SYNC_STALE_WARN_SECONDS": "180"}):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                    execution_state=ExecutionState(None, None),
                    trader=FailingTrader(),  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=FakeStrategy(),  # type: ignore[arg-type]
                    journal=FakeJournal(),  # type: ignore[arg-type]
                    state_store=FakeStateStore(),  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=999,
                    cash_log_min_delta_usdt=999,
                )
            )
            with self.assertLogs("scripts.run_boll_cvd_live", level="WARNING") as logs:
                await asyncio.wait_for(fetched.wait(), timeout=0.2)
                await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        output = "\n".join(logs.output)
        self.assertEqual(output.count("ACCOUNT_SYNC_FAILED"), 1)
        self.assertNotIn("Traceback", output)

    async def test_account_sync_recovery_resets_failures(self) -> None:
        fetched_after_recovery = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1"))

        class RecoveringTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.calls = 0

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                inner_self.calls += 1
                if inner_self.calls in {1, 2, 4}:
                    if inner_self.calls == 4:
                        fetched_after_recovery.set()
                    raise TimeoutError("private REST timeout")
                return live_position

        with patch.dict(os.environ, {"ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS": "0", "ACCOUNT_SYNC_STALE_WARN_SECONDS": "999"}):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                    execution_state=ExecutionState("pos-1", 100.0),
                    trader=RecoveringTrader(),  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=FakeStrategy(),  # type: ignore[arg-type]
                    journal=FakeJournal(),  # type: ignore[arg-type]
                    state_store=FakeStateStore(),  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=999,
                    cash_log_min_delta_usdt=999,
                )
            )
            with self.assertLogs("scripts.run_boll_cvd_live", level="WARNING") as logs:
                await asyncio.wait_for(fetched_after_recovery.wait(), timeout=0.2)
                await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        output = "\n".join(logs.output)
        self.assertIn("ACCOUNT_SYNC_RECOVERED | failures=2", output)
        self.assertIn("ACCOUNT_SYNC_FAILED | failures=1", output.split("ACCOUNT_SYNC_RECOVERED | failures=2", 1)[1])


if __name__ == "__main__":
    unittest.main()
