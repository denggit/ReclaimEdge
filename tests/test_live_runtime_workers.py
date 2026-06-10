from __future__ import annotations

import asyncio
import contextlib
import copy
import datetime as dt
import importlib.util
import logging
import os
import sys
import time
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

from src.live.workers.account_position_sync_worker import account_position_sync_worker  # noqa: E402
from src.live.workers.execution_worker import execution_worker  # noqa: E402
from src.live.workers.strategy_tick_worker import strategy_tick_worker  # noqa: E402
from src.live.startup_recovery.trust_validation import (  # noqa: E402
    trusted_startup_saved_state,
)
from src.execution.trader import LiveTradeResult, PositionSnapshot  # noqa: E402
from src.live.account_sync.flat_balance import fetch_settled_flat_balance  # noqa: E402
from src.live.queue_helpers import (  # noqa: E402
    enqueue_execution_command,
    enqueue_strategy_tick,
    queue_log_level,
    queue_oldest_command_age_seconds,
)
from src.live.runtime_types import AccountSnapshot, ExecutionState, TradeCommand  # noqa: E402
from src.live.time_utils import next_daily_report_time, next_weekly_summary_time  # noqa: E402
from src.indicators.cvd_tracker import CvdSnapshot  # noqa: E402
from src.monitors.boll_band_breakout_monitor import BollSnapshot, MarketTickEvent, TradeTick  # noqa: E402
from src.live.startup_recovery.basic_restore import (  # noqa: E402
    restore_strategy_from_position,
)
from src.position_management.runner_live_helpers import (  # noqa: E402
    apply_three_stage_startup_safety_gate,
    three_stage_post_tp1_current_price,
)
from src.position_management.tp_progress import (  # noqa: E402
    mark_middle_runner_active_if_position_reduced,
    mark_three_stage_progress_if_position_reduced,
)
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig, StrategyPositionState, \
    TradeIntent  # noqa: E402
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
        self.flats = []
        self.cash_transfers = []
        self.cash_drifts = []
        self.events = []

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"

    def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.entries.append(kwargs["intent"].ts_ms)

    def record_tp_update(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_error(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_flat(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if kwargs.get("cash_before_position") is not None and kwargs.get("cash_after") is not None:
            pnl = kwargs["cash_after"] - kwargs["cash_before_position"]
            kwargs["realized_pnl_usdt_est"] = pnl
            kwargs["realized_pnl_pct_est"] = pnl / kwargs["cash_before_position"] * 100 if kwargs[
                "cash_before_position"] else None
        self.flats.append(kwargs)

    def record_cash_transfer(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.cash_transfers.append(kwargs)

    def record_account_cash_drift(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.cash_drifts.append(kwargs)

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event_name, dict(payload), position_id))


class FakeStateStore:
    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        pass

    def clear(self) -> None:
        pass


class RecordingStateStore(FakeStateStore):
    def __init__(self) -> None:
        self.clear_calls = 0
        self.saved_states = []

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved_states.append(state)

    def clear(self) -> None:
        self.clear_calls += 1


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
        self.post_tp1_stop_orders = []
        self.cancelled_post_tp1_stop_ids = []
        self.cancel_post_tp1_ok = True
        self.middle_runner_stop_orders: list[dict] = []
        self.cancelled_middle_runner_stop_ids: list[str | None] = []
        self.cancel_middle_runner_ok = True
        self.market_exits: list[tuple] = []
        self.market_exit_ok = True
        self.market_exit_message = "ok"
        self.tp_order_id = ""
        self.pending_orders: list[dict[str, str]] = []

    async def fetch_pending_orders(self) -> list[dict[str, str]]:
        return list(self.pending_orders)

    async def execute_intent(self, trade_intent: TradeIntent) -> LiveTradeResult:
        if self.execute_delay:
            await asyncio.sleep(self.execute_delay)
        self.executed.append(trade_intent.ts_ms)
        return LiveTradeResult(True, trade_intent.intent_type, f"ord-{trade_intent.ts_ms}", "tp", "1", "101", "ok",
                               True, True)

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

    async def place_three_stage_post_tp1_protective_stop_with_retries(self, side, contracts, stop_price, retry_count,
                                                                      retry_interval_seconds):  # type: ignore[no-untyped-def]
        order_id = f"post-tp1-{len(self.post_tp1_stop_orders) + 1}"
        self.post_tp1_stop_orders.append(
            {
                "side": side,
                "contracts": contracts,
                "stop_price": stop_price,
                "retry_count": retry_count,
                "retry_interval_seconds": retry_interval_seconds,
                "order_id": order_id,
            }
        )
        return True, order_id, "protective_sl_placed"

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_post_tp1_stop_ids.append(order_id)
        return self.cancel_post_tp1_ok

    async def place_middle_runner_protective_stop_with_retries(self, side, contracts, stop_price, retry_count,
                                                               retry_interval_seconds):  # type: ignore[no-untyped-def]
        order_id = f"mid-runner-{len(self.middle_runner_stop_orders) + 1}"
        self.middle_runner_stop_orders.append(
            {
                "side": side,
                "contracts": contracts,
                "stop_price": stop_price,
                "retry_count": retry_count,
                "retry_interval_seconds": retry_interval_seconds,
                "order_id": order_id,
            }
        )
        return True, order_id, "protective_sl_placed"

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_middle_runner_stop_ids.append(order_id)
        return self.cancel_middle_runner_ok

    async def market_exit_remaining_position_with_retries(self, side, retry_count, *, context="generic", retry_interval_seconds=None):  # type: ignore[no-untyped-def]
        self.market_exits.append((side, retry_count, context, retry_interval_seconds))
        return self.market_exit_ok, self.market_exit_message


class SidecarWorkerTrader(FakeTrader):
    def __init__(self) -> None:
        super().__init__()
        self.account_equity_usdt = 100.0
        self.leverage = "50"
        self.contract_multiplier = Decimal("0.1")
        self.contract_precision = Decimal("0.01")
        self.executed_intents: list[TradeIntent] = []
        self.sidecar_tps = []
        self.market_exits = []

    def eth_qty_to_contracts(self, qty: Decimal) -> Decimal:
        if qty <= 0:
            return Decimal("0")
        return (qty / self.contract_multiplier).quantize(self.contract_precision)

    async def execute_intent(self, trade_intent: TradeIntent) -> LiveTradeResult:
        self.executed_intents.append(trade_intent)
        self.executed.append(trade_intent.ts_ms)
        contracts = str(self.eth_qty_to_contracts(Decimal(str(trade_intent.size.eth_qty))))
        return LiveTradeResult(True, trade_intent.intent_type, f"ord-{trade_intent.ts_ms}", "tp", contracts, "101",
                               "ok", True, True)

    async def place_sidecar_fixed_take_profit(self, *, side, contracts, tp_price,
                                              client_order_id=None):  # type: ignore[no-untyped-def]
        self.sidecar_tps.append((side, contracts, tp_price, client_order_id))
        return "sidecar-tp"

    async def market_exit_remaining_position_with_retries(self, side, retry_count, *, context="generic", retry_interval_seconds=None):  # type: ignore[no-untyped-def]
        self.market_exits.append((side, retry_count, context, retry_interval_seconds))
        return True, "ok"


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


class RaceFullQueue(asyncio.Queue[TradeCommand]):
    def full(self) -> bool:
        return False

    def put_nowait(self, item: TradeCommand) -> None:
        raise asyncio.QueueFull


class LiveRuntimeWorkerTest(unittest.IsolatedAsyncioTestCase):
    def test_queue_log_level_thresholds(self) -> None:
        self.assertIsNone(queue_log_level(0))
        self.assertIsNone(queue_log_level(499))
        self.assertEqual(queue_log_level(500), logging.INFO)
        self.assertEqual(queue_log_level(1999), logging.INFO)
        self.assertEqual(queue_log_level(2000), logging.WARNING)
        self.assertEqual(queue_log_level(7999), logging.WARNING)
        self.assertEqual(queue_log_level(8000), logging.ERROR)

    async def test_queue_oldest_command_age_seconds_returns_zero_on_empty_queue(self) -> None:
        queue: asyncio.Queue[TradeCommand] = asyncio.Queue()

        self.assertEqual(queue_oldest_command_age_seconds(queue), 0.0)

    async def test_queue_oldest_command_age_seconds_uses_oldest_created_monotonic(self) -> None:
        queue: asyncio.Queue[TradeCommand] = asyncio.Queue()
        await queue.put(TradeCommand(intent(1_000), StrategyPositionState(), 1_000, time.monotonic() - 10.0, 0, "old"))
        await queue.put(TradeCommand(intent(1_001), StrategyPositionState(), 1_001, time.monotonic(), 0, "new"))

        self.assertGreaterEqual(queue_oldest_command_age_seconds(queue), 9.0)

    async def test_enqueue_strategy_tick_skips_event_without_boll(self) -> None:
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        execution_state = ExecutionState(None, None)

        await enqueue_strategy_tick(
            MarketTickEvent(TradeTick("ETH-USDT-SWAP", 100.0, 1.0, "buy", 1_000), None),
            queue,
            asyncio.Lock(),
            execution_state,
        )

        self.assertTrue(queue.empty())
        self.assertFalse(execution_state.trading_halted)

    async def test_enqueue_strategy_tick_halts_when_queue_full(self) -> None:
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(maxsize=1)
        await queue.put(tick(1_000))
        execution_state = ExecutionState(None, None)

        await enqueue_strategy_tick(tick(1_001), queue, asyncio.Lock(), execution_state)

        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(queue.qsize(), 1)

    async def test_enqueue_execution_command_success_increments_pending_order_count(self) -> None:
        queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1)
        execution_state = ExecutionState(None, None)
        command = TradeCommand(intent(1_000), StrategyPositionState(), 1_000, time.monotonic(), 0, "test")

        ok = await enqueue_execution_command(command, queue, asyncio.Lock(), execution_state)

        self.assertTrue(ok)
        self.assertEqual(execution_state.pending_order_count, 1)
        self.assertIs(queue.get_nowait(), command)

    async def test_enqueue_execution_command_full_queue_halts_without_incrementing_pending_order_count(self) -> None:
        queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1)
        await queue.put(TradeCommand(intent(999), StrategyPositionState(), 999, time.monotonic(), 0, "existing"))
        execution_state = ExecutionState(None, None)
        command = TradeCommand(intent(1_000), StrategyPositionState(), 1_000, time.monotonic(), 0, "test")

        ok = await enqueue_execution_command(command, queue, asyncio.Lock(), execution_state)

        self.assertFalse(ok)
        self.assertEqual(execution_state.pending_order_count, 0)
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(queue.qsize(), 1)

    async def test_enqueue_execution_command_put_nowait_race_rolls_back_pending_order_count(self) -> None:
        queue = RaceFullQueue(maxsize=1)
        execution_state = ExecutionState(None, None)
        command = TradeCommand(intent(1_000), StrategyPositionState(), 1_000, time.monotonic(), 0, "test")

        ok = await enqueue_execution_command(command, queue, asyncio.Lock(), execution_state)

        self.assertFalse(ok)
        self.assertEqual(execution_state.pending_order_count, 0)
        self.assertTrue(execution_state.trading_halted)

    def three_stage_strategy(self, side: str = "LONG") -> BollCvdShockReclaimStrategy:
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(
                three_stage_runner_enabled=True,
                three_stage_post_tp1_sl_extension_trigger_ratio=0.6,
            ),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side=side,  # type: ignore[arg-type]
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0 if side == "LONG" else 90.0,
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=101.0 if side == "LONG" else 99.0,
            three_stage_tp2_price=110.0 if side == "LONG" else 90.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
        )
        return strategy

    async def test_middle_runner_activation_logs_cost_basis_after_first_close(self) -> None:
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(middle_runner_enabled=True, breakeven_fee_buffer_pct=0.001),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=105.0,
            partial_tp_ratio=0.8,
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=105.0,
            middle_runner_final_tp_price=110.0,
            position_cost_entry_notional=100.0,
            position_cost_remaining_qty=1.0,
        )
        position = PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        with self.assertLogs("src.position_management.tp_progress", level="WARNING") as logs:
            activated = mark_middle_runner_active_if_position_reduced(strategy, position)

        self.assertTrue(activated)
        joined = "\n".join(logs.output)
        self.assertIn("MIDDLE_RUNNER_COST_BASIS_AFTER_FIRST_CLOSE", joined)
        self.assertIn("net_remaining_breakeven_price", joined)
        self.assertIn("position_cost_exit_notional=84.0000", joined)

    async def test_three_stage_tp1_logs_cost_basis_after_tp1(self) -> None:
        strategy = self.three_stage_strategy("LONG")
        strategy.state.position_cost_entry_notional = 100.0
        strategy.state.position_cost_remaining_qty = 1.0
        position = PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        with self.assertLogs("src.position_management.tp_progress", level="WARNING") as logs:
            event = mark_three_stage_progress_if_position_reduced(strategy, position, 10_000)

        self.assertEqual(event, "TP1")
        joined = "\n".join(logs.output)
        self.assertIn("THREE_STAGE_COST_BASIS_AFTER_TP1", joined)
        self.assertIn("net_remaining_breakeven_price", joined)
        self.assertIn("position_cost_exit_notional=60.6000", joined)

    async def run_account_sync_until(self, predicate, *, account_snapshot, execution_state, trader, strategy, journal,
                                     state_store, timeout: float = 0.5):  # type: ignore[no-untyped-def]
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,
                journal=journal,
                state_store=state_store,
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        try:
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.01)
            self.fail("account sync predicate was not satisfied")
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_strategy_tick_worker_updates_latest_market_price(self) -> None:
        processed = asyncio.Event()
        strategy = FakeStrategy(processed=processed)
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        account_snapshot = AccountSnapshot(flat_position(), 100.0, 100.0, 0.0, 0, 0)
        task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=queue,
                execution_queue=asyncio.Queue(maxsize=1000),
                state_lock=asyncio.Lock(),
                account_snapshot=account_snapshot,
                execution_state=ExecutionState(None, None),
                cvd=FakeCvd(),  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=1_000_000_000_000,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        await queue.put(MarketTickEvent(TradeTick("ETH-USDT-SWAP", 106.4, 1.0, "buy", 12_345), boll()))
        await asyncio.wait_for(processed.wait(), timeout=0.2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(account_snapshot.latest_market_price, 106.4)
        self.assertEqual(account_snapshot.latest_market_price_ts_ms, 12_345)

    async def test_three_stage_tp1_sync_places_long_post_tp1_sl_with_extension(self) -> None:
        class Tp1Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        strategy = self.three_stage_strategy("LONG")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
                                           latest_market_price=106.4, latest_market_price_ts_ms=latest_ts_ms)
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: len(trader.post_tp1_stop_orders) == 1,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertGreaterEqual(trader.post_tp1_stop_orders[0]["stop_price"], 101.0)
        self.assertTrue(strategy.state.three_stage_post_tp1_sl_extension_triggered)
        self.assertEqual(strategy.state.three_stage_post_tp1_protective_sl_order_id, "post-tp1-1")
        self.assertIn("THREE_STAGE_TP1_PROTECTIVE_SL_PLACED", [event[0] for event in journal.events])

    async def test_three_stage_tp1_sync_places_short_post_tp1_sl_with_extension(self) -> None:
        class Tp1Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("SHORT", Decimal("4"), 100.0, 0.4, Decimal("-4"))

        strategy = self.three_stage_strategy("SHORT")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
                                           latest_market_price=93.6, latest_market_price_ts_ms=latest_ts_ms)
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: len(trader.post_tp1_stop_orders) == 1,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertLessEqual(trader.post_tp1_stop_orders[0]["stop_price"], 99.0)
        self.assertTrue(strategy.state.three_stage_post_tp1_sl_extension_triggered)
        self.assertEqual(strategy.state.three_stage_post_tp1_protective_sl_order_id, "post-tp1-1")

    async def test_three_stage_tp1_sync_falls_back_when_latest_market_price_missing(self) -> None:
        class Tp1Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        strategy = self.three_stage_strategy("LONG")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        account_snapshot = AccountSnapshot(None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1)
        execution_state = ExecutionState("pos-1", 100.0)

        with self.assertLogs("src.position_management.runner_live_helpers", level="WARNING") as logs:
            await self.run_account_sync_until(
                lambda: len(trader.post_tp1_stop_orders) == 1,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,
                strategy=strategy,
                journal=journal,
                state_store=state_store,
            )

        self.assertIn("THREE_STAGE_POST_TP1_SL_PRICE_FALLBACK", "\n".join(logs.output))
        placed_payloads = [payload for event_name, payload, _position_id in journal.events if
                           event_name == "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED"]
        self.assertEqual(placed_payloads[0]["reason"], "three_stage_tp1_filled")

    async def test_three_stage_tp2_cancel_post_tp1_sl_success_clears_state(self) -> None:
        class Tp2Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        strategy = self.three_stage_strategy("LONG")
        strategy.state.partial_tp_consumed = True
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True
        trader = Tp2Trader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        account_snapshot = AccountSnapshot(None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
                                           latest_market_price=110.0)
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: any(event[0] == "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2" for event in journal.events),
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertEqual(trader.cancelled_post_tp1_stop_ids, ["old-post"])
        self.assertIsNone(strategy.state.three_stage_post_tp1_protective_sl_order_id)
        self.assertIsNone(strategy.state.three_stage_post_tp1_protective_sl_price)
        self.assertFalse(strategy.state.three_stage_post_tp1_protected)
        self.assertFalse(execution_state.trading_halted)

    async def test_three_stage_tp2_cancel_post_tp1_sl_failure_preserves_state_and_halts(self) -> None:
        class Tp2Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        strategy = self.three_stage_strategy("LONG")
        strategy.state.partial_tp_consumed = True
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True
        trader = Tp2Trader()
        trader.cancel_post_tp1_ok = False
        journal = FakeJournal()
        state_store = RecordingStateStore()
        account_snapshot = AccountSnapshot(None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
                                           latest_market_price=110.0)
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: execution_state.trading_halted,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertEqual(trader.cancelled_post_tp1_stop_ids, ["old-post"])
        self.assertEqual(strategy.state.three_stage_post_tp1_protective_sl_order_id, "old-post")
        self.assertEqual(strategy.state.three_stage_post_tp1_protective_sl_price, 101.0)
        self.assertTrue(strategy.state.three_stage_post_tp1_protected)
        self.assertEqual(execution_state.halt_reason, "three_stage_post_tp1_sl_cancel_failed_on_tp2")
        self.assertIn("THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2", [event[0] for event in journal.events])
        self.assertEqual(state_store.saved_states[-1].three_stage_post_tp1_protective_sl_order_id, "old-post")
        self.assertEqual(trader.post_tp1_stop_orders, [])

        blocked_strategy = FakeStrategy(processed=asyncio.Event())
        queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue()
        tick_task = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=queue,
                execution_queue=asyncio.Queue(maxsize=1000),
                state_lock=asyncio.Lock(),
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                cvd=FakeCvd(),  # type: ignore[arg-type]
                strategy=blocked_strategy,  # type: ignore[arg-type]
                heartbeat_seconds=1_000_000_000_000,
                account_stale_warn_seconds=1_000_000_000_000,
                strategy_lag_warn_seconds=1_000_000_000_000,
            )
        )
        await queue.put(MarketTickEvent(TradeTick("ETH-USDT-SWAP", 111.0, 1.0, "buy", 99_999), boll()))
        await asyncio.wait_for(queue.join(), timeout=0.2)
        tick_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick_task
        self.assertEqual(blocked_strategy.processed_ts, [])

    async def test_startup_restores_dirty_post_tp1_sl_after_tp2_and_halts(self) -> None:
        strategy = self.three_stage_strategy("LONG")
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_tp2_consumed = True
        strategy.state.trend_runner_active = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        state_store = RecordingStateStore()

        applied = apply_three_stage_startup_safety_gate(
            strategy=strategy,
            execution_state=execution_state,
            saved_state=types.SimpleNamespace(position_id="pos-1"),
            startup_position=PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2")),
            journal=journal,  # type: ignore[arg-type]
            state_store=state_store,  # type: ignore[arg-type]
            trader_symbol="ETH-USDT-SWAP",
        )

        self.assertTrue(applied)
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(execution_state.halt_reason, "three_stage_post_tp1_sl_cancel_failed_on_tp2_restart")
        self.assertEqual(strategy.state.three_stage_post_tp1_protective_sl_order_id, "old-post")
        self.assertEqual(strategy.state.three_stage_post_tp1_protective_sl_price, 101.0)
        self.assertTrue(strategy.state.three_stage_post_tp1_protected)
        self.assertIn("THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2_RESTORED",
                      [event[0] for event in journal.events])
        self.assertEqual(state_store.saved_states[-1].three_stage_post_tp1_protective_sl_order_id, "old-post")

    async def test_dirty_post_tp1_sl_blocks_runner_update(self) -> None:
        strategy = self.three_stage_strategy("LONG")
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_tp2_consumed = True
        strategy.state.trend_runner_active = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True
        runner_update = intent(123_456, intent_type="UPDATE_TP")
        runner_update = TradeIntent(
            **{
                **runner_update.__dict__,
                "tp_plan": "THREE_STAGE_RUNNER",
                "trend_runner_active": True,
                "three_stage_tp2_consumed": True,
                "trend_runner_tp_price": 111.0,
                "trend_runner_sl_price": 101.0,
            }
        )
        queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        await queue.put(TradeCommand(runner_update, copy.deepcopy(strategy.state), runner_update.ts_ms,
                                     asyncio.get_running_loop().time(), 0, runner_update.reason))
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        task = asyncio.create_task(
            execution_worker(
                execution_queue=queue,
                state_lock=asyncio.Lock(),
                execution_state=execution_state,
                account_snapshot=AccountSnapshot(PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2")),
                                                 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
                trader=trader,  # type: ignore[arg-type]
                strategy=strategy,
                journal=journal,  # type: ignore[arg-type]
                state_store=state_store,  # type: ignore[arg-type]
                email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
            )
        )
        await asyncio.wait_for(queue.join(), timeout=0.2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(trader.executed, [])
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(execution_state.halt_reason, "three_stage_post_tp1_sl_dirty_state_blocked")
        self.assertIn("THREE_STAGE_DIRTY_POST_TP1_SL_BLOCKED_RUNNER_UPDATE", [event[0] for event in journal.events])
        self.assertEqual(state_store.saved_states[-1].three_stage_post_tp1_protective_sl_order_id, "old-post")

    async def test_tp2_cancel_pending_does_not_overwrite_existing_critical_halt(self) -> None:
        class Tp2Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        strategy = self.three_stage_strategy("LONG")
        strategy.state.partial_tp_consumed = True
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True
        trader = Tp2Trader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        account_snapshot = AccountSnapshot(None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
                                           latest_market_price=110.0)
        execution_state = ExecutionState("pos-1", 100.0, trading_halted=True, halt_reason="some_existing_critical_halt")

        await self.run_account_sync_until(
            lambda: any(event[0] == "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2" for event in journal.events),
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertEqual(trader.cancelled_post_tp1_stop_ids, ["old-post"])
        self.assertIsNone(strategy.state.three_stage_post_tp1_protective_sl_order_id)
        self.assertIsNone(strategy.state.three_stage_post_tp1_protective_sl_price)
        self.assertFalse(strategy.state.three_stage_post_tp1_protected)
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(execution_state.halt_reason, "some_existing_critical_halt")

    def test_latest_market_price_stale_falls_back(self) -> None:
        account_snapshot = AccountSnapshot(
            None,
            100.0,
            100.0,
            0.0,
            0,
            1,
            latest_market_price=106.4,
            latest_market_price_ts_ms=10_000,
        )
        position = PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        with patch.dict(os.environ, {"LATEST_MARKET_PRICE_MAX_AGE_SECONDS": "30"}):
            with self.assertLogs("src.position_management.runner_live_helpers", level="WARNING") as logs:
                current_price, source = three_stage_post_tp1_current_price(account_snapshot, position, boll(),
                                                                           now_ms=100_000)

        self.assertEqual(current_price, 100.0)
        self.assertEqual(source, "position_avg_entry")
        self.assertIn("latest_market_price_stale", "\n".join(logs.output))

    def test_restore_strategy_from_position_sets_conservative_first_entry_clock(self) -> None:
        strategy = BollCvdShockReclaimStrategy(BollCvdReclaimStrategyConfig(),
                                               SimplePositionSizer(SimplePositionSizerConfig()))
        position = PositionSnapshot("LONG", Decimal("3"), 100.5, 0.3, Decimal("3"))

        restore_strategy_from_position(strategy, position, now_ms=123_456)

        self.assertEqual(strategy.state.first_entry_ts_ms, 123_456)
        self.assertEqual(strategy.state.last_order_ts_ms, 123_456)
        self.assertEqual(strategy.state.layers, 1)
        self.assertEqual(strategy.state.avg_entry_price, position.avg_entry_price)

    def test_daily_report_time_uses_live_report_timezone(self) -> None:
        real_datetime = dt.datetime
        sg_tz = ZoneInfo("Asia/Singapore")

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
                dt.datetime(2026, 6, 1, 0, 30, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 6, 1, 10, 0, tzinfo=sg_tz),
            ),
            (
                dt.datetime(2026, 6, 1, 3, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 6, 2, 10, 0, tzinfo=sg_tz),
            ),
        ]
        for now_value, expected in cases:
            with patch.dict(os.environ, {"LIVE_REPORT_TIMEZONE": "Asia/Singapore"}):
                with patch("src.live.time_utils.dt.datetime", fixed_datetime(now_value)):
                    actual = next_daily_report_time(10, 0)

            self.assertEqual(actual, expected)
            self.assertEqual(actual.tzinfo, sg_tz)
            self.assertEqual(actual.utcoffset(), dt.timedelta(hours=8))

    def test_weekly_summary_time_uses_live_report_timezone(self) -> None:
        real_datetime = dt.datetime
        sg_tz = ZoneInfo("Asia/Singapore")

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
                dt.datetime(2026, 6, 1, 1, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 6, 1, 10, 0, tzinfo=sg_tz),
            ),
            (
                dt.datetime(2026, 6, 1, 3, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 6, 8, 10, 0, tzinfo=sg_tz),
            ),
            (
                dt.datetime(2026, 6, 2, 1, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 6, 8, 10, 0, tzinfo=sg_tz),
            ),
        ]
        for now_value, expected in cases:
            with patch.dict(os.environ, {"LIVE_REPORT_TIMEZONE": "Asia/Singapore"}):
                with patch("src.live.time_utils.dt.datetime", fixed_datetime(now_value)):
                    actual = next_weekly_summary_time(10, 0, weekday=0)

            self.assertEqual(actual, expected)
            self.assertEqual(actual.tzinfo, sg_tz)
            self.assertEqual(actual.utcoffset(), dt.timedelta(hours=8))
            self.assertEqual(actual.weekday(), 0)

    async def run_strategy_worker_once(self, strategy: FakeStrategy, cvd: FakeCvd,
                                       queue: asyncio.Queue[MarketTickEvent]) -> asyncio.Queue[TradeCommand]:
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        worker = asyncio.create_task(
            strategy_tick_worker(
                strategy_tick_queue=queue,
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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

    async def test_flat_transition_waits_for_settled_balance_before_record_flat(self) -> None:
        cleared = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1"))

        class SettlingFlatTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.cash_values = [20.9853, 22.3400, 22.3401]
                inner_self.equity_values = [22.3717, 22.3401, 22.3401]

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                return inner_self.equity_values.pop(0)

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(inner_self.cash_values.pop(0))}]}]}

        class ClearingStateStore(RecordingStateStore):
            def clear(inner_self) -> None:
                super().clear()
                cleared.set()

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
        account_snapshot = AccountSnapshot(live_position, 22.3401, 22.3401, asyncio.get_running_loop().time(), 0, 1)
        execution_state = ExecutionState("pos-1", 22.3401)
        journal = FakeJournal()
        state_store = ClearingStateStore()

        with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "3",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "FLAT_BALANCE_STABLE_DELTA_USDT": "0.05",
                    "FLAT_BALANCE_CASH_EQUITY_MAX_DIFF_USDT": "0.10",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    trader=SettlingFlatTrader(),  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=strategy,  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=state_store,  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=999,
                    cash_log_min_delta_usdt=999,
                )
            )
            await asyncio.wait_for(cleared.wait(), timeout=0.2)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(journal.flats), 1)
        self.assertEqual(journal.flats[0]["cash_after"], 22.3401)
        self.assertEqual(journal.flats[0]["equity_after"], 22.3401)
        self.assertAlmostEqual(journal.flats[0]["realized_pnl_usdt_est"], 0.0, places=6)
        self.assertEqual(journal.cash_transfers, [])
        self.assertEqual(state_store.clear_calls, 1)
        self.assertFalse(execution_state.trading_halted)
        self.assertEqual(strategy.state.layers, 0)

    async def test_flat_balance_timeout_falls_back_to_equity(self) -> None:
        flat_recorded = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1"))

        class UnsettledFlatTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                return 22.34

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": "20.98"}]}]}

        class RecordingJournal(FakeJournal):
            def record_flat(inner_self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().record_flat(**kwargs)
                flat_recorded.set()

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
        account_snapshot = AccountSnapshot(live_position, 22.34, 22.34, asyncio.get_running_loop().time(), 0, 1)
        journal = RecordingJournal()

        with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "FLAT_BALANCE_STABLE_DELTA_USDT": "0.05",
                    "FLAT_BALANCE_CASH_EQUITY_MAX_DIFF_USDT": "0.10",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            with self.assertLogs("src.live.account_sync.flat_settlement_phase", level="WARNING") as logs:
                task = asyncio.create_task(
                    account_position_sync_worker(
                        state_lock=asyncio.Lock(),
                        account_snapshot=account_snapshot,
                        execution_state=ExecutionState("pos-1", 22.34),
                        trader=UnsettledFlatTrader(),  # type: ignore[arg-type]
                        sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                        strategy=strategy,  # type: ignore[arg-type]
                        journal=journal,  # type: ignore[arg-type]
                        state_store=FakeStateStore(),  # type: ignore[arg-type]
                        position_sync_seconds=0,
                        account_sync_seconds=999,
                        cash_log_min_delta_usdt=999,
                    )
                )
                await asyncio.wait_for(flat_recorded.wait(), timeout=0.2)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self.assertEqual(len(journal.flats), 1)
        self.assertEqual(journal.flats[0]["cash_after"], 22.34)
        self.assertEqual(journal.flats[0]["equity_after"], 22.34)
        self.assertIn("fallback_to_equity_after_timeout", "\n".join(logs.output))
        self.assertEqual(journal.cash_transfers, [])

    async def test_fetch_settled_flat_balance_returns_stable_cash_when_cash_equity_converge(self) -> None:
        class SettlingFlatTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.cash_values = [20.9853, 22.3400, 22.3401]
                inner_self.equity_values = [22.3717, 22.3401, 22.3401]

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                return inner_self.equity_values.pop(0)

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(inner_self.cash_values.pop(0))}]}]}

        settled = await fetch_settled_flat_balance(
            SettlingFlatTrader(),  # type: ignore[arg-type]
            attempts=3,
            interval_seconds=0,
            stable_delta_usdt=0.05,
            cash_equity_max_diff_usdt=0.10,
        )

        self.assertTrue(settled.stable)
        self.assertEqual(settled.reason, "cash_equity_stable")
        self.assertEqual(settled.cash, 22.3401)
        self.assertEqual(settled.equity, 22.3401)
        self.assertEqual(settled.attempts, 3)

    async def test_flat_transition_skips_cash_transfer_same_sync_cycle(self) -> None:
        cleared = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1"))
        cash_after = 20.9853

        class SameCycleFlatTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.cash_values = [cash_after, cash_after]
                inner_self.equity_values = [cash_after, cash_after]

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                if inner_self.equity_values:
                    return inner_self.equity_values.pop(0)
                return cash_after

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                if inner_self.cash_values:
                    cash = inner_self.cash_values.pop(0)
                else:
                    cash = cash_after
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(cash)}]}]}

        class RecordingStateStore(FakeStateStore):
            def __init__(inner_self) -> None:
                inner_self.clear_calls = 0

            def clear(inner_self) -> None:
                inner_self.clear_calls += 1
                cleared.set()

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
        account_snapshot = AccountSnapshot(live_position, 22.3401, 22.3401, asyncio.get_running_loop().time(), 0, 1)
        execution_state = ExecutionState("pos-1", 22.3401)
        journal = FakeJournal()
        state_store = RecordingStateStore()

        with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    trader=SameCycleFlatTrader(),  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=strategy,  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=state_store,  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=0,
                    cash_log_min_delta_usdt=999,
                )
            )
            await asyncio.wait_for(cleared.wait(), timeout=0.2)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(journal.flats), 1)
        self.assertEqual(journal.cash_transfers, [])
        self.assertEqual(journal.cash_drifts, [])
        self.assertEqual(account_snapshot.cash, cash_after)
        self.assertIsNone(execution_state.current_position_id)
        self.assertEqual(strategy.state.layers, 0)
        self.assertEqual(state_store.clear_calls, 1)

    async def test_flat_settle_cash_delta_does_not_record_cash_transfer(self) -> None:
        flat_recorded = asyncio.Event()
        allow_second_sync = asyncio.Event()
        second_sync = asyncio.Event()
        live_position = PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1"))

        class FlatSettleTrader(FakeTrader):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.position_calls = 0
                inner_self.cash_values = [100.0, 150.0, 150.0, 140.0]
                inner_self.equity_values = [100.0, 150.0, 150.0, 140.0]

            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                inner_self.position_calls += 1
                if inner_self.position_calls >= 2:
                    second_sync.set()
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                if inner_self.equity_values:
                    return inner_self.equity_values.pop(0)
                return 140.0

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                if inner_self.cash_values:
                    cash = inner_self.cash_values.pop(0)
                else:
                    cash = 140.0
                if cash == 140.0:
                    await allow_second_sync.wait()
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(cash)}]}]}

        class RecordingJournal(FakeJournal):
            def record_flat(inner_self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().record_flat(**kwargs)
                flat_recorded.set()

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
        execution_state = ExecutionState("pos-1", 100.0)
        journal = RecordingJournal()
        trader = FlatSettleTrader()

        with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    trader=trader,  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=strategy,  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=FakeStateStore(),  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=0,
                    cash_log_min_delta_usdt=999,
                )
            )
            await asyncio.wait_for(flat_recorded.wait(), timeout=0.2)
            self.assertEqual(account_snapshot.cash, 150.0)
            allow_second_sync.set()
            await asyncio.wait_for(second_sync.wait(), timeout=0.2)
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(journal.flats), 1)
        self.assertEqual(journal.flats[0]["cash_after"], 150.0)
        self.assertEqual(account_snapshot.cash, 140.0)
        self.assertEqual(journal.cash_transfers, [])
        self.assertEqual(len(journal.cash_drifts), 1)
        self.assertIn("flat_settle_cooldown", journal.cash_drifts[0]["reason"])

    async def test_cash_transfer_allowed_after_flat_cooldown_expires(self) -> None:
        transfer_recorded = asyncio.Event()

        class FlatCashTransferTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return flat_position()

            async def fetch_usdt_equity(inner_self) -> float:
                return 98.5

            async def request(inner_self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
                return {"data": [{"details": [{"ccy": "USDT", "cashBal": "98.5"}]}]}

        class RecordingJournal(FakeJournal):
            def record_cash_transfer(inner_self, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().record_cash_transfer(**kwargs)
                transfer_recorded.set()

        strategy = FakeStrategy()
        account_snapshot = AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1)
        execution_state = ExecutionState(None, None)
        journal = RecordingJournal()

        with patch.dict(
                os.environ,
                {
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    trader=FlatCashTransferTrader(),  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=strategy,  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=FakeStateStore(),  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=0,
                    cash_log_min_delta_usdt=999,
                )
            )
            await asyncio.wait_for(transfer_recorded.wait(), timeout=0.2)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(journal.cash_transfers), 1)
        self.assertEqual(journal.cash_transfers[0]["direction"], "WITHDRAWAL")
        self.assertEqual(journal.cash_transfers[0]["cash_before"], 100.0)
        self.assertEqual(journal.cash_transfers[0]["cash_after"], 98.5)
        self.assertEqual(journal.cash_drifts, [])

    async def test_account_sync_syncs_strategy_cost_even_with_pending_orders(self) -> None:
        """Strategy cost must sync even when execution is pending.
        (Changed from pre-fix behavior where sync was gated on pending_order_count==0.)"""
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
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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
        # Cost is now synced even with pending orders
        self.assertEqual(strategy.state.total_entry_qty, 2.0,
                         "total_entry_qty must sync from position even with pending orders")
        self.assertEqual(strategy.state.avg_entry_price, 99.0,
                         "avg_entry_price must sync from position even with pending orders")

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
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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
            await execution_queue.put(
                TradeCommand(intent(ts_ms), StrategyPositionState(), ts_ms, asyncio.get_running_loop().time(), 0,
                             "test"))
        trader = FakeTrader()
        task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                execution_state=ExecutionState(None, None),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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

    async def test_execution_worker_respects_sidecar_skip_first_layer_false(self) -> None:
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        entry_intent = intent(1_000, "OPEN_LONG")
        state = StrategyPositionState(
            side="LONG",
            sidecar_enabled_for_position=True,
            sidecar_margin_pct=0.01,
            sidecar_tp_pct=0.004,
        )
        await execution_queue.put(
            TradeCommand(entry_intent, copy.deepcopy(state), entry_intent.ts_ms, asyncio.get_running_loop().time(), 0,
                         "test")
        )
        trader = SidecarWorkerTrader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        strategy = types.SimpleNamespace(state=state, config=BollCvdReclaimStrategyConfig())
        task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                execution_state=ExecutionState(None, None),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
                trader=trader,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                journal=journal,  # type: ignore[arg-type]
                state_store=state_store,  # type: ignore[arg-type]
                email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
                sidecar_skip_first_layer=False,
            )
        )
        await asyncio.wait_for(execution_queue.join(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(len(trader.executed_intents), 1)
        executed_intent = trader.executed_intents[0]
        self.assertEqual(executed_intent.size.eth_qty, 1.0)
        self.assertEqual(executed_intent.managed_core_contracts, "5.00")
        self.assertEqual(executed_intent.managed_core_eth_qty, 0.5)
        self.assertEqual(len(trader.sidecar_tps), 1)
        self.assertEqual(trader.sidecar_tps[0][0], "LONG")
        self.assertEqual(trader.sidecar_tps[0][1], "5.00")
        self.assertAlmostEqual(trader.sidecar_tps[0][2], 100.4)
        self.assertEqual([event[0] for event in journal.events], ["SIDECAR_LEG_OPENED", "SIDECAR_TP_PLACED"])
        self.assertEqual(state.sidecar_legs[0]["layer_index"], 1)
        self.assertEqual(trader.sidecar_tps[0][3], state.sidecar_legs[0]["sidecar_client_order_id"])

    async def test_execution_worker_respects_sidecar_skip_first_layer_true(self) -> None:
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        entry_intent = intent(1_000, "OPEN_LONG")
        state = StrategyPositionState(
            side="LONG",
            sidecar_enabled_for_position=True,
            sidecar_margin_pct=0.01,
            sidecar_tp_pct=0.004,
        )
        await execution_queue.put(
            TradeCommand(entry_intent, copy.deepcopy(state), entry_intent.ts_ms, asyncio.get_running_loop().time(), 0,
                         "test")
        )
        trader = SidecarWorkerTrader()
        journal = FakeJournal()
        strategy = types.SimpleNamespace(state=state, config=BollCvdReclaimStrategyConfig())
        task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                execution_state=ExecutionState(None, None),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
                trader=trader,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                journal=journal,  # type: ignore[arg-type]
                state_store=RecordingStateStore(),  # type: ignore[arg-type]
                email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
                sidecar_skip_first_layer=True,
            )
        )
        await asyncio.wait_for(execution_queue.join(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertEqual(len(trader.executed_intents), 1)
        executed_intent = trader.executed_intents[0]
        self.assertEqual(executed_intent.size.eth_qty, 0.5)
        self.assertEqual(executed_intent.managed_core_contracts, "5.00")
        self.assertEqual(executed_intent.managed_core_eth_qty, 0.5)
        self.assertEqual(trader.sidecar_tps, [])
        self.assertNotIn("SIDECAR_LEG_OPENED", [event[0] for event in journal.events])
        self.assertNotIn("SIDECAR_TP_PLACED", [event[0] for event in journal.events])
        self.assertEqual(state.sidecar_legs, [])

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
        with self.assertLogs("src.live.workers.strategy_tick_worker", level="WARNING") as logs:
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
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
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
        with self.assertLogs("src.live.workers.account_position_sync_worker", level="WARNING"):
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

        with patch.dict(os.environ,
                        {"ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS": "60", "ACCOUNT_SYNC_STALE_WARN_SECONDS": "180"}):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(),
                                                     0, 1),
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
            with self.assertLogs("src.live.workers.account_position_sync_worker", level="WARNING") as logs:
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

        with patch.dict(os.environ,
                        {"ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS": "0", "ACCOUNT_SYNC_STALE_WARN_SECONDS": "999"}):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(),
                                                     0, 1),
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
            with self.assertLogs("src.live.workers.account_position_sync_worker", level="WARNING") as logs:
                await asyncio.wait_for(fetched_after_recovery.wait(), timeout=0.2)
                await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        output = "\n".join(logs.output)
        self.assertIn("ACCOUNT_SYNC_RECOVERED | failures=2", output)
        self.assertIn("ACCOUNT_SYNC_FAILED | failures=1", output.split("ACCOUNT_SYNC_RECOVERED | failures=2", 1)[1])

    # ── startup force TP reconcile ──────────────────────────────────────

    def test_startup_recovery_sets_force_tp_reconcile_flag(self) -> None:
        """After startup recovery with has_position, startup_force_tp_reconcile must be True."""
        from src.live.startup_recovery.basic_restore import restore_strategy_from_saved_state
        import types

        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        config = BollCvdReclaimStrategyConfig(three_stage_runner_enabled=False)
        strategy = BollCvdShockReclaimStrategy(config, sizer)

        saved_state = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            tp_price=110.0,
            tp_order_id=None,
            tp_order_ids=[],
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan="SINGLE",
            partial_tp_consumed=False,
            last_order_ts_ms=1_000,
            first_entry_ts_ms=1_000,
            add_freeze_until_ts_ms=2_800_000,
            add_freeze_penalty_count=1,
            last_tp_update_ts_ms=1_000,
            last_tp_update_candle_ts_ms=1_000,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            breakeven_price=100.0,
            position_cost_entry_notional=100.0,
            position_cost_exit_notional=0.0,
            position_cost_remaining_qty=1.0,
            net_remaining_breakeven_price=100.0,
            tp_mode="MIDDLE",
            startup_force_tp_reconcile=False,
        )

        restore_strategy_from_saved_state(strategy, saved_state)
        # Startup code sets the flag AFTER restore via strategy.state.startup_force_tp_reconcile = True
        strategy.state.startup_force_tp_reconcile = True

        self.assertTrue(strategy.state.startup_force_tp_reconcile)
        self.assertEqual(strategy.state.side, "LONG")
        self.assertEqual(strategy.state.layers, 1)

    def test_saved_state_restore_preserves_startup_force_tp_reconcile(self) -> None:
        """Restore from saved state reads startup_force_tp_reconcile."""
        from src.live.startup_recovery.basic_restore import restore_strategy_from_saved_state
        import types

        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        config = BollCvdReclaimStrategyConfig(three_stage_runner_enabled=False)
        strategy = BollCvdShockReclaimStrategy(config, sizer)

        saved_state = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            last_entry_price=100.0,
            tp_price=110.0,
            tp_order_id=None,
            tp_order_ids=[],
            partial_tp_price=None,
            partial_tp_ratio=0.0,
            tp_plan="SINGLE",
            partial_tp_consumed=False,
            last_order_ts_ms=1_000,
            first_entry_ts_ms=1_000,
            add_freeze_until_ts_ms=2_800_000,
            add_freeze_penalty_count=1,
            last_tp_update_ts_ms=1_000,
            last_tp_update_candle_ts_ms=1_000,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            breakeven_price=100.0,
            position_cost_entry_notional=100.0,
            position_cost_exit_notional=0.0,
            position_cost_remaining_qty=1.0,
            net_remaining_breakeven_price=100.0,
            tp_mode="MIDDLE",
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            three_stage_pre_tp1_degraded_ts_ms=10_900_000,
            middle_runner_sl_time_tighten_candle_count=5,
            middle_runner_sl_time_tighten_last_candle_ts_ms=50_000,
            three_stage_post_tp1_sl_time_tighten_candle_count=6,
            three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=60_000,
            startup_force_tp_reconcile=True,  # saved as True from a previous interrupted startup
        )

        restore_strategy_from_saved_state(strategy, saved_state)

        self.assertTrue(strategy.state.startup_force_tp_reconcile,
                        "startup_force_tp_reconcile should be restored from saved state")
        self.assertEqual(strategy.state.add_freeze_until_ts_ms, 2_800_000)
        self.assertEqual(strategy.state.add_freeze_penalty_count, 1)
        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, 10_900_000)
        self.assertEqual(strategy.state.middle_runner_sl_time_tighten_candle_count, 5)
        self.assertEqual(strategy.state.middle_runner_sl_time_tighten_last_candle_ts_ms, 50_000)
        self.assertEqual(strategy.state.three_stage_post_tp1_sl_time_tighten_candle_count, 6)
        self.assertEqual(strategy.state.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms, 60_000)

    def test_force_tp_reconcile_not_armed_when_flat(self) -> None:
        """When startup position is FLAT, startup_force_tp_reconcile should remain False."""
        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        config = BollCvdReclaimStrategyConfig()
        strategy = BollCvdShockReclaimStrategy(config, sizer)

        # Simulate a FLAT startup: the flag is never set
        self.assertFalse(strategy.state.startup_force_tp_reconcile)
        # _maybe_update_tp should return None because side is None
        result = strategy._maybe_update_tp(100.0, 2_000, boll(), cvd_snapshot(2_000))
        self.assertIsNone(result)

    def test_force_tp_reconcile_protected_order_ids_includes_sidecar_tp(self) -> None:
        """UPDATE_TP intent with sidecar enabled includes sidecar TP in protected_order_ids."""
        strat = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(three_stage_runner_enabled=False),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strat.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_mode="MIDDLE",
            tp_plan="SINGLE",
            last_tp_update_candle_ts_ms=500,
            startup_force_tp_reconcile=True,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "pos-1:SC:1:1000",
                    "position_id": "pos-1",
                    "layer_index": 1,
                    "side": "LONG",
                    "entry_price": 100.0,
                    "qty": 0.1,
                    "contracts": "1",
                    "tp_price": 100.4,
                    "tp_order_id": "sc-tp-12345",
                    "status": "OPEN",
                    "ts_ms": 1_000,
                }
            ],
            core_contracts="9",
            core_eth_qty=0.9,
        )

        bands = BollSnapshot("ETH-USDT-SWAP", 2_000, 105.0, 102.0, 112.0, 92.0, 0.1, 0.1, True, True)
        got = strat._maybe_update_tp(105.0, 2_000, bands, cvd_snapshot(2_000))

        self.assertIsNotNone(got)
        self.assertIn("sc-tp-12345", got.protected_order_ids,
                      "Sidecar TP order ID must be in protected_order_ids")
        self.assertIn("startup_force_tp_reconcile", got.reason)
        self.assertIsNotNone(got.managed_core_contracts,
                             "managed_core_contracts must be set when sidecar enabled")

    # ── trusted startup saved state ─────────────────────────────────────

    def test_trusted_startup_saved_state_matching_side_layers_avg_and_qty(self) -> None:
        """Saved state matching current OKX position (side, layers, avg, qty) is trusted."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=100.0,
            total_entry_qty=6.0,
            sidecar_enabled_for_position=False,
        )
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 6.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved)

    def test_trusted_startup_saved_state_none_saved_state(self) -> None:
        """None saved_state returns None regardless of position."""
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 6.0, Decimal("6"))
        self.assertIsNone(trusted_startup_saved_state(None, pos))

    def test_trusted_startup_saved_state_flat_position(self) -> None:
        """Even with matching saved_state, a FLAT position rejects it."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
        )
        pos = PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))
        self.assertFalse(pos.has_position)

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result)

    def test_trusted_startup_saved_state_side_mismatch(self) -> None:
        """Saved state from a LONG position must not bind to current SHORT position."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            tp_order_id="old-long-tp",
        )
        pos = PositionSnapshot("SHORT", Decimal("6"), 100.0, 6.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result)

    def test_trusted_startup_saved_state_zero_layers(self) -> None:
        """Saved state with layers=0 is not trusted."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=0,
        )
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 6.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result)

    def test_untrusted_saved_state_not_used_for_main_tp_startup_recovery(self) -> None:
        """When saved_state is untrusted, apply_main_tp_startup_recovery gets None."""
        from src.live.startup_recovery.order_recovery import apply_main_tp_startup_recovery

        saved_state = types.SimpleNamespace(
            position_id="pos-old",
            side="LONG",
            layers=3,
            tp_order_id="old-tp-order-id",
        )
        # Current OKX position is SHORT → side mismatch → untrusted
        startup_pos = PositionSnapshot("SHORT", Decimal("6"), 100.0, 6.0, Decimal("6"))

        trusted = trusted_startup_saved_state(saved_state, startup_pos)
        self.assertIsNone(trusted, "Side-mismatched saved_state must not be trusted")

        # The caller in main() should pass trusted_saved_state (None) to the recovery function.
        # We verify that apply_main_tp_startup_recovery with saved_state=None does NOT
        # restore the old tp_order_id.
        trader = FakeTrader()
        execution_state = ExecutionState("pos-new", 100.0)
        journal = FakeJournal()

        async def _run():
            await apply_main_tp_startup_recovery(
                execution_state=execution_state,
                saved_state=None,  # ← trusted_saved_state is None
                startup_position=startup_pos,
                trader=trader,  # type: ignore[arg-type]
                journal=journal,  # type: ignore[arg-type]
            )

        asyncio.get_event_loop().run_until_complete(_run())
        # tp_order_id must NOT be set from the untrusted saved_state
        self.assertEqual(trader.tp_order_id, "")

    def test_trusted_saved_state_still_used_for_main_tp_startup_recovery(self) -> None:
        """When saved_state is trusted, tp_order_id is restored."""
        from src.live.startup_recovery.order_recovery import apply_main_tp_startup_recovery

        saved_state = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=100.0,
            total_entry_qty=6.0,
            tp_order_id="trusted-tp-order-id",
            tp_order_ids=[],
            sidecar_enabled_for_position=False,
            sidecar_legs=[],
        )
        startup_pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 6.0, Decimal("6"))

        trusted = trusted_startup_saved_state(saved_state, startup_pos)
        self.assertIs(trusted, saved_state, "Matching saved_state must be trusted")

        trader = FakeTrader()
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()

        async def _run():
            await apply_main_tp_startup_recovery(
                execution_state=execution_state,
                saved_state=saved_state,
                startup_position=startup_pos,
                trader=trader,  # type: ignore[arg-type]
                journal=journal,  # type: ignore[arg-type]
            )

        asyncio.get_event_loop().run_until_complete(_run())
        # tp_order_id must be restored from the trusted saved_state
        self.assertEqual(trader.tp_order_id, "trusted-tp-order-id")

    def test_untrusted_saved_state_not_used_for_three_stage_safety_gate(self) -> None:
        """Three-stage safety gate must not inspect untrusted saved_state."""
        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        config = BollCvdReclaimStrategyConfig(three_stage_runner_enabled=True)
        strategy = BollCvdShockReclaimStrategy(config, sizer)
        # Simulate a dirty post-TP1 state in saved_state
        saved_state = types.SimpleNamespace(
            position_id="pos-old",
            side="LONG",
            layers=3,
            three_stage_post_tp1_protective_sl_order_id="old-post-tp1-sl",
        )
        # Current position is SHORT → mismatch
        startup_pos = PositionSnapshot("SHORT", Decimal("6"), 100.0, 6.0, Decimal("6"))

        trusted = trusted_startup_saved_state(saved_state, startup_pos)
        self.assertIsNone(trusted, "Side-mismatched saved_state must not be trusted")

        execution_state = ExecutionState("pos-new", 100.0)
        journal = FakeJournal()
        state_store = RecordingStateStore()

        applied = apply_three_stage_startup_safety_gate(
            strategy=strategy,
            execution_state=execution_state,
            saved_state=None,  # ← trusted_saved_state is None
            startup_position=startup_pos,
            journal=journal,  # type: ignore[arg-type]
            state_store=state_store,  # type: ignore[arg-type]
            trader_symbol="ETH-USDT-SWAP",
        )
        self.assertFalse(applied, "Untrusted saved_state must not trigger safety gate")

    def test_sidecar_not_recovered_from_untrusted_saved_state(self) -> None:
        """Sidecar enabled + legs in untrusted saved_state must not pollute current strategy."""
        from src.live.startup_recovery.order_recovery import apply_sidecar_startup_recovery

        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        config = BollCvdReclaimStrategyConfig()
        strategy = BollCvdShockReclaimStrategy(config, sizer)

        # Old saved_state has sidecar enabled with an OPEN leg
        saved_state = types.SimpleNamespace(
            position_id="pos-old",
            side="LONG",
            layers=3,
            sidecar_enabled_for_position=True,
            sidecar_margin_pct=0.05,
            sidecar_tp_pct=0.004,
            sidecar_legs=[
                {
                    "leg_id": "pos-old:SC:1:1000",
                    "position_id": "pos-old",
                    "layer_index": 1,
                    "side": "LONG",
                    "entry_price": 100.0,
                    "qty": 0.1,
                    "contracts": "1",
                    "tp_price": 100.4,
                    "tp_order_id": "sc-old-tp",
                    "status": "OPEN",
                    "ts_ms": 1_000,
                }
            ],
            sidecar_dirty=False,
            sidecar_halt_reason=None,
        )
        # Current position is SHORT → mismatch
        startup_pos = PositionSnapshot("SHORT", Decimal("6"), 100.0, 6.0, Decimal("6"))

        trusted = trusted_startup_saved_state(saved_state, startup_pos)
        self.assertIsNone(trusted, "Side-mismatched saved_state must not be trusted")

        execution_state = ExecutionState("pos-new", 100.0)
        journal = FakeJournal()
        state_store = RecordingStateStore()

        async def _run():
            await apply_sidecar_startup_recovery(
                strategy=strategy,
                execution_state=execution_state,
                saved_state=None,  # ← trusted_saved_state is None
                startup_position=startup_pos,
                trader=FakeTrader(),  # type: ignore[arg-type]
                journal=journal,  # type: ignore[arg-type]
                state_store=state_store,  # type: ignore[arg-type]
            )

        asyncio.get_event_loop().run_until_complete(_run())
        # Sidecar must NOT be enabled from untrusted saved_state
        self.assertFalse(
            strategy.state.sidecar_enabled_for_position,
            "Sidecar must not be enabled from untrusted saved_state",
        )
        self.assertEqual(
            len(strategy.state.sidecar_legs), 0,
            "No sidecar legs should be recovered from untrusted saved_state",
        )

    # ── tightened trusted_startup_saved_state checks ───────────────────

    def test_trusted_startup_saved_state_avg_mismatch_rejected(self) -> None:
        """Saved avg differs from OKX avg beyond tolerance → untrusted."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=1700.0,
            total_entry_qty=6.0,
            sidecar_enabled_for_position=False,
        )
        # OKX avg is 1750 → diff = 50/1750 ≈ 2.86%  > 0.3%
        pos = PositionSnapshot("LONG", Decimal("6"), 1750.0, 6.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result, "Avg mismatch beyond tolerance must reject saved_state")

    def test_trusted_startup_saved_state_qty_mismatch_rejected(self) -> None:
        """Saved qty differs from OKX qty beyond tolerance → untrusted."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=100.0,
            total_entry_qty=1.0,
            sidecar_enabled_for_position=False,
        )
        # OKX qty is 2.0 → diff = 1.0/2.0 = 50%  > 5%
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 2.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result, "Qty mismatch beyond tolerance must reject saved_state")

    def test_trusted_startup_saved_state_sidecar_qty_uses_core_plus_open_sidecar(self) -> None:
        """When sidecar enabled, expected qty = core_eth_qty + sidecar open qty."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=100.0,
            total_entry_qty=2.5,  # not used when sidecar enabled
            sidecar_enabled_for_position=True,
            core_eth_qty=2.0,
            sidecar_legs=[
                {
                    "leg_id": "pos-1:SC:1:1000",
                    "position_id": "pos-1",
                    "layer_index": 1,
                    "side": "LONG",
                    "entry_price": 100.0,
                    "qty": 0.5,
                    "contracts": "5",
                    "tp_price": 100.4,
                    "tp_order_id": "sc-tp-12345",
                    "status": "OPEN",
                    "ts_ms": 1_000,
                }
            ],
        )
        # core_eth_qty(2.0) + sidecar_open_qty(0.5) = 2.5 → matches OKX qty
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 2.5, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved)

    def test_trusted_startup_saved_state_sidecar_qty_mismatch_rejected(self) -> None:
        """Sidecar core+open qty differs from OKX qty → untrusted."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=100.0,
            sidecar_enabled_for_position=True,
            core_eth_qty=1.0,
            sidecar_legs=[
                {
                    "leg_id": "pos-1:SC:1:1000",
                    "qty": 0.5,
                    "status": "OPEN",
                    "tp_order_id": "sc-tp-12345",
                }
            ],
        )
        # core_eth_qty(1.0) + sidecar_open_qty(0.5) = 1.5, but OKX qty = 3.0
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 3.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result, "Sidecar qty mismatch must reject saved_state")

    def test_trusted_startup_saved_state_missing_avg_rejected(self) -> None:
        """Saved state with missing or zero avg_entry_price is untrusted."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=3,
            avg_entry_price=0.0,
            total_entry_qty=6.0,
            sidecar_enabled_for_position=False,
        )
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 6.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result, "Zero avg_entry in saved_state must be rejected")

    # ── expected remaining qty after partial exits ──────────────────────

    def test_trusted_startup_saved_state_after_three_stage_tp1_uses_remaining_qty(self) -> None:
        """After TP1 consumed, expected qty = total_entry * (tp2_ratio + runner_ratio)."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=10.0,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            trend_runner_active=False,
            sidecar_enabled_for_position=False,
        )
        # expected remaining: 10 * (0.2 + 0.2) = 4.0
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 4.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved,
                      "Three-Stage TP1 consumed must use remaining qty, not total_entry_qty=10")

    def test_trusted_startup_saved_state_after_three_stage_tp2_uses_runner_qty(self) -> None:
        """After TP2 consumed, expected qty = total_entry * runner_ratio."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=10.0,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=True,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            trend_runner_active=True,
            sidecar_enabled_for_position=False,
        )
        # expected remaining: 10 * 0.2 = 2.0
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 2.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved,
                      "Three-Stage TP2 consumed must use runner qty, not total_entry_qty=10")

    def test_trusted_startup_saved_state_middle_runner_active_uses_keep_ratio(self) -> None:
        """Middle Runner active: expected qty = total_entry * keep_ratio."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=10.0,
            middle_runner_enabled_for_position=True,
            middle_runner_pending=False,
            middle_runner_active=True,
            middle_runner_keep_ratio=0.2,
            sidecar_enabled_for_position=False,
        )
        # expected remaining: 10 * 0.2 = 2.0
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 2.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved,
                      "Middle Runner active must use keep_ratio qty, not total_entry_qty=10")

    def test_trusted_startup_saved_state_prefers_core_eth_qty(self) -> None:
        """core_eth_qty takes priority over total_entry_qty for remaining qty."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=10.0,
            core_eth_qty=4.0,
            sidecar_enabled_for_position=False,
        )
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 4.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved, "core_eth_qty must be preferred over total_entry_qty=10")

    def test_trusted_startup_saved_state_position_cost_remaining_qty_fallback(self) -> None:
        """position_cost_remaining_qty used when core_eth_qty is zero/absent."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=10.0,
            core_eth_qty=0.0,
            position_cost_remaining_qty=4.0,
            sidecar_enabled_for_position=False,
        )
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 4.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIs(result, saved, "position_cost_remaining_qty must be used when core_eth_qty=0")

    def test_trusted_startup_saved_state_remaining_qty_still_rejects_mismatch(self) -> None:
        """Even with correct remaining qty calc, a genuine mismatch is still rejected."""
        saved = types.SimpleNamespace(
            position_id="pos-1",
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=10.0,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            sidecar_enabled_for_position=False,
        )
        # expected remaining: 10 * (0.2 + 0.2) = 4.0, but OKX qty = 6.0 → diff 50%
        pos = PositionSnapshot("LONG", Decimal("6"), 100.0, 6.0, Decimal("6"))

        result = trusted_startup_saved_state(saved, pos)
        self.assertIsNone(result, "Genuine remaining qty mismatch must still be rejected")

    # ── helpers for Decimal contracts tests ──

    def _middle_runner_strategy(self, side: str = "LONG") -> BollCvdShockReclaimStrategy:
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(
                middle_runner_enabled=True,
                middle_runner_protective_sl_enabled=True,
                breakeven_fee_buffer_pct=0.001,
            ),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side=side,  # type: ignore[arg-type]
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0 if side == "LONG" else 90.0,
            tp_plan="MIDDLE_RUNNER",
            middle_runner_pending=True,
            middle_runner_first_close_ratio=0.8,
            middle_runner_keep_ratio=0.2,
            middle_runner_first_tp_price=105.0 if side == "LONG" else 95.0,
            middle_runner_final_tp_price=110.0 if side == "LONG" else 90.0,
            position_cost_entry_notional=100.0,
            position_cost_remaining_qty=0.2,
            net_remaining_breakeven_price=100.0,
        )
        return strategy

    # ── Test: three_stage post-TP1 payload passes Decimal contracts to Trader ──

    async def test_three_stage_post_tp1_sl_payload_passes_decimal_contracts_to_trader(self) -> None:
        class Tp1Trader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        strategy = self.three_stage_strategy("LONG")
        trader = Tp1Trader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=106.4, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: len(trader.post_tp1_stop_orders) >= 1,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        received = trader.post_tp1_stop_orders[0]
        self.assertIsInstance(received["contracts"], Decimal, "contracts must be Decimal, not float")
        self.assertEqual(received["contracts"], Decimal("4"))
        self.assertNotIsInstance(received["contracts"], float)

    # ── Test: middle_runner payload passes Decimal contracts to Trader ──

    async def test_middle_runner_sl_payload_passes_decimal_contracts_to_trader(self) -> None:
        class MidRunnerTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

        strategy = self._middle_runner_strategy("LONG")
        trader = MidRunnerTrader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        account_snapshot = AccountSnapshot(
            None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=106.4,
        )
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: len(trader.middle_runner_stop_orders) >= 1,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        received = trader.middle_runner_stop_orders[0]
        self.assertIsInstance(received["contracts"], Decimal, "contracts must be Decimal, not float")
        self.assertEqual(received["contracts"], Decimal("2"))
        self.assertNotIsInstance(received["contracts"], float)

    # ── Test: Trader.decimal_to_str accepts Decimal, str, int, float ──

    def test_trader_decimal_to_str_accepts_decimal_str_int_float(self) -> None:
        from src.execution.trader import Trader

        # Decimal
        self.assertEqual(Trader.decimal_to_str(Decimal("0.54")), "0.54")
        # str
        self.assertEqual(Trader.decimal_to_str("0.54"), "0.54")
        # float — must not raise
        result_float = Trader.decimal_to_str(0.54)
        self.assertIsInstance(result_float, str)
        # int
        self.assertEqual(Trader.decimal_to_str(1), "1")

    # ── Test: Trader._to_decimal converts various types ──

    def test_trader_to_decimal_converts_decimal_str_int_float(self) -> None:
        from src.execution.trader import Trader

        self.assertEqual(Trader._to_decimal(Decimal("0.54")), Decimal("0.54"))
        self.assertEqual(Trader._to_decimal("0.54"), Decimal("0.54"))
        self.assertIsInstance(Trader._to_decimal(0.54), Decimal)
        self.assertEqual(Trader._to_decimal(1), Decimal("1"))

    # ── Test: three_stage post-TP1 SL exception enters failure fallback ──

    async def test_three_stage_post_tp1_sl_exception_enters_failure_fallback(self) -> None:
        class ExceptionTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

            async def place_three_stage_post_tp1_protective_stop_with_retries(
                    inner_self, side, contracts, stop_price, retry_count, retry_interval_seconds,
            ):
                raise AttributeError("'float' object has no attribute 'normalize'")

        strategy = self.three_stage_strategy("LONG")
        trader = ExceptionTrader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        latest_ts_ms = int(dt.datetime.now().timestamp() * 1000)
        account_snapshot = AccountSnapshot(
            None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=106.4, latest_market_price_ts_ms=latest_ts_ms,
        )
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: execution_state.trading_halted,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertTrue(execution_state.trading_halted)
        self.assertIn(
            execution_state.halt_reason,
            ("three_stage_post_tp1_sl_failed_delayed_market_exit_armed",),
            f"Expected delayed market exit armed halt reason, got: {execution_state.halt_reason}",
        )
        # No immediate market exit
        self.assertEqual(
            len(trader.market_exits), 0,
            "market_exit_remaining_position_with_retries must NOT be called on SL failure (delayed exit armed instead)",
        )
        failed_events = [e for e in journal.events if e[0] == "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED"]
        self.assertEqual(len(failed_events), 1,
                         "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED journal event must be logged")

    # ── Test: middle_runner SL exception enters failure fallback ──

    async def test_middle_runner_sl_exception_enters_failure_fallback(self) -> None:
        class ExceptionTrader(FakeTrader):
            async def fetch_position_snapshot(inner_self) -> PositionSnapshot:
                return PositionSnapshot("LONG", Decimal("2"), 100.0, 0.2, Decimal("2"))

            async def place_middle_runner_protective_stop_with_retries(
                    inner_self, side, contracts, stop_price, retry_count, retry_interval_seconds,
            ):
                raise AttributeError("'float' object has no attribute 'normalize'")

        strategy = self._middle_runner_strategy("LONG")
        trader = ExceptionTrader()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        account_snapshot = AccountSnapshot(
            None, 100.0, 100.0, asyncio.get_running_loop().time(), 0, 1,
            latest_market_price=106.4,
        )
        execution_state = ExecutionState("pos-1", 100.0)

        await self.run_account_sync_until(
            lambda: execution_state.trading_halted,
            account_snapshot=account_snapshot,
            execution_state=execution_state,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        self.assertTrue(execution_state.trading_halted)
        # No immediate market exit on SL failure
        self.assertEqual(
            len(trader.market_exits), 0,
            "market_exit_remaining_position_with_retries must NOT be called (delayed exit armed instead)",
        )
        warning_events = [e for e in journal.events if e[0] == "MIDDLE_RUNNER_ORDER_WARNING"]
        self.assertEqual(len(warning_events), 1, "MIDDLE_RUNNER_ORDER_WARNING journal event must be logged")


# ============================================================================
# E06: source guard — worker_event_emitter wired through account worker
# ============================================================================


class TestE06WorkerEventEmitterSourceGuard(unittest.TestCase):
    def test_account_worker_signature_contains_worker_event_emitter(self) -> None:
        import inspect
        sig = inspect.signature(account_position_sync_worker)
        assert "worker_event_emitter" in sig.parameters, (
            "account_position_sync_worker must accept worker_event_emitter"
        )

    def test_account_worker_passes_worker_event_emitter_to_finalize(self) -> None:
        source_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "live" / "workers" / "account_position_sync_worker.py"
        )
        source = source_path.read_text()
        assert "worker_event_emitter=worker_event_emitter" in source, (
            "account_position_sync_worker must pass worker_event_emitter "
            "to finalize_account_sync_flat_settlement_phase"
        )

    def test_account_worker_passes_worker_event_emitter_to_resume(self) -> None:
        source_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "live" / "workers" / "account_position_sync_worker.py"
        )
        source = source_path.read_text()
        # The resume branch also calls record_and_notify_rolling_loss_guard
        # with worker_event_emitter=worker_event_emitter.
        # Verify there are two occurrences (one for finalize, one for resume).
        count = source.count("worker_event_emitter=worker_event_emitter")
        assert count >= 2, (
            f"account_position_sync_worker must pass worker_event_emitter "
            f"to both finalize and resume branches, found {count} occurrences"
        )

    def test_flat_settlement_finalize_signature_contains_worker_event_emitter(self) -> None:
        from src.live.account_sync.flat_settlement_phase import (
            finalize_account_sync_flat_settlement_phase,
        )
        import inspect
        sig = inspect.signature(finalize_account_sync_flat_settlement_phase)
        assert "worker_event_emitter" in sig.parameters, (
            "finalize_account_sync_flat_settlement_phase must accept worker_event_emitter"
        )

    def test_flat_settlement_finalize_passes_worker_event_emitter(self) -> None:
        source_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "live" / "account_sync" / "flat_settlement_phase.py"
        )
        source = source_path.read_text()
        assert "worker_event_emitter=worker_event_emitter" in source, (
            "finalize_account_sync_flat_settlement_phase must pass "
            "worker_event_emitter to record_and_notify_rolling_loss_guard"
        )


if __name__ == "__main__":
    unittest.main()
