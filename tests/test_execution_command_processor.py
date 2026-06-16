from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
import types
import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

import pytest

if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader
from src.live.runtime_types import AccountSnapshot, ExecutionState, TradeCommand
from src.live.workers.execution_command_processor import ExecutionCommandProcessor
from src.live.workers.execution_worker import execution_worker
from src.reporting.live_state_store import LiveStateStore
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig


# ── lightweight fakes ────────────────────────────────────────────────────

def flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


def long_position() -> PositionSnapshot:
    return PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))


def make_intent(ts_ms: int = 1_000, intent_type: str = "OPEN_LONG") -> TradeIntent:
    return TradeIntent(
        intent_type=intent_type,  # type: ignore[arg-type]
        side="LONG",
        price=100.0,
        layer_index=1,
        tp_price=101.0,
        reason="test",
        size=PositionSize(1.0, 50.0, 1.0, 1, 1.0),
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


class FakeJournal:
    def __init__(self) -> None:
        self.entries: list[int] = []
        self.events: list[tuple] = []
        self.tp_updates: list[dict] = []
        self.trend_exits: list[dict] = []
        self.errors: list[dict] = []

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"

    def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.entries.append(kwargs["intent"].ts_ms)

    def record_tp_update(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.tp_updates.append(kwargs)

    def record_trend_runner_market_exit(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.trend_exits.append(kwargs)

    def record_error(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.errors.append(kwargs)

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event_name, dict(payload), position_id))

    def record_flat(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_cash_transfer(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_account_cash_drift(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved.append(state)

    def clear(self) -> None:
        pass


class FakeEmailSender:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    async def send_email_async(self, subject: str, content: str, content_type: str = "html") -> bool:
        self.sent.append((subject, content, content_type))
        return True


class FakeTradingClient:
    """Minimal TradingClientPort stub backing the FakeTrader below."""

    def __init__(self, trader: object) -> None:
        self._trader = trader

    async def fetch_balance(self):
        from decimal import Decimal as D
        from src.execution.trading_client_port import BalanceSnapshot

        res = await self._trader.request(  # type: ignore[attr-defined]
            "GET", "/api/v5/account/balance?ccy=USDT"
        )
        data = res.get("data", [])
        equity = 0.0
        cash = 0.0
        if data:
            details = data[0].get("details", [])
            for item in details:
                if item.get("ccy") == "USDT":
                    equity = float(
                        item.get("eq")
                        or item.get("availEq")
                        or item.get("availBal")
                        or 0.0
                    )
                    cash = float(
                        item.get("cashBal")
                        or item.get("availBal")
                        or item.get("availEq")
                        or item.get("eq")
                        or 0.0
                    )
                    break
            if equity == 0.0:
                equity = float(data[0].get("totalEq") or 0.0)
                if cash == 0.0:
                    cash = equity
        return BalanceSnapshot(
            asset="USDT",
            total=D(str(equity)),
            available=D(str(cash)) if cash else None,
            raw={"account_equity_usdt": equity, "cash_balance_usdt": cash},
        )


class FakeTrader:
    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.account_equity_usdt = 1000.0
        self.position_contracts = Decimal("0")
        self.executed: list[int] = []
        self.executed_intents: list[TradeIntent] = []
        self._next_result: LiveTradeResult | None = None
        self._position: PositionSnapshot = flat_position()
        self._cash_balance: float = 1000.0
        self.config = types.SimpleNamespace(leverage=50)
        self.market_exits: list[tuple] = []
        self._market_exit_returns: tuple[bool, str] = (True, "ok")
        self._fetch_position_raises: Exception | None = None
        self._market_exit_raises: Exception | None = None
        self.trading_client = FakeTradingClient(self)

    async def execute_intent(self, trade_intent: TradeIntent) -> LiveTradeResult:
        self.executed.append(trade_intent.ts_ms)
        self.executed_intents.append(trade_intent)
        if self._next_result is not None:
            r = self._next_result
            self._next_result = None
            return r
        return LiveTradeResult(
            ok=True,
            action=trade_intent.intent_type,
            order_id=f"ord-{trade_intent.ts_ms}",
            tp_order_id=f"tp-{trade_intent.ts_ms}",
            contracts="10",
            tp_price="101",
            message="ok",
            entry_filled=True,
            tp_ok=True,
            tp_order_ids=(f"tp-{trade_intent.ts_ms}",),
        )

    async def market_exit_remaining_position_with_retries(
        self, side: str, retry_count: int, *, context: str = "generic", retry_interval_seconds: float | None = None,
    ) -> tuple[bool, str]:
        self.market_exits.append((side, retry_count, context, retry_interval_seconds))
        if self._market_exit_raises is not None:
            raise self._market_exit_raises
        return self._market_exit_returns

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        if self._fetch_position_raises is not None:
            raise self._fetch_position_raises
        return self._position

    async def fetch_usdt_equity(self) -> float:
        return self.account_equity_usdt

    async def request(self, method: str, endpoint: str, payload=None) -> dict:  # type: ignore[no-untyped-def]
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": str(self._cash_balance)}]}]}

    def set_next_result(self, result: LiveTradeResult) -> None:
        self._next_result = result

    def set_position(self, position: PositionSnapshot) -> None:
        self._position = position

    def set_cash_balance(self, balance: float) -> None:
        self._cash_balance = balance


# ── helpers ──────────────────────────────────────────────────────────────

def make_command(
    ts_ms: int = 1_000,
    intent_type: str = "OPEN_LONG",
    snapshot: StrategyPositionState | None = None,
) -> TradeCommand:
    if snapshot is None:
        snapshot = StrategyPositionState(side="LONG")
    return TradeCommand(
        make_intent(ts_ms, intent_type),
        copy.deepcopy(snapshot),
        ts_ms,
        asyncio.get_running_loop().time(),
        0,
        "test",
    )


def make_processor(
    state_lock: asyncio.Lock | None = None,
    execution_state: ExecutionState | None = None,
    account_snapshot: AccountSnapshot | None = None,
    trader: FakeTrader | None = None,
    strategy: BollCvdShockReclaimStrategy | None = None,
    journal: FakeJournal | None = None,
    state_store: FakeStateStore | None = None,
    email_sender: FakeEmailSender | None = None,
) -> tuple[ExecutionCommandProcessor, FakeTrader, FakeJournal, FakeStateStore]:
    if state_lock is None:
        state_lock = asyncio.Lock()
    if execution_state is None:
        execution_state = ExecutionState(None, None)
    if account_snapshot is None:
        account_snapshot = AccountSnapshot(flat_position(), 1000.0, 1000.0, 0.0, 0, 1)
    if trader is None:
        trader = FakeTrader()
    if strategy is None:
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
    if journal is None:
        journal = FakeJournal()
    if state_store is None:
        state_store = FakeStateStore()
    if email_sender is None:
        email_sender = FakeEmailSender()
    processor = ExecutionCommandProcessor(
        state_lock=state_lock,
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,  # type: ignore[arg-type]
        strategy=strategy,
        journal=journal,  # type: ignore[arg-type]
        state_store=state_store,  # type: ignore[arg-type]
        email_sender=email_sender,  # type: ignore[arg-type]
    )
    return processor, trader, journal, state_store


def three_stage_strategy(side: str = "LONG") -> BollCvdShockReclaimStrategy:
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


# ── tests ────────────────────────────────────────────────────────────────

class TestExecutionCommandProcessor(unittest.IsolatedAsyncioTestCase):
    """Tests for ExecutionCommandProcessor.process()."""

    # ── worker delegates to processor ────────────────────────────────────

    async def test_worker_delegates_to_processor(self) -> None:
        """execution_worker calls processor.process and still decrements pending / task_done."""
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        command = make_command(1_000, "OPEN_LONG")
        await execution_queue.put(command)

        execution_state = ExecutionState(None, None, pending_order_count=1)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()

        with patch.object(
            ExecutionCommandProcessor,
            "process",
            return_value=LiveTradeResult(
                ok=True,
                action="OPEN_LONG",
                order_id="ord-1000",
                tp_order_id="tp-1000",
                contracts="10",
                tp_price="101",
                message="ok",
                entry_filled=True,
                tp_ok=True,
            ),
        ) as mock_process:
            task = asyncio.create_task(
                execution_worker(
                    execution_queue=execution_queue,
                    state_lock=asyncio.Lock(),
                    execution_state=execution_state,
                    account_snapshot=AccountSnapshot(flat_position(), 1000.0, 1000.0, 0.0, 0, 1),
                    trader=trader,  # type: ignore[arg-type]
                    strategy=BollCvdShockReclaimStrategy(
                        BollCvdReclaimStrategyConfig(),
                        SimplePositionSizer(SimplePositionSizerConfig()),
                    ),
                    journal=journal,  # type: ignore[arg-type]
                    state_store=state_store,  # type: ignore[arg-type]
                    email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                    backlog_log_seconds=999,
                )
            )
            await asyncio.wait_for(execution_queue.join(), timeout=1)
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task

        mock_process.assert_called_once()
        self.assertEqual(execution_state.pending_order_count, 0)

    async def test_worker_failure_handler_called_on_exception(self) -> None:
        """failure handler still called by execution_worker when processor.process raises."""
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        command = make_command(1_000, "OPEN_LONG")
        await execution_queue.put(command)

        execution_state = ExecutionState(None, None, pending_order_count=1)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()

        with patch.object(
            ExecutionCommandProcessor,
            "process",
            side_effect=RuntimeError("test error"),
        ):
            task = asyncio.create_task(
                execution_worker(
                    execution_queue=execution_queue,
                    state_lock=asyncio.Lock(),
                    execution_state=execution_state,
                    account_snapshot=AccountSnapshot(flat_position(), 1000.0, 1000.0, 0.0, 0, 1),
                    trader=trader,  # type: ignore[arg-type]
                    strategy=BollCvdShockReclaimStrategy(
                        BollCvdReclaimStrategyConfig(),
                        SimplePositionSizer(SimplePositionSizerConfig()),
                    ),
                    journal=journal,  # type: ignore[arg-type]
                    state_store=state_store,  # type: ignore[arg-type]
                    email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                    backlog_log_seconds=999,
                )
            )
            await asyncio.wait_for(execution_queue.join(), timeout=1)
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task

        self.assertEqual(execution_state.pending_order_count, 0)
        # When result is None and contracts=0, failure handler rolls back state without halting
        self.assertEqual(len(journal.errors), 1)

    # ── dirty post-TP1 SL guard ──────────────────────────────────────────

    async def test_dirty_post_tp1_sl_guard_blocks_command(self) -> None:
        """When three_stage_dirty_post_tp1_sl_after_tp2 is True, command is skipped."""
        strategy = three_stage_strategy("LONG")
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_tp2_consumed = True
        strategy.state.trend_runner_active = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True

        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )

        command = make_command(123_456, "UPDATE_TP")
        result = await processor.process(command)

        self.assertIsNone(result)
        self.assertEqual(trader.executed, [])
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(
            execution_state.halt_reason,
            "three_stage_post_tp1_sl_dirty_state_blocked",
        )

    async def test_dirty_post_tp1_sl_guard_does_not_double_record(self) -> None:
        """When already halted with dirty reason, does not re-append event."""
        strategy = three_stage_strategy("LONG")
        strategy.state.three_stage_tp1_consumed = True
        strategy.state.three_stage_tp2_consumed = True
        strategy.state.trend_runner_active = True
        strategy.state.three_stage_post_tp1_protective_sl_order_id = "old-post"
        strategy.state.three_stage_post_tp1_protective_sl_price = 101.0
        strategy.state.three_stage_post_tp1_protected = True

        execution_state = ExecutionState(
            "pos-1",
            100.0,
            trading_halted=True,
            halt_reason="three_stage_post_tp1_sl_dirty_state_blocked",
        )
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
        )

        command = make_command(123_456, "UPDATE_TP")
        result = await processor.process(command)

        self.assertIsNone(result)
        # Should NOT have appended a duplicate event
        dirty_events = [
            e for e in journal.events
            if e[0] == "THREE_STAGE_DIRTY_POST_TP1_SL_BLOCKED_RUNNER_UPDATE"
        ]
        self.assertEqual(len(dirty_events), 0)

    # ── trading halted guard ─────────────────────────────────────────────

    async def test_trading_halted_guard_skips_normal_command(self) -> None:
        """Normal entry command skipped when trading is halted."""
        execution_state = ExecutionState(
            "pos-1", 100.0, trading_halted=True, halt_reason="test_halt"
        )
        trader = FakeTrader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state, trader=trader
        )

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        self.assertIsNone(result)
        self.assertEqual(trader.executed, [])

    async def test_rolling_management_allowed_allows_management_intents(self) -> None:
        """UPDATE_TP allowed through when halt_reason is in ROLLING_LOSS_HALT_REASONS."""
        from src.risk.rolling_loss_guard import ROLLING_LOSS_HALT_REASONS

        halt_reason = next(iter(ROLLING_LOSS_HALT_REASONS))
        execution_state = ExecutionState(
            "pos-1", 100.0, trading_halted=True, halt_reason=halt_reason
        )
        trader = FakeTrader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state, trader=trader
        )

        command = make_command(1_000, "UPDATE_TP")
        # This should NOT be blocked by trading_halted guard
        result = await processor.process(command)

        # It should have executed (not None from trading halted skip)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(trader.executed), 1)

    async def test_rolling_management_does_not_allow_entry_intents(self) -> None:
        """OPEN_LONG still blocked even when halt_reason is rolling loss."""
        from src.risk.rolling_loss_guard import ROLLING_LOSS_HALT_REASONS

        halt_reason = next(iter(ROLLING_LOSS_HALT_REASONS))
        execution_state = ExecutionState(
            "pos-1", 100.0, trading_halted=True, halt_reason=halt_reason
        )
        trader = FakeTrader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state, trader=trader
        )

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        self.assertIsNone(result)
        self.assertEqual(trader.executed, [])

    # ── entry command creates position id ────────────────────────────────

    async def test_entry_creates_position_id_when_none(self) -> None:
        """OPEN_LONG creates new position_id when execution_state has None."""
        execution_state = ExecutionState(None, None)
        journal = FakeJournal()
        processor, trader, _, _ = make_processor(
            execution_state=execution_state,
            journal=journal,
        )

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertIsNotNone(execution_state.current_position_id)
        self.assertIn("ETH-USDT-SWAP", execution_state.current_position_id or "")
        self.assertEqual(len(journal.entries), 1)
        self.assertIsNotNone(execution_state.cash_before_position)

    # ── UPDATE_TP result application ─────────────────────────────────────

    async def test_update_tp_result_updates_strategy_state(self) -> None:
        """UPDATE_TP applies tp_order_id, journal.record_tp_update, state_store.save."""
        strategy = three_stage_strategy("LONG")
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        state_store = FakeStateStore()
        trader = FakeTrader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
            trader=trader,
        )

        command = make_command(1_000, "UPDATE_TP")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertEqual(strategy.state.tp_order_id, "tp-1000")
        self.assertEqual(len(journal.tp_updates), 1)
        self.assertGreaterEqual(len(state_store.saved), 1)

    async def test_update_tp_middle_runner_journal_append(self) -> None:
        """UPDATE_TP with middle_runner_active appends MIDDLE_RUNNER_TP_UPDATED."""
        strategy = three_stage_strategy("LONG")
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
        )

        intent = make_intent(1_000, "UPDATE_TP")
        # Add middle_runner_active attribute to intent
        middle_intent = TradeIntent(
            **{
                **intent.__dict__,
                "middle_runner_active": True,
                "middle_runner_pending": False,
            }
        )
        command = TradeCommand(
            middle_intent,
            copy.deepcopy(strategy.state),
            1_000,
            0.0,
            0,
            "test",
        )
        result = await processor.process(command)

        self.assertIsNotNone(result)
        middle_events = [
            e for e in journal.events if e[0] == "MIDDLE_RUNNER_TP_UPDATED"
        ]
        self.assertEqual(len(middle_events), 1)

    async def test_update_tp_trend_runner_journal_append(self) -> None:
        """UPDATE_TP with trend_runner_active appends TREND_RUNNER_UPDATE."""
        strategy = three_stage_strategy("LONG")
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
        )

        intent = make_intent(1_000, "UPDATE_TP")
        trend_intent = TradeIntent(
            **{
                **intent.__dict__,
                "trend_runner_active": True,
                "trend_runner_tp_price": 111.0,
                "trend_runner_sl_price": 101.0,
                "trend_runner_adjust_count": 1,
            }
        )
        command = TradeCommand(
            trend_intent,
            copy.deepcopy(strategy.state),
            1_000,
            0.0,
            0,
            "test",
        )
        result = await processor.process(command)

        self.assertIsNotNone(result)
        trend_events = [
            e for e in journal.events if e[0] == "TREND_RUNNER_UPDATE"
        ]
        self.assertEqual(len(trend_events), 1)

    async def test_update_tp_three_stage_post_tp1_journal_append(self) -> None:
        """UPDATE_TP with three_stage post TP1 appends THREE_STAGE_TP1_PROTECTIVE_SL_UPDATED."""
        strategy = three_stage_strategy("LONG")
        strategy.state.three_stage_tp1_consumed = True
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
        )

        intent = make_intent(1_000, "UPDATE_TP")
        ts_intent = TradeIntent(
            **{
                **intent.__dict__,
                "three_stage_post_tp1_protective_sl_price": 101.0,
                "three_stage_tp1_consumed": True,
                "three_stage_tp1_price": 101.0,
                "three_stage_tp1_ratio": 0.6,
                "three_stage_tp2_price": 110.0,
                "three_stage_tp2_ratio": 0.2,
                "three_stage_runner_ratio": 0.2,
                "trend_runner_active": False,
            }
        )
        command = TradeCommand(
            ts_intent,
            copy.deepcopy(strategy.state),
            1_000,
            0.0,
            0,
            "test",
        )
        result = await processor.process(command)

        self.assertIsNotNone(result)
        ts_events = [
            e for e in journal.events if e[0] == "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATED"
        ]
        self.assertEqual(len(ts_events), 1)

    # ── MARKET_EXIT_RUNNER ───────────────────────────────────────────────

    async def test_market_exit_runner_halts_and_journals(self) -> None:
        """MARKET_EXIT_RUNNER halts trading and records trend runner exit."""
        strategy = three_stage_strategy("LONG")
        strategy.state.trend_runner_active = True
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
        )

        command = make_command(1_000, "MARKET_EXIT_RUNNER")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(
            execution_state.halt_reason,
            "trend_runner_market_exit_waiting_flat",
        )
        self.assertEqual(len(journal.trend_exits), 1)
        self.assertGreaterEqual(len(state_store.saved), 1)

    # ── entry result ─────────────────────────────────────────────────────

    async def test_entry_success_journal_and_state(self) -> None:
        """Entry result calls journal.record_entry and state_store.save."""
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            journal=journal,
            state_store=state_store,
        )

        # Set strategy state with layers so it's not the default
        processor.strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="SINGLE",
        )

        command = make_command(1_000, "ADD_LONG")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertEqual(len(journal.entries), 1)
        self.assertGreaterEqual(len(state_store.saved), 1)
        self.assertEqual(processor.strategy.state.tp_order_id, "tp-1000")

    async def test_entry_middle_runner_planned_journal_append(self) -> None:
        """Entry with MIDDLE_RUNNER tp_plan appends MIDDLE_RUNNER_PLANNED."""
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            journal=journal,
        )
        processor.strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=105.0,
            partial_tp_ratio=0.8,
            middle_runner_keep_ratio=0.2,
        )

        intent = make_intent(1_000, "ADD_LONG")
        mr_intent = TradeIntent(
            **{
                **intent.__dict__,
                "tp_plan": "MIDDLE_RUNNER",
                "partial_tp_price": 105.0,
                "partial_tp_ratio": 0.8,
                "middle_runner_keep_ratio": 0.2,
            }
        )
        command = TradeCommand(
            mr_intent,
            copy.deepcopy(processor.strategy.state),
            1_000,
            0.0,
            0,
            "test",
        )
        result = await processor.process(command)

        self.assertIsNotNone(result)
        mr_events = [
            e for e in journal.events if e[0] == "MIDDLE_RUNNER_PLANNED"
        ]
        self.assertEqual(len(mr_events), 1)

    async def test_entry_three_stage_runner_planned_journal_append(self) -> None:
        """Entry with THREE_STAGE_RUNNER tp_plan appends THREE_STAGE_RUNNER_PLANNED."""
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            journal=journal,
        )
        processor.strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=102.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_price=110.0,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            trend_runner_tp_price=120.0,
            trend_runner_sl_price=100.0,
        )

        intent = make_intent(1_000, "ADD_LONG")
        ts_intent = TradeIntent(
            **{
                **intent.__dict__,
                "tp_plan": "THREE_STAGE_RUNNER",
                "three_stage_tp1_price": 102.0,
                "three_stage_tp1_ratio": 0.6,
                "three_stage_tp2_price": 110.0,
                "three_stage_tp2_ratio": 0.2,
                "three_stage_runner_ratio": 0.2,
                "trend_runner_tp_price": 120.0,
                "trend_runner_sl_price": 100.0,
            }
        )
        command = TradeCommand(
            ts_intent,
            copy.deepcopy(processor.strategy.state),
            1_000,
            0.0,
            0,
            "test",
        )
        result = await processor.process(command)

        self.assertIsNotNone(result)
        ts_events = [
            e for e in journal.events if e[0] == "THREE_STAGE_RUNNER_PLANNED"
        ]
        self.assertEqual(len(ts_events), 1)

    # ── result.ok False returns result (RuntimeError raised by worker) ──

    async def test_result_not_ok_returns_without_applying(self) -> None:
        """When trader.execute_intent returns ok=False, processor returns result without applying."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            tp_order_id="old-tp",
        )
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        trader.set_next_result(
            LiveTradeResult(
                ok=False,
                action="OPEN_LONG",
                order_id="",
                tp_order_id="",
                contracts="",
                tp_price="",
                message="insufficient margin",
            )
        )
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
        )

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        # Processor returns the result (even when ok=False)
        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertEqual(result.message, "insufficient margin")
        # Strategy state should NOT be modified (result application skipped)
        self.assertEqual(strategy.state.tp_order_id, "old-tp")

    # ── entry_cash_before for new positions ──────────────────────────────

    async def test_entry_cash_before_fetched_when_no_current_position(self) -> None:
        """When current_position_id is None, entry_cash_before is fetched from exchange."""
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        trader.set_cash_balance(999.99)
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
        )

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(execution_state.cash_before_position, 999.99)

    async def test_entry_cash_before_not_queried_for_update_tp(self) -> None:
        """UPDATE_TP does NOT query cash balance even when current_position_id is None."""
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        trader.set_cash_balance(999.99)
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
        )

        command = make_command(1_000, "UPDATE_TP")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        # For UPDATE_TP with None position_id, entry_cash_before = cash_before_position (None)
        self.assertIsNone(execution_state.cash_before_position)


if __name__ == "__main__":
    unittest.main()
