from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
import types
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader
from src.live.runtime_types import AccountSnapshot, ExecutionState, TradeCommand
from src.live.workers.execution_command_processor import ExecutionCommandProcessor
from src.portfolio.capital_ledger import CapitalLedgerSnapshot, SymbolCapitalState, default_snapshot
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
        self.near_tp_reduces: list[dict] = []
        self.trend_exits: list[dict] = []
        self.errors: list[dict] = []

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"

    def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.entries.append(kwargs["intent"].ts_ms)

    def record_tp_update(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.tp_updates.append(kwargs)

    def record_near_tp_reduce(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.near_tp_reduces.append(kwargs)

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


class FakeTrader:
    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.account_equity_usdt = 1000.0
        self.contract_multiplier = Decimal("0.1")
        self.contract_precision = Decimal("0.01")
        self.position_contracts = Decimal("0")
        self.executed: list[int] = []
        self.executed_intents: list[TradeIntent] = []
        self._next_result: LiveTradeResult | None = None
        self._position: PositionSnapshot = flat_position()
        self._cash_balance: float = 1000.0
        self.config = types.SimpleNamespace(leverage=50)
        # sidecar tracking
        self.cancelled_sidecar_tps: list[str] = []
        self.sidecar_tps: list[tuple] = []
        self.market_exits: list[tuple] = []
        self._cancel_sidecar_tp_returns: bool = True
        self._place_sidecar_tp_raises: Exception | None = None
        self._market_exit_returns: tuple[bool, str] = (True, "ok")
        self._fetch_position_raises: Exception | None = None
        self._market_exit_raises: Exception | None = None

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

    async def cancel_sidecar_take_profit(self, order_id: str) -> bool:
        self.cancelled_sidecar_tps.append(order_id)
        return self._cancel_sidecar_tp_returns

    async def place_sidecar_fixed_take_profit(
        self, *, side: str, contracts: str, tp_price: float, client_order_id: str | None = None
    ) -> str:
        if self._place_sidecar_tp_raises is not None:
            raise self._place_sidecar_tp_raises
        self.sidecar_tps.append((side, contracts, tp_price, client_order_id))
        return f"sidecar-tp-{len(self.sidecar_tps)}"

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


def add_layer_snapshot(planned_main_contracts: tuple[str, ...]) -> CapitalLedgerSnapshot:
    base = default_snapshot(updated_ms=1000)
    symbols = dict(base.symbols)
    symbols["ETH-USDT-SWAP"] = SymbolCapitalState(
        state="OPEN",
        side="LONG",
        used_layers=1,
        position_plan_id="plan-1",
        planned_main_contracts=planned_main_contracts,
        base_main_contracts="1",
        plan_max_layers=3,
        permission_max_layers=3,
        main_used_margin_usdt="30",
        sidecar_enabled=True,
    )
    return CapitalLedgerSnapshot(
        version=base.version,
        updated_ms=base.updated_ms,
        leader_symbol=base.leader_symbol,
        global_no_new_entry=base.global_no_new_entry,
        symbols=symbols,
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
    sidecar_skip_first_layer: bool = True,
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
        sidecar_skip_first_layer=sidecar_skip_first_layer,
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

    async def test_worker_prioritizes_market_exit_runner_over_update_tp(self) -> None:
        execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
        strategy = three_stage_strategy("LONG")
        update_command = make_command(1_000, "UPDATE_TP", snapshot=strategy.state)
        exit_command = make_command(1_001, "MARKET_EXIT_RUNNER", snapshot=strategy.state)
        await execution_queue.put(update_command)
        await execution_queue.put(exit_command)

        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=2)
        trader = FakeTrader()
        journal = FakeJournal()

        task = asyncio.create_task(
            execution_worker(
                execution_queue=execution_queue,
                state_lock=asyncio.Lock(),
                execution_state=execution_state,
                account_snapshot=AccountSnapshot(long_position(), 1000.0, 1000.0, 0.0, 0, 1),
                trader=trader,  # type: ignore[arg-type]
                strategy=strategy,
                journal=journal,  # type: ignore[arg-type]
                state_store=FakeStateStore(),  # type: ignore[arg-type]
                email_sender=FakeEmailSender(),  # type: ignore[arg-type]
                backlog_log_seconds=999,
            )
        )
        await asyncio.wait_for(execution_queue.join(), timeout=1)
        task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await task

        self.assertEqual(trader.executed_intents[0].intent_type, "MARKET_EXIT_RUNNER")
        reorder_events = [e for e in journal.events if e[0] == "TRADE_COMMAND_PRIORITY_REORDERED"]
        self.assertEqual(len(reorder_events), 1)

    async def test_market_exit_runner_allowed_during_full_halt(self) -> None:
        execution_state = ExecutionState(
            "pos-1", 100.0, trading_halted=True, halt_reason="execution_failure_live_position"
        )
        trader = FakeTrader()
        processor, _, _, _ = make_processor(execution_state=execution_state, trader=trader)

        result = await processor.process(make_command(1_000, "MARKET_EXIT_RUNNER"))

        self.assertIsNotNone(result)
        self.assertEqual(trader.executed_intents[0].intent_type, "MARKET_EXIT_RUNNER")

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

    # ── sidecar blocks NEAR_TP_REDUCE ────────────────────────────────────

    async def test_sidecar_blocks_near_tp_reduce(self) -> None:
        """NEAR_TP_REDUCE skipped and trading halted when sidecar enabled."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            total_entry_qty=1.0,
            sidecar_enabled_for_position=True,
        )
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

        command = make_command(1_000, "NEAR_TP_REDUCE")
        result = await processor.process(command)

        self.assertIsNone(result)
        self.assertEqual(trader.executed, [])
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(execution_state.halt_reason, "sidecar_blocks_near_tp_reduce")
        self.assertTrue(strategy.state.sidecar_dirty)
        self.assertEqual(strategy.state.sidecar_halt_reason, "sidecar_blocks_near_tp_reduce")
        self.assertGreaterEqual(len(state_store.saved), 1)
        sidecar_events = [
            e for e in journal.events if e[0] == "SIDECAR_BLOCKS_NEAR_TP_REDUCE"
        ]
        self.assertEqual(len(sidecar_events), 1)

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

    async def test_stale_update_tp_result_journals_and_preserves_existing_tp(self) -> None:
        strategy = three_stage_strategy("LONG")
        strategy.state.tp_order_id = "old-tp"
        strategy.state.tp_order_ids = ["old-tp"]
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()

        async def execute_stale(trade_intent: TradeIntent) -> LiveTradeResult:
            return LiveTradeResult(
                ok=True,
                action="UPDATE_TP",
                order_id=None,
                tp_order_id=None,
                contracts="0.71",
                tp_price="101.00",
                message="stale_tp_update_skipped_net_reduced",
                entry_filled=False,
                tp_ok=True,
            )

        trader.execute_intent = execute_stale  # type: ignore[method-assign]
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        intent = TradeIntent(
            **{
                **make_intent(1_000, "UPDATE_TP").__dict__,
                "managed_core_contracts": "1.41",
            }
        )

        result = await processor.process(
            TradeCommand(intent, copy.deepcopy(strategy.state), 1_000, 0.0, 0, "test")
        )

        self.assertIsNotNone(result)
        self.assertFalse(execution_state.trading_halted)
        self.assertEqual(strategy.state.tp_order_id, "old-tp")
        stale_events = [e for e in journal.events if e[0] == "STALE_TP_UPDATE_SKIPPED_NET_REDUCED"]
        self.assertEqual(len(stale_events), 1)

    async def test_update_tp_partial_success_persists_tp_id_before_sl_failure(self) -> None:
        strategy = three_stage_strategy("LONG")
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()

        async def execute_with_partial_success(trade_intent: TradeIntent) -> LiveTradeResult:
            callback = getattr(trader, "_on_tp_order_placed_after_place", None)
            self.assertIsNotNone(callback)
            await callback(
                intent=trade_intent,
                label="final",
                contracts=Decimal("1.00"),
                price=trade_intent.tp_price,
                order_id="tp-new",
                placed_order_ids=("tp-new",),
            )
            return LiveTradeResult(
                ok=False,
                action="UPDATE_TP",
                order_id=None,
                tp_order_id="tp-new",
                contracts="1",
                tp_price="110.00",
                message="trend_runner_protective_sl_failed: 51280",
                entry_filled=False,
                tp_ok=True,
                tp_order_ids=("tp-new",),
                protective_sl_ok=False,
            )

        trader.execute_intent = execute_with_partial_success  # type: ignore[method-assign]
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )

        result = await processor.process(make_command(1_000, "UPDATE_TP", snapshot=strategy.state))

        self.assertIsNotNone(result)
        self.assertEqual(strategy.state.tp_order_id, "tp-new")
        self.assertEqual(state_store.saved[-1].tp_order_id, "tp-new")
        persisted = [e for e in journal.events if e[0] == "TP_ORDER_ID_PERSISTED_AFTER_PLACE"]
        self.assertEqual(len(persisted), 1)

    async def test_update_tp_invalid_runner_sl_old_sl_active_journals_no_halt(self) -> None:
        strategy = three_stage_strategy("LONG")
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()

        async def execute_invalid_old_sl(trade_intent: TradeIntent) -> LiveTradeResult:
            return LiveTradeResult(
                ok=True,
                action="UPDATE_TP",
                order_id=None,
                tp_order_id="tp-new",
                contracts="1",
                tp_price="110.00",
                message="trend_runner_sl_update_skipped_invalid_but_old_sl_active",
                entry_filled=False,
                tp_ok=True,
                tp_order_ids=("tp-new",),
                protective_sl_order_id="old-sl",
                protective_sl_price="1670.00",
                protective_sl_ok=True,
            )

        trader.execute_intent = execute_invalid_old_sl  # type: ignore[method-assign]
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
        )
        intent = TradeIntent(
            **{
                **make_intent(1_000, "UPDATE_TP").__dict__,
                "trend_runner_active": True,
                "trend_runner_sl_price": 1679.22,
                "trend_runner_sl_order_id": "old-sl",
            }
        )

        result = await processor.process(
            TradeCommand(intent, copy.deepcopy(strategy.state), 1_000, 0.0, 0, "test")
        )

        self.assertIsNotNone(result)
        self.assertFalse(execution_state.trading_halted)
        events = [
            e for e in journal.events
            if e[0] == "TREND_RUNNER_SL_UPDATE_SKIPPED_INVALID_BUT_OLD_SL_ACTIVE"
        ]
        self.assertEqual(len(events), 1)

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

    # ── NEAR_TP_REDUCE result application ────────────────────────────────

    async def test_near_tp_reduce_protected_result(self) -> None:
        """NEAR_TP_REDUCE with protective_sl_ok applies protected state."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(near_tp_disable_add_after_reduce=True),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=3,
            total_entry_qty=3.0,
            total_entry_notional=300.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            tp_plan="SINGLE",
            near_tp_reduce_pending=True,
        )
        execution_state = ExecutionState("pos-1", 100.0)
        journal = FakeJournal()
        state_store = FakeStateStore()
        trader = FakeTrader()
        trader.set_position(long_position())

        near_tp_result = LiveTradeResult(
            ok=True,
            action="NEAR_TP_REDUCE",
            order_id="ord-near",
            tp_order_id="tp-near",
            contracts="7",
            tp_price="110",
            message="ok",
            protective_sl_ok=True,
            protective_sl_order_id="sl-near",
            protective_sl_price="105",
            contracts_before="10",
            contracts_reduced="3",
            contracts_after="7",
            near_tp_exit_all=False,
            tp_order_ids=("tp-near",),
        )
        trader.set_next_result(near_tp_result)

        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            journal=journal,
            state_store=state_store,
            trader=trader,
        )

        command = make_command(1_000, "NEAR_TP_REDUCE")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertTrue(strategy.state.near_tp_protected)
        self.assertFalse(strategy.state.near_tp_reduce_pending)
        self.assertTrue(strategy.state.near_tp_add_disabled)
        self.assertEqual(strategy.state.near_tp_protective_sl_order_id, "sl-near")
        self.assertEqual(strategy.state.tp_plan, "SINGLE")
        self.assertTrue(strategy.state.partial_tp_consumed)
        self.assertEqual(len(journal.near_tp_reduces), 1)
        self.assertGreaterEqual(len(state_store.saved), 1)

    async def test_near_tp_reduce_exit_all_arms_delayed_exit(self) -> None:
        """NEAR_TP_REDUCE with near_tp_exit_all arms delayed market exit (no immediate exit)."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=3,
            total_entry_qty=3.0,
            total_entry_notional=300.0,
            avg_entry_price=100.0,
            tp_price=110.0,
            near_tp_reduce_pending=True,
        )
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()

        near_tp_result = LiveTradeResult(
            ok=True,
            action="NEAR_TP_REDUCE",
            order_id="ord-near",
            tp_order_id="tp-near",
            contracts="7",
            tp_price="110",
            message="ok",
            protective_sl_ok=False,
            near_tp_exit_all=True,
            contracts_before="10",
            contracts_reduced="10",
            contracts_after="0",
        )
        trader.set_next_result(near_tp_result)

        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
        )

        command = make_command(1_000, "NEAR_TP_REDUCE")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertTrue(execution_state.trading_halted)
        # Delayed market exit armed, not market exit success
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))

    async def test_near_tp_market_exit_on_sl_fail(self) -> None:
        """NEAR_TP_REDUCE with protective_sl_ok=False arms delayed market exit email."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=3,
            total_entry_qty=3.0,
            near_tp_reduce_pending=True,
        )
        execution_state = ExecutionState("pos-1", 100.0)
        email_sender = FakeEmailSender()
        trader = FakeTrader()

        near_tp_result = LiveTradeResult(
            ok=True,
            action="NEAR_TP_REDUCE",
            order_id="ord-fail",
            tp_order_id="tp-fail",
            contracts="0",
            tp_price="110",
            message="market exited",
            protective_sl_ok=False,
            near_tp_exit_all=True,
            contracts_before="10",
            contracts_reduced="10",
            contracts_after="0",
        )
        trader.set_next_result(near_tp_result)

        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            email_sender=email_sender,
        )

        command = make_command(1_000, "NEAR_TP_REDUCE")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        # Delayed market exit should be armed
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))
        self.assertEqual(len(email_sender.sent), 1)
        self.assertIn("delayed market exit", email_sender.sent[0][0].lower())

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


class TestExecutionCommandProcessorWithSidecar(unittest.IsolatedAsyncioTestCase):
    """Sidecar-specific processor tests."""

    @staticmethod
    def _sidecar_trader() -> FakeTrader:
        trader = FakeTrader()
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        return trader

    async def test_sidecar_combined_entry_plan_no_sidecar_enabled(self) -> None:
        """When sidecar not enabled, only core entry executes."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=0,
            sidecar_enabled_for_position=False,
        )
        execution_state = ExecutionState(None, None)
        trader = self._sidecar_trader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
        )

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        self.assertIsNotNone(result)
        # Should have executed only core entry
        self.assertGreaterEqual(len(trader.executed), 1)

    # ── sidecar core exit safety tests ────────────────────────────────

    @staticmethod
    def _unsafe_update_tp_command(
        ts_ms: int = 5_000,
        core_tp_price: float = 105.0,
        side: str = "LONG",
    ) -> TradeCommand:
        intent = TradeIntent(
            intent_type="UPDATE_TP",  # type: ignore[arg-type]
            side=side,  # type: ignore[arg-type]
            price=104.0,
            layer_index=1,
            tp_price=core_tp_price,
            reason="test_unsafe_core_tp",
            size=PositionSize(1.0, 50.0, 1.0, 1, 1.0),
            fast_cvd=1.0,
            previous_fast_cvd=0.0,
            buy_ratio=1.0,
            sell_ratio=0.0,
            boll_upper=110.0,
            boll_middle=100.0,
            boll_lower=90.0,
            ts_ms=ts_ms,
            avg_entry_price=100.0,
            breakeven_price=100.0,
            tp_mode="MIDDLE",
        )
        return TradeCommand(
            intent,
            StrategyPositionState(side=side),
            ts_ms,
            asyncio.get_running_loop().time(),
            0,
            "test_unsafe_core_tp",
        )

    async def test_update_tp_unsafe_sidecar_realigns_before_execute(self) -> None:
        """Test 8: UPDATE_TP with unsafe core TP realigns sidecar TP before execute_intent."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-test", 1000.0)
        trader = self._sidecar_trader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )

        # core_tp=105, sidecar leg tp=106 → LONG sidecar tp beyond core → risky
        command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)

        result = await processor.process(command)

        self.assertIsNotNone(result)
        # Assert cancel called before place
        self.assertIn("old123", trader.cancelled_sidecar_tps)
        self.assertGreaterEqual(len(trader.sidecar_tps), 1)
        placed_tp = trader.sidecar_tps[0]
        self.assertEqual(placed_tp[2], 105.0)  # tp_price matches core
        # client_order_id must be SCE-prefixed (not f"{leg_id}-COREEXIT")
        client_order_id = placed_tp[3]
        self.assertIsNotNone(client_order_id)
        self.assertTrue(str(client_order_id).startswith("SCE"))
        # Assert execute_intent called after realignment
        self.assertGreaterEqual(len(trader.executed), 1)
        # Assert leg updated
        leg = strategy.state.sidecar_legs[0]
        self.assertEqual(leg["tp_price"], 105.0)
        self.assertEqual(leg["tp_order_id"], "sidecar-tp-1")
        self.assertTrue(leg.get("core_exit_aligned"))
        self.assertEqual(leg.get("core_exit_alignment_reason"), "sidecar_tp_beyond_core_final_exit")
        # Assert journal event
        realign_events = [e for e in journal.events if e[0] == "SIDECAR_TP_REALIGNED_TO_CORE_EXIT"]
        self.assertEqual(len(realign_events), 1)
        self.assertTrue(realign_events[0][1]["client_order_id"].startswith("SCE"))

    async def test_update_tp_safe_sidecar_does_nothing(self) -> None:
        """Test 9: UPDATE_TP with safe core TP does not touch sidecar TPs."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-safe", 1000.0)
        trader = self._sidecar_trader()
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
        )

        # core_tp=107, sidecar leg tp=106 → LONG sidecar tp before core → safe
        command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=107.0)

        result = await processor.process(command)

        self.assertIsNotNone(result)
        # No sidecar cancel/place calls
        self.assertEqual(len(trader.cancelled_sidecar_tps), 0)
        self.assertEqual(len(trader.sidecar_tps), 0)
        # execute_intent called normally
        self.assertGreaterEqual(len(trader.executed), 1)
        # No realignment journal
        realign_events = [e for e in journal.events if e[0] == "SIDECAR_TP_REALIGNED_TO_CORE_EXIT"]
        self.assertEqual(len(realign_events), 0)

    async def test_realign_failure_arms_delayed_exit_without_immediate_market_exit(self) -> None:
        """Test 10: realign failure arms delayed exit WITHOUT immediate market_exit call."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-cancel-fail", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False  # simulate cancel failure
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        # Default delay is 900s — we want the delay=900 behavior
        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "900"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        # Should return synthetic ok result (no RuntimeError)
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)  # type: ignore[union-attr]
        self.assertEqual(result.action, "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED")  # type: ignore[union-attr]

        # market exit NOT called immediately
        self.assertEqual(len(trader.market_exits), 0)
        # execute_intent NOT called
        self.assertEqual(len(trader.executed), 0)
        # trading halted with ARMED reason
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )
        # sidecar_dirty is True
        self.assertTrue(strategy.state.sidecar_dirty)
        # legs NOT yet marked FORCE_CLOSED (delayed task handles that)
        leg = strategy.state.sidecar_legs[0]
        self.assertEqual(leg["status"], "OPEN")
        # journal ARMED event
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)
        self.assertEqual(arm_events[0][1]["delay_seconds"], 900)
        # email sent
        arm_emails = [e for e in email_sender.sent if "delayed market exit armed" in e[0].lower()]
        self.assertGreaterEqual(len(arm_emails), 1)

    async def test_realign_failure_place_fails_arms_delayed_exit(self) -> None:
        """Test 11: place failure → arms delayed exit, no immediate market exit."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-place-fail", 1000.0)
        trader = self._sidecar_trader()
        trader._place_sidecar_tp_raises = RuntimeError("place_tp_failed")
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        # Use env to set delay
        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "900"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        # Should return synthetic ok result (no RuntimeError)
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)  # type: ignore[union-attr]
        self.assertEqual(result.action, "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED")  # type: ignore[union-attr]

        # cancel succeeded (before place failure)
        self.assertIn("old123", trader.cancelled_sidecar_tps)
        # market exit NOT called immediately
        self.assertEqual(len(trader.market_exits), 0)
        # execute_intent NOT called
        self.assertEqual(len(trader.executed), 0)
        # trading halted with ARMED reason
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )
        # journal ARMED event
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)

    async def test_realign_failure_arms_delayed_exit_no_background_task(self) -> None:
        """Alignment failure arms DME state.  Background task is deprecated;
        the account sync DME phase handles execution."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-delay0", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False  # simulate cancel failure
        trader.set_position(long_position())
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "0"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        # process returns synthetic ok result
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)  # type: ignore[union-attr]

        # NO immediate market exit
        self.assertEqual(len(trader.market_exits), 0)
        self.assertEqual(len(trader.executed), 0)

        # Journal ARMED event present
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)

        # DME state armed (no background task — DME phase handles execution)
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )
        all_event_names = [e[0] for e in journal.events]
        self.assertIn("SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED", all_event_names)

    async def test_delayed_exit_arms_unified_dme_state(self) -> None:
        """Alignment failure arms unified DME state (no background task)."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-flat", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False  # simulate cancel failure
        trader.set_position(flat_position())
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "0"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)  # type: ignore[union-attr]

        # NO immediate market exit
        self.assertEqual(len(trader.market_exits), 0)

        # DME state armed via unified arm_delayed_market_exit
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )

        # Journal ARMED event present
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)

    async def test_realign_failure_negative_delay_disables_auto_exit(self) -> None:
        """delay < 0: no background task scheduled, manual intervention required."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-neg", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False  # simulate cancel failure
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "-1"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)  # type: ignore[union-attr]

        # Drain event loop
        await asyncio.sleep(0)

        # market exit NOT called (disabled)
        self.assertEqual(len(trader.market_exits), 0)
        # ARMED event with manual_intervention_required=True
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)
        self.assertTrue(arm_events[0][1]["manual_intervention_required"])
        # Background task count should be 0 (no task was scheduled)
        self.assertEqual(len(processor._background_tasks), 0)

    async def test_update_tp_two_sidecar_legs_unique_client_order_ids(self) -> None:
        """Two sidecar legs with unsafe core TP get unique clOrdId values."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-ETHUSDTSWAPSHORT178079055281199a",
                    "tp_order_id": "3633692596170973184",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
                {
                    "leg_id": "leg-ETHUSDTSWAPSHORT178079055281199b",
                    "tp_order_id": "3633692596170974000",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-test", 1000.0)
        trader = self._sidecar_trader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )

        command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
        result = await processor.process(command)

        self.assertIsNotNone(result)
        # Two sidecar TPs placed
        self.assertEqual(len(trader.sidecar_tps), 2)
        # Client order IDs must be unique
        cid1 = trader.sidecar_tps[0][3]
        cid2 = trader.sidecar_tps[1][3]
        self.assertNotEqual(cid1, cid2)
        self.assertTrue(str(cid1).startswith("SCE"))
        self.assertTrue(str(cid2).startswith("SCE"))
        # execute_intent called after realignment
        self.assertGreaterEqual(len(trader.executed), 1)

    async def test_update_tp_sidecar_not_enabled_no_alignment(self) -> None:
        """UPDATE_TP with sidecar disabled does nothing even if unsafe."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=False,  # disabled
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
        )
        execution_state = ExecutionState("pos-disabled", 1000.0)
        trader = self._sidecar_trader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
        )

        command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
        result = await processor.process(command)

        self.assertIsNotNone(result)
        # No sidecar cancel/place
        self.assertEqual(len(trader.cancelled_sidecar_tps), 0)
        self.assertEqual(len(trader.sidecar_tps), 0)
        # execute_intent called normally
        self.assertGreaterEqual(len(trader.executed), 1)

    async def test_update_tp_no_open_sidecar_legs_no_alignment(self) -> None:
        """UPDATE_TP with no open sidecar legs does nothing."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-filled",
                    "status": "TP_FILLED",
                    "tp_price": 106.0,
                    "tp_order_id": "old456",
                    "contracts": "1",
                    "qty": 0.1,
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-no-open", 1000.0)
        trader = self._sidecar_trader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
        )

        command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
        result = await processor.process(command)

        self.assertIsNotNone(result)
        # No sidecar cancel/place
        self.assertEqual(len(trader.cancelled_sidecar_tps), 0)
        self.assertEqual(len(trader.sidecar_tps), 0)
        # execute_intent called normally
        self.assertGreaterEqual(len(trader.executed), 1)

    # ── delayed exit position fetch failure tests ──────────────────────

    async def test_delayed_exit_arms_unified_dme_fetch_failure(self) -> None:
        """Alignment failure with fetch failure → DME state armed (no background task)."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-fetch-fail", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False
        trader._fetch_position_raises = RuntimeError("fetch failed")
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "0"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        self.assertIsNotNone(result)
        self.assertTrue(result.ok)  # type: ignore[union-attr]

        # NO immediate market exit (DME phase handles it later)
        self.assertEqual(len(trader.market_exits), 0)

        # DME state armed
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )

        # ARM journal present
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)

    async def test_delayed_exit_arms_unified_dme_market_exit_fails(self) -> None:
        """Alignment failure → DME state armed.  Background task is deprecated."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-ff-fail", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False
        trader._fetch_position_raises = RuntimeError("fetch failed")
        trader._market_exit_returns = (False, "market failed")
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "0"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        self.assertIsNotNone(result)

        # NO immediate market exit
        self.assertEqual(len(trader.market_exits), 0)

        # DME state armed
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )
        # ARM journal present (background task deprecated — DME phase handles execution)
        all_event_names = [e[0] for e in journal.events]
        self.assertIn("SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED", all_event_names)
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)

    async def test_delayed_exit_arms_unified_dme_unhandled_exception(self) -> None:
        """Alignment failure → DME state armed (background task deprecated)."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        strategy.state = StrategyPositionState(
            side="LONG",
            layers=1,
            sidecar_enabled_for_position=True,
            sidecar_legs=[
                {
                    "leg_id": "leg-open-1",
                    "tp_order_id": "old123",
                    "tp_price": 106.0,
                    "contracts": "1",
                    "qty": 0.1,
                    "status": "OPEN",
                    "entry_price": 3000.0,
                    "side": "LONG",
                    "layer_index": 1,
                    "tp_pct": 0.004,
                    "margin_pct": 0.01,
                    "layer_multiplier": 1.0,
                    "position_id": "pos-1",
                    "created_ts_ms": 1000,
                    "updated_ts_ms": 1000,
                },
            ],
            breakeven_price=100.0,
        )
        execution_state = ExecutionState("pos-boom", 1000.0)
        trader = self._sidecar_trader()
        trader._cancel_sidecar_tp_returns = False
        trader.set_position(long_position())
        trader._market_exit_raises = RuntimeError("unhandled market exit crash")
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            strategy=strategy,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        with patch.dict("os.environ", {"SIDECAR_CORE_EXIT_ALIGNMENT_FAIL_AUTO_EXIT_DELAY_SECONDS": "0"}):
            command = self._unsafe_update_tp_command(ts_ms=5_000, core_tp_price=105.0)
            result = await processor.process(command)

        self.assertIsNotNone(result)

        # No background task — DME phase handles execution
        # ARM journal present
        arm_events = [e for e in journal.events if e[0] == "SIDECAR_CORE_EXIT_ALIGNMENT_DELAYED_MARKET_EXIT_ARMED"]
        self.assertEqual(len(arm_events), 1)

        # DME state armed with unified state
        self.assertTrue(getattr(strategy.state, "delayed_market_exit_armed", False))
        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(
            execution_state.halt_reason,
            "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        )


# ── G05: portfolio allocator shadow tests ────────────────────────────────────


class SlowShadowRunner:
    """Fake shadow runner that blocks on an asyncio.Event to test non-blocking."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[dict] = []

    async def run_entry_shadow_check(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.started.set()
        self.calls.append(kwargs)
        await self.release.wait()


class FailingShadowRunner:
    """Fake shadow runner that always raises."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_entry_shadow_check(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        raise RuntimeError("shadow boom")


class TestExecutionCommandProcessorShadow(unittest.IsolatedAsyncioTestCase):
    """G05 shadow mode tests for ExecutionCommandProcessor."""

    @staticmethod
    async def _drain_bg() -> None:
        """Yield enough times for background tasks to start."""
        for _ in range(5):
            await asyncio.sleep(0)

    async def test_slow_shadow_does_not_block_execution(self) -> None:
        """Slow shadow runner does NOT delay trader.execute_intent."""
        slow_runner = SlowShadowRunner()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_shadow_runner = slow_runner  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")

        # Process command — this should NOT be blocked by the slow shadow
        result = await processor.process(command)

        # trader.execute_intent should have been called (order executed)
        self.assertGreaterEqual(len(trader.executed), 1)
        self.assertIn(1_000, trader.executed)

        # Let background tasks run
        await self._drain_bg()

        # Shadow should have started (but not completed, waiting on release)
        self.assertTrue(slow_runner.started.is_set())
        self.assertEqual(len(slow_runner.calls), 1)

        # Cleanup: release the shadow runner so background task finishes
        slow_runner.release.set()
        await asyncio.sleep(0)

        self.assertIsNotNone(result)

    async def test_shadow_failure_does_not_block_execution(self) -> None:
        """Shadow runner exception does NOT prevent trader.execute_intent."""
        failing_runner = FailingShadowRunner()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_shadow_runner = failing_runner  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        # Let background tasks run (failure is caught by bg done callback)
        await self._drain_bg()

        # trader.execute_intent should STILL have been called
        self.assertGreaterEqual(len(trader.executed), 1)
        self.assertIn(1_000, trader.executed)

        # Shadow runner should have been called
        self.assertEqual(len(failing_runner.calls), 1)

        # Trading should NOT be halted
        self.assertFalse(execution_state.trading_halted)

        self.assertIsNotNone(result)

    async def test_shadow_runner_none_skips_cleanly(self) -> None:
        """When shadow_runner is None, everything works normally."""
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        # shadow_runner defaults to None in make_processor
        self.assertIsNone(processor.portfolio_allocator_shadow_runner)

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        self.assertGreaterEqual(len(trader.executed), 1)
        self.assertIsNotNone(result)

    async def test_non_entry_intents_do_not_schedule_shadow(self) -> None:
        """UPDATE_TP, NEAR_TP_REDUCE, MARKET_EXIT_RUNNER do NOT schedule shadow."""
        slow_runner = SlowShadowRunner()
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()

        for intent_type in ("UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"):
            processor, _, _, _ = make_processor(
                execution_state=execution_state,
                trader=trader,
                journal=journal,
                state_store=state_store,
            )
            processor.portfolio_allocator_shadow_runner = slow_runner  # type: ignore[assignment]

            command = make_command(1_000, intent_type)
            result = await processor.process(command)

            # Let background tasks run
            await self._drain_bg()

            # Shadow runner should NOT have been called
            self.assertEqual(len(slow_runner.calls), 0)
            # But trader should have executed
            self.assertGreaterEqual(len(trader.executed), 1)

            # Reset for next iteration
            slow_runner.calls.clear()
            slow_runner.started.clear()
            slow_runner.release.clear()
            trader.executed.clear()

    async def test_shadow_scheduled_before_execute_intent(self) -> None:
        """Shadow task is created before trader.execute_intent is called."""
        slow_runner = SlowShadowRunner()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()

        original_execute = trader.execute_intent
        execute_called = asyncio.Event()

        async def _wrapped_execute(intent: TradeIntent) -> LiveTradeResult:
            execute_called.set()
            return await original_execute(intent)

        trader.execute_intent = _wrapped_execute  # type: ignore[assignment]

        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
        )
        processor.portfolio_allocator_shadow_runner = slow_runner  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        # Let background tasks run
        await self._drain_bg()

        # Both should have happened
        self.assertTrue(slow_runner.started.is_set())
        self.assertTrue(execute_called.is_set())
        self.assertIsNotNone(result)

        # Cleanup
        slow_runner.release.set()
        await asyncio.sleep(0)


# ── G06a: portfolio allocator enforce tests ──────────────────────────────────


class AllowingEnforcer:
    """Fake enforcer that always allows."""

    def __init__(self) -> None:
        self.precheck_calls: list[dict] = []
        self.commit_calls: list[dict] = []
        self.committed = False

    async def precheck_entry_allocation(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        self.precheck_calls.append(kwargs)
        from src.live.portfolio_allocator_enforcer import PortfolioAllocatorPrecheckResult
        from src.portfolio.capital_ledger import default_snapshot
        return PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=True,
            reason="ALLOCATOR_ENFORCE_ALLOWED",
            projected_snapshot=default_snapshot(updated_ms=1000),
        )

    async def commit_projected_snapshot_after_fill(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.commit_calls.append(kwargs)
        self.committed = True


class RejectingEnforcer:
    """Fake enforcer that always rejects."""

    def __init__(self, reason: str = "GLOBAL_NO_NEW_ENTRY") -> None:
        self.reason = reason
        self.precheck_calls: list[dict] = []

    async def precheck_entry_allocation(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        self.precheck_calls.append(kwargs)
        from src.live.portfolio_allocator_enforcer import PortfolioAllocatorPrecheckResult
        return PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=False,
            reason=self.reason,
        )

    async def commit_projected_snapshot_after_fill(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class ErrorEnforcer:
    """Fake enforcer whose precheck returns an error result (fail-closed)."""

    def __init__(self) -> None:
        self.precheck_calls: list[dict] = []

    async def precheck_entry_allocation(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        self.precheck_calls.append(kwargs)
        from src.live.portfolio_allocator_enforcer import PortfolioAllocatorPrecheckResult
        return PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=False,
            reason="ALLOCATOR_ENFORCE_ERROR",
        )

    async def commit_projected_snapshot_after_fill(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class TestExecutionCommandProcessorEnforce(unittest.IsolatedAsyncioTestCase):
    """G06a enforce mode tests for ExecutionCommandProcessor."""

    async def test_enforce_none_keeps_old_behavior(self) -> None:
        """15. processor without enforcer: OPEN_LONG still executes."""
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        # enforcer defaults to None
        assert processor.portfolio_allocator_enforcer is None

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        assert result is not None
        assert len(trader.executed) >= 1

    async def test_enforce_allowed_executes_order(self) -> None:
        """16. enforce allowed: trader.execute_intent called, commit called after."""
        allowing = AllowingEnforcer()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        email_sender = FakeEmailSender()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )
        processor.portfolio_allocator_enforcer = allowing  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        assert result is not None
        assert len(trader.executed) >= 1
        # Commit should have been called
        assert allowing.committed is True
        assert len(allowing.precheck_calls) == 1
        assert allowing.commit_calls[0]["email_sender"] is email_sender

    async def test_enforce_rejected_skips_order(self) -> None:
        """17. enforce rejected: returns None, trader.execute_intent not called, no halt."""
        rejecting = RejectingEnforcer(reason="GLOBAL_NO_NEW_ENTRY")
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_enforcer = rejecting  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        assert result is None
        assert len(trader.executed) == 0
        assert execution_state.trading_halted is False
        assert len(rejecting.precheck_calls) == 1

    async def test_enforce_error_fail_closed_no_crash(self) -> None:
        """18. enforce error: returns None, no halt, no exception."""
        error_enforcer = ErrorEnforcer()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_enforcer = error_enforcer  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        # Must not raise
        result = await processor.process(command)

        assert result is None
        assert len(trader.executed) == 0
        assert execution_state.trading_halted is False

    # ── G06b: position id rollback on enforce reject ──────────────────────

    async def test_open_long_enforce_rejected_rolls_back_new_position_id(self) -> None:
        """OPEN_LONG enforce rejected rolls back newly created position_id and cash_before_position."""
        rejecting = RejectingEnforcer(reason="GLOBAL_NO_NEW_ENTRY")
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_enforcer = rejecting  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        assert result is None
        assert len(trader.executed) == 0
        assert execution_state.trading_halted is False
        assert execution_state.current_position_id is None
        assert execution_state.cash_before_position is None

    async def test_add_long_enforce_rejected_keeps_existing_position_id(self) -> None:
        """ADD_LONG enforce rejected keeps existing position_id and cash_before_position."""
        rejecting = RejectingEnforcer(reason="GLOBAL_NO_NEW_ENTRY")
        execution_state = ExecutionState("pos-existing", 123.45)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_enforcer = rejecting  # type: ignore[assignment]

        command = make_command(1_000, "ADD_LONG")
        result = await processor.process(command)

        assert result is None
        assert len(trader.executed) == 0
        assert execution_state.current_position_id == "pos-existing"
        assert execution_state.cash_before_position == 123.45

    async def test_add_long_aligns_execution_intent_to_ledger_plan_before_enforce(self) -> None:
        """ADD_LONG final execution intent is aligned to planned layer contracts."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        from src.portfolio.capital_allocator import AllocationDecision

        execution_state = ExecutionState("pos-existing", 123.45)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        enforcer = PortfolioAllocatorEnforcer.from_config(
            PortfolioAllocatorEnforceConfig(enabled=True),
        )
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = add_layer_snapshot(
            ("1", "1.15", "1.30"),
        )
        processor.portfolio_allocator_enforcer = enforcer

        captured: dict[str, object] = {}

        def fake_check(*, snapshot, request, leader_follower_config=None):
            captured["request"] = request
            return AllocationDecision(
                allowed=True,
                reason="ADD_MAIN_ALLOWED",
                inst_id=request.inst_id,
                action=request.action,
                requested_layer=request.requested_layer,
                leader_symbol=None,
                permission=None,
                projected_snapshot=snapshot,
            )

        raw_intent = make_intent(1_000, "ADD_LONG")
        add_intent = TradeIntent(
            **{
                **raw_intent.__dict__,
                "layer_index": 2,
                "size": PositionSize(2.4, 120.0, 0.12, 2, 1.15),
            }
        )
        command = TradeCommand(
            add_intent,
            StrategyPositionState(side="LONG", layers=1),
            1_000,
            0.0,
            0,
            "test",
        )

        with patch(
            "src.live.portfolio_allocator_enforcer.check_allocation_dry_run",
            side_effect=fake_check,
        ):
            result = await processor.process(command)

        assert result is not None
        request = captured["request"]
        assert request.requested_main_contracts == "1.15"  # type: ignore[attr-defined]
        assert trader.executed_intents[-1].size.eth_qty == 0.115

    async def test_add_long_missing_expected_contracts_fail_closed(self) -> None:
        """Missing planned layer keeps original ADD intent and allocator rejects it."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )

        execution_state = ExecutionState("pos-existing", 123.45)
        trader = FakeTrader()
        journal = FakeJournal()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
        )
        enforcer = PortfolioAllocatorEnforcer.from_config(
            PortfolioAllocatorEnforceConfig(enabled=True),
        )
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = add_layer_snapshot(("1",))
        processor.portfolio_allocator_enforcer = enforcer

        raw_intent = make_intent(1_000, "ADD_LONG")
        add_intent = TradeIntent(
            **{
                **raw_intent.__dict__,
                "layer_index": 2,
                "size": PositionSize(2.4, 120.0, 0.12, 2, 1.15),
            }
        )
        command = TradeCommand(
            add_intent,
            StrategyPositionState(side="LONG", layers=1),
            1_000,
            0.0,
            0,
            "test",
        )

        result = await processor.process(command)

        assert result is None
        assert len(trader.executed_intents) == 0
        rejected_events = [
            event
            for event in journal.events
            if event[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED"
        ]
        assert rejected_events[-1][1]["reason"] == "MISSING_EXPECTED_MAIN_CONTRACTS"

    async def test_open_long_does_not_run_add_alignment(self) -> None:
        """OPEN_LONG still executes the combined execution intent unchanged."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        from src.portfolio.capital_allocator import AllocationDecision

        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
        )
        enforcer = PortfolioAllocatorEnforcer.from_config(
            PortfolioAllocatorEnforceConfig(enabled=True),
        )
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = default_snapshot(updated_ms=1000)
        processor.portfolio_allocator_enforcer = enforcer

        def fake_check(*, snapshot, request, leader_follower_config=None):
            return AllocationDecision(
                allowed=True,
                reason="OPEN_MAIN_ALLOWED",
                inst_id=request.inst_id,
                action=request.action,
                requested_layer=request.requested_layer,
                leader_symbol=None,
                permission=None,
                projected_snapshot=snapshot,
            )

        raw_intent = make_intent(1_000, "OPEN_LONG")
        open_intent = TradeIntent(
            **{
                **raw_intent.__dict__,
                "size": PositionSize(2.0, 100.0, 1.0, 1, 1.0),
            }
        )
        command = TradeCommand(
            open_intent,
            StrategyPositionState(side="LONG"),
            1_000,
            0.0,
            0,
            "test",
        )

        with patch(
            "src.live.portfolio_allocator_enforcer.check_allocation_dry_run",
            side_effect=fake_check,
        ):
            result = await processor.process(command)

        assert result is not None
        assert trader.executed_intents[-1].size.eth_qty == 1.0

    async def test_open_long_enforce_allowed_keeps_generated_position_id(self) -> None:
        """OPEN_LONG enforce allowed keeps generated position_id and cash_before_position."""
        allowing = AllowingEnforcer()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_enforcer = allowing  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        assert result is not None
        assert len(trader.executed) >= 1
        assert execution_state.current_position_id is not None
        assert execution_state.cash_before_position is not None

    async def test_exit_reduce_not_enforced(self) -> None:
        """19. UPDATE_TP, NEAR_TP_REDUCE, MARKET_EXIT_RUNNER: enforcer not called."""
        rejecting = RejectingEnforcer(reason="GLOBAL_NO_NEW_ENTRY")
        execution_state = ExecutionState("pos-1", 100.0)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()

        for intent_type in ("UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"):
            processor, _, _, _ = make_processor(
                execution_state=execution_state,
                trader=trader,
                journal=journal,
                state_store=state_store,
            )
            processor.portfolio_allocator_enforcer = rejecting  # type: ignore[assignment]

            command = make_command(1_000, intent_type)
            result = await processor.process(command)

            # Trader should have executed (not blocked)
            assert len(trader.executed) >= 1
            # Enforcer should NOT have been called for non-entry intents
            # (the precheck is in the entry block which is skipped)

            # Reset
            rejecting.precheck_calls.clear()
            trader.executed.clear()

    async def test_shadow_and_enforce_coexist(self) -> None:
        """20. shadow still scheduled fire-and-forget, enforce awaited, order executes."""
        from tests.test_execution_command_processor import SlowShadowRunner

        slow_runner = SlowShadowRunner()
        allowing = AllowingEnforcer()
        execution_state = ExecutionState(None, None)
        trader = FakeTrader()
        journal = FakeJournal()
        state_store = FakeStateStore()
        processor, _, _, _ = make_processor(
            execution_state=execution_state,
            trader=trader,
            journal=journal,
            state_store=state_store,
        )
        processor.portfolio_allocator_shadow_runner = slow_runner  # type: ignore[assignment]
        processor.portfolio_allocator_enforcer = allowing  # type: ignore[assignment]

        command = make_command(1_000, "OPEN_LONG")
        result = await processor.process(command)

        # Let background tasks run a bit
        for _ in range(5):
            await asyncio.sleep(0)

        # Order should have executed
        assert result is not None
        assert len(trader.executed) >= 1
        # Enforce precheck should have been called
        assert len(allowing.precheck_calls) == 1
        # Enforce commit should have been called
        assert allowing.committed is True
        # Shadow should have been scheduled (started but not completed)
        assert slow_runner.started.is_set()
        assert len(slow_runner.calls) == 1

        # Cleanup
        slow_runner.release.set()
        await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
