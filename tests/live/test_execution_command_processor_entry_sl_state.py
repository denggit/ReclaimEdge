"""Tests for _maybe_apply_entry_protective_sl_state boundary.

Only entry intents that carry ``entry_protective_sl_price`` should write
to ``StrategyPositionState.entry_protective_sl_*``.  Non-entry intents
(UPDATE_TP, runner SL) must NOT pollute entry SL state.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.live.runtime_types import TradeCommand
from src.live.workers.execution_command_processor import ExecutionCommandProcessor
from src.execution.trader import LiveTradeResult


# ── helpers ──────────────────────────────────────────────────────────────

_FAKE_SIZE = PositionSize(1.0, 50.0, 1.0, 1, 1.0)


def _entry_command_with_sl_price(sl_price: float | None) -> TradeCommand:
    """Build an OPEN_LONG command that carries entry_protective_sl_price."""
    intent = TradeIntent(
        intent_type="OPEN_LONG",  # type: ignore[arg-type]
        side="LONG",  # type: ignore[arg-type]
        price=100.0,
        layer_index=1,
        tp_price=101.0,
        reason="test",
        size=_FAKE_SIZE,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        boll_upper=101.0,
        boll_middle=100.0,
        boll_lower=99.0,
        ts_ms=1000,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
    )
    if sl_price is not None:
        intent = TradeIntent(**{**intent.__dict__, "entry_protective_sl_price": sl_price})
    return TradeCommand(
        intent,
        StrategyPositionState(side="LONG"),
        1000,
        0.0,
        0,
        "test",
    )


def _update_tp_command_without_entry_sl() -> TradeCommand:
    """Build an UPDATE_TP command (no entry_protective_sl_price)."""
    intent = TradeIntent(
        intent_type="UPDATE_TP",  # type: ignore[arg-type]
        side="LONG",  # type: ignore[arg-type]
        price=100.0,
        layer_index=1,
        tp_price=101.0,
        reason="test",
        size=_FAKE_SIZE,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        boll_upper=101.0,
        boll_middle=100.0,
        boll_lower=99.0,
        ts_ms=1000,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
    )
    return TradeCommand(
        intent,
        StrategyPositionState(side="LONG"),
        1000,
        0.0,
        0,
        "test",
    )


def _result_with_protective_sl(order_id: str, ok: bool = True) -> LiveTradeResult:
    """Build a LiveTradeResult carrying a protective SL."""
    return LiveTradeResult(
        ok=True,
        action="EXECUTED",
        order_id="ord-1",
        tp_order_id="tp-1",
        contracts="10",
        tp_price="101",
        message="ok",
        protective_sl_ok=ok,
        protective_sl_order_id=order_id,
        protective_sl_price="99.0",
    )


def _result_without_protective_sl() -> LiveTradeResult:
    """Build a LiveTradeResult without any protective SL fields."""
    return LiveTradeResult(
        ok=True,
        action="EXECUTED",
        order_id="ord-1",
        tp_order_id="tp-1",
        contracts="10",
        tp_price="101",
        message="ok",
    )


# ── minimum fake processor ──────────────────────────────────────────────


def _make_minimum_processor() -> ExecutionCommandProcessor:
    """Build a minimum ExecutionCommandProcessor for testing the helper.

    Only the strategy and state_lock are required; all other fields
    can be dummy/MagicMock since the helper never touches them.
    """
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    strategy.state = StrategyPositionState(side="LONG")

    from src.live.runtime_types import AccountSnapshot, ExecutionState
    from src.execution.trader import PositionSnapshot

    return ExecutionCommandProcessor(
        state_lock=MagicMock(),  # not actually acquired by the helper
        execution_state=ExecutionState(None, None),
        account_snapshot=AccountSnapshot(
            PositionSnapshot(None, None, 0.0, 0.0, None), 1000.0, 1000.0, 0.0, 0, 1
        ),
        trader=MagicMock(),
        strategy=strategy,
        journal=MagicMock(),
        state_store=MagicMock(),
        email_sender=MagicMock(),
    )


# ── tests ────────────────────────────────────────────────────────────────


class TestMaybeApplyEntryProtectiveSlState(unittest.TestCase):
    """Test _maybe_apply_entry_protective_sl_state boundary behaviour."""

    def test_case_a_entry_intent_with_sl_price_writes_state(self) -> None:
        """Case A: entry intent has entry_protective_sl_price, result has protective_sl_order_id → writes state."""
        processor = _make_minimum_processor()
        strategy = processor.strategy

        command = _entry_command_with_sl_price(sl_price=99.0)
        result = _result_with_protective_sl(order_id="entry-sl-001", ok=True)

        returned = processor._maybe_apply_entry_protective_sl_state(command, result)

        self.assertTrue(returned)
        self.assertEqual(strategy.state.entry_protective_sl_order_id, "entry-sl-001")
        self.assertEqual(strategy.state.entry_protective_sl_price, 99.0)
        self.assertTrue(strategy.state.entry_protective_sl_protected)

    def test_case_b_non_entry_intent_without_sl_price_does_not_write(self) -> None:
        """Case B: non-entry intent (UPDATE_TP) has no entry_protective_sl_price → does NOT write entry SL state.

        The result still carries a protective_sl_order_id (it is a runner protective SL),
        but the helper must reject it because the intent is not an entry intent.
        """
        processor = _make_minimum_processor()
        strategy = processor.strategy

        # Pre-populate existing entry protective SL state to verify it is NOT overwritten
        strategy.state.entry_protective_sl_order_id = "existing-entry-sl"
        strategy.state.entry_protective_sl_price = 98.0
        strategy.state.entry_protective_sl_protected = True

        command = _update_tp_command_without_entry_sl()
        result = _result_with_protective_sl(order_id="runner-sl-002", ok=True)

        returned = processor._maybe_apply_entry_protective_sl_state(command, result)

        self.assertFalse(returned)
        # Existing entry SL state must NOT be overwritten
        self.assertEqual(strategy.state.entry_protective_sl_order_id, "existing-entry-sl")
        self.assertEqual(strategy.state.entry_protective_sl_price, 98.0)
        self.assertTrue(strategy.state.entry_protective_sl_protected)

    def test_case_c_entry_intent_but_result_has_no_sl_order_id_returns_false(self) -> None:
        """Case C: intent has entry_protective_sl_price but result has no protective_sl_order_id → return False."""
        processor = _make_minimum_processor()
        strategy = processor.strategy

        command = _entry_command_with_sl_price(sl_price=98.5)
        result = _result_without_protective_sl()

        returned = processor._maybe_apply_entry_protective_sl_state(command, result)

        self.assertFalse(returned)
        # No state should have been written
        self.assertIsNone(strategy.state.entry_protective_sl_order_id)
        self.assertIsNone(strategy.state.entry_protective_sl_price)
        self.assertFalse(strategy.state.entry_protective_sl_protected)

    def test_entry_intent_with_none_sl_price_returns_false(self) -> None:
        """Intent has entry_protective_sl_price=None → returns False even with valid result."""
        processor = _make_minimum_processor()
        strategy = processor.strategy

        command = _entry_command_with_sl_price(sl_price=None)
        result = _result_with_protective_sl(order_id="should-not-write")

        returned = processor._maybe_apply_entry_protective_sl_state(command, result)

        self.assertFalse(returned)
        self.assertIsNone(strategy.state.entry_protective_sl_order_id)

    def test_protective_sl_ok_false_still_writes_when_order_id_present(self) -> None:
        """When order_id present but protective_sl_ok=False, state is still written (protected=False)."""
        processor = _make_minimum_processor()
        strategy = processor.strategy

        command = _entry_command_with_sl_price(sl_price=99.0)
        result = _result_with_protective_sl(order_id="entry-sl-fail", ok=False)

        returned = processor._maybe_apply_entry_protective_sl_state(command, result)

        self.assertTrue(returned)
        self.assertEqual(strategy.state.entry_protective_sl_order_id, "entry-sl-fail")
        self.assertFalse(strategy.state.entry_protective_sl_protected)


if __name__ == "__main__":
    unittest.main()
