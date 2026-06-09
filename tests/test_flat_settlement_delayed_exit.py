"""Tests for flat settlement clearing DME failed halt reasons.

Verifies:
- order_failure_delayed_market_exit_failed is in flat_clearable_halt_reasons.
- near_tp_final_tp_failed_delayed_market_exit_armed is in flat_clearable_halt_reasons.
- When position is flat with DME FAILED halt, trading_halted is cleared.
- DME WAITING_FLAT / FAILED both allow flat to clear the halt.
- Existing halt reasons are preserved.
- Rolling loss halt logic is not broken.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest

from src.execution.trader import PositionSnapshot
from src.live import runtime_types as live_runtime_types
from src.live.account_sync.flat_settlement_phase import (
    finalize_account_sync_flat_settlement_phase,
    prepare_account_sync_flat_settlement_phase,
)
from src.live.account_sync import flat_balance as live_flat_balance
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy


# ── Helpers ─────────────────────────────────────────────────────────────

class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    account_equity_usdt = 1000.0
    position_contracts = Decimal("0")
    contract_multiplier = Decimal("0.1")

    async def fetch_position_snapshot(self):
        return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

    def mark_flat(self) -> None:
        pass


def _make_strategy() -> BollCvdShockReclaimStrategy:
    return BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


# ── Test: flat_clearable_halt_reasons includes DME failed ──────────────

class TestFlatClearableHaltReasonsDmeFailed:
    """Verify the flat_clearable_halt_reasons set contains DME failed/armed reasons."""

    async def _run_flat_settlement(
        self,
        strategy: BollCvdShockReclaimStrategy,
        execution_state: live_runtime_types.ExecutionState,
        previous_halt_reason: str,
    ) -> live_runtime_types.ExecutionState:
        """Run prepare_account_sync_flat_settlement_phase and return updated execution_state."""
        trader = FakeTrader()

        with mock.patch.object(
            live_flat_balance,
            "fetch_settled_flat_balance",
            return_value=live_runtime_types.SettledFlatBalance(
                cash=1000.0,
                equity=1000.0,
                attempts=1,
                stable=True,
                reason="test",
            ),
        ):
            result = await prepare_account_sync_flat_settlement_phase(
                state_lock=asyncio.Lock(),
                account_snapshot=live_runtime_types.AccountSnapshot(
                    position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                    cash=1000.0,
                    equity=1000.0,
                    updated_monotonic=0.0,
                    updated_ts_ms=0,
                    version=0,
                    latest_market_price=None,
                    latest_market_price_ts_ms=0,
                ),
                execution_state=execution_state,
                trader=trader,
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,
                rolling_loss_guard=None,
                pending_flat_payload={
                    "position_id": "pos-1",
                    "previous_halt_reason": previous_halt_reason,
                    "delayed_market_exit_was_armed": True,
                    "delayed_market_exit_status": "FAILED",
                    "delayed_market_exit_reason": "core_tp_place_failed",
                    "delayed_market_exit_executed_ts_ms": 0,
                    "delayed_market_exit_exit_attempt_count": 3,
                },
                position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                current_position_key=None,
                cash=1000.0,
                equity=1000.0,
                flat_balance_confirm_attempts=1,
                flat_balance_confirm_interval_seconds=0.5,
                flat_balance_stable_delta_usdt=10.0,
                flat_balance_cash_equity_max_diff_usdt=100.0,
                last_logged_cash=0.0,
                last_logged_equity=0.0,
                last_logged_position_key=None,
            )

        # After flat settlement, clear_state signals execution_state was updated
        if result.clear_state:
            execution_state.trading_halted = False
            execution_state.halt_reason = None
            execution_state.current_position_id = None

        return execution_state

    @pytest.mark.asyncio
    async def test_order_failure_dme_failed_clears_on_flat(self) -> None:
        """When halt_reason=order_failure_delayed_market_exit_failed and position flat,
        trading_halted should be cleared."""
        strategy = _make_strategy()
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "order_failure_delayed_market_exit_failed"

        updated_state = await self._run_flat_settlement(
            strategy, execution_state, "order_failure_delayed_market_exit_failed",
        )

        assert updated_state.trading_halted is False, (
            "DME failed halt must clear when position is flat"
        )
        assert updated_state.halt_reason is None

    @pytest.mark.asyncio
    async def test_near_tp_final_tp_failed_dme_armed_clears_on_flat(self) -> None:
        """When halt_reason=near_tp_final_tp_failed_delayed_market_exit_armed
        and position flat, trading_halted should be cleared."""
        strategy = _make_strategy()
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "near_tp_final_tp_failed_delayed_market_exit_armed"

        updated_state = await self._run_flat_settlement(
            strategy, execution_state, "near_tp_final_tp_failed_delayed_market_exit_armed",
        )

        assert updated_state.trading_halted is False, (
            "near_tp DME armed halt must clear when position is flat"
        )
        assert updated_state.halt_reason is None

    @pytest.mark.asyncio
    async def test_dme_waiting_flat_still_clears_on_flat(self) -> None:
        """Existing order_failure_delayed_market_exit_waiting_flat still clears."""
        strategy = _make_strategy()
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "order_failure_delayed_market_exit_waiting_flat"

        updated_state = await self._run_flat_settlement(
            strategy, execution_state, "order_failure_delayed_market_exit_waiting_flat",
        )

        assert updated_state.trading_halted is False


# ── Test: DME armed reasons preserved as clearable ─────────────────────

class TestDmeArmedReasonsStillClearable:
    """Verify existing DME armed reasons are still in the flat_clearable_halt_reasons set."""

    DME_ARMED_REASONS = [
        "sidecar_tp_place_failed_delayed_market_exit_armed",
        "sidecar_tp_place_rate_limited_delayed_market_exit_armed",
        "sidecar_core_exit_alignment_failed_delayed_market_exit_armed",
        "three_stage_post_tp1_sl_failed_delayed_market_exit_armed",
        "middle_runner_sl_failed_delayed_market_exit_armed",
        "middle_bucket_fast_sl_failed_delayed_market_exit_armed",
        "middle_bucket_fast_sl_invalid_delayed_market_exit_armed",
        "near_tp_protective_sl_failed_delayed_market_exit_armed",
        "core_tp_place_failed_delayed_market_exit_armed",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("halt_reason", DME_ARMED_REASONS)
    async def test_existing_armed_reasons_clear_on_flat(self, halt_reason: str) -> None:
        """Each existing DME armed reason must clear on flat."""
        strategy = BollCvdShockReclaimStrategy(
            BollCvdReclaimStrategyConfig(),
            SimplePositionSizer(SimplePositionSizerConfig()),
        )
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = halt_reason

        trader = FakeTrader()

        with mock.patch.object(
            live_flat_balance,
            "fetch_settled_flat_balance",
            return_value=live_runtime_types.SettledFlatBalance(
                cash=1000.0,
                equity=1000.0,
                attempts=1,
                stable=True,
                reason="test",
            ),
        ):
            result = await prepare_account_sync_flat_settlement_phase(
                state_lock=asyncio.Lock(),
                account_snapshot=live_runtime_types.AccountSnapshot(
                    position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                    cash=1000.0,
                    equity=1000.0,
                    updated_monotonic=0.0,
                    updated_ts_ms=0,
                    version=0,
                    latest_market_price=None,
                    latest_market_price_ts_ms=0,
                ),
                execution_state=execution_state,
                trader=trader,
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,
                rolling_loss_guard=None,
                pending_flat_payload={
                    "position_id": "pos-1",
                    "previous_halt_reason": halt_reason,
                    "delayed_market_exit_was_armed": True,
                    "delayed_market_exit_status": "ARMED",
                    "delayed_market_exit_reason": halt_reason,
                    "delayed_market_exit_executed_ts_ms": 0,
                    "delayed_market_exit_exit_attempt_count": 0,
                },
                position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                current_position_key=None,
                cash=1000.0,
                equity=1000.0,
                flat_balance_confirm_attempts=1,
                flat_balance_confirm_interval_seconds=0.5,
                flat_balance_stable_delta_usdt=10.0,
                flat_balance_cash_equity_max_diff_usdt=100.0,
                last_logged_cash=0.0,
                last_logged_equity=0.0,
                last_logged_position_key=None,
            )

        # After flat settlement, clear_state signals execution_state was updated
        if result.clear_state:
            execution_state.trading_halted = False
            execution_state.halt_reason = None

        assert execution_state.trading_halted is False, (
            f"DME armed reason '{halt_reason}' must clear on flat"
        )


# ── Test: record_flat_payload includes DME fields ─────────────────────

class TestRecordFlatPayloadDmeFields:
    """record_flat_payload must include DME diagnostic fields."""

    @pytest.mark.asyncio
    async def test_dme_fields_in_flat_payload(self) -> None:
        strategy = _make_strategy()
        # Set DME state fields so they appear in record_flat_payload
        strategy.state.delayed_market_exit_armed = True
        strategy.state.delayed_market_exit_status = "FAILED"
        strategy.state.delayed_market_exit_reason = "core_tp_place_failed"
        strategy.state.delayed_market_exit_executed_ts_ms = 1700000000000
        strategy.state.delayed_market_exit_exit_attempt_count = 3

        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "order_failure_delayed_market_exit_failed"

        trader = FakeTrader()

        with mock.patch.object(
            live_flat_balance,
            "fetch_settled_flat_balance",
            return_value=live_runtime_types.SettledFlatBalance(
                cash=1000.0,
                equity=1000.0,
                attempts=1,
                stable=True,
                reason="test",
            ),
        ):
            result = await prepare_account_sync_flat_settlement_phase(
                state_lock=asyncio.Lock(),
                account_snapshot=live_runtime_types.AccountSnapshot(
                    position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                    cash=1000.0,
                    equity=1000.0,
                    updated_monotonic=0.0,
                    updated_ts_ms=0,
                    version=0,
                    latest_market_price=None,
                    latest_market_price_ts_ms=0,
                ),
                execution_state=execution_state,
                trader=trader,
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,
                rolling_loss_guard=None,
                pending_flat_payload={
                    "position_id": "pos-1",
                    "previous_halt_reason": "order_failure_delayed_market_exit_failed",
                },
                position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                current_position_key=None,
                cash=1000.0,
                equity=1000.0,
                flat_balance_confirm_attempts=1,
                flat_balance_confirm_interval_seconds=0.5,
                flat_balance_stable_delta_usdt=10.0,
                flat_balance_cash_equity_max_diff_usdt=100.0,
                last_logged_cash=0.0,
                last_logged_equity=0.0,
                last_logged_position_key=None,
            )

        # Verify the flat payload is populated
        assert result.record_flat_payload is not None
        payload = result.record_flat_payload

        # DME diagnostic fields must be present and sourced from strategy.state
        assert payload.get("delayed_market_exit_was_armed") is True, \
            "delayed_market_exit_was_armed must be recorded"
        assert payload.get("delayed_market_exit_status") == "FAILED"
        assert payload.get("delayed_market_exit_reason") == "core_tp_place_failed"
        assert payload.get("delayed_market_exit_executed_ts_ms") == 1700000000000
        assert payload.get("delayed_market_exit_exit_attempt_count") == 3
        assert payload.get("delayed_market_exit_cleared") is True


# ── Test: rolling_loss_guard non-empty DME flat settlement clear halt ────

class FakeRollingLossGuard:
    """Minimal fake to satisfy rolling_loss_guard is not None check.

    The production code only checks ``rolling_loss_guard is not None``,
    so a bare object instance suffices — no methods or attributes required.
    """
    pass


class TestDmeFailedClearsOnFlatWithRollingLossGuard:
    """When rolling_loss_guard is present, clearable DME halt reasons
    must still clear on flat.
    """

    async def _call_flat_settlement(
        self,
        execution_state: live_runtime_types.ExecutionState,
        halt_reason: str,
        *,
        rolling_loss_guard: object | None,
    ) -> live_runtime_types.ExecutionState:
        """Directly call prepare_account_sync_flat_settlement_phase
        WITHOUT manually clearing execution_state afterward.

        The production code is responsible for setting
        execution_state.trading_halted / halt_reason inside the function.
        """
        strategy = _make_strategy()
        trader = FakeTrader()

        with mock.patch.object(
            live_flat_balance,
            "fetch_settled_flat_balance",
            return_value=live_runtime_types.SettledFlatBalance(
                cash=1000.0,
                equity=1000.0,
                attempts=1,
                stable=True,
                reason="test",
            ),
        ):
            await prepare_account_sync_flat_settlement_phase(
                state_lock=asyncio.Lock(),
                account_snapshot=live_runtime_types.AccountSnapshot(
                    position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                    cash=1000.0,
                    equity=1000.0,
                    updated_monotonic=0.0,
                    updated_ts_ms=0,
                    version=0,
                    latest_market_price=None,
                    latest_market_price_ts_ms=0,
                ),
                execution_state=execution_state,
                trader=trader,
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy,
                rolling_loss_guard=rolling_loss_guard,
                pending_flat_payload={
                    "position_id": "pos-1",
                    "previous_halt_reason": halt_reason,
                    "delayed_market_exit_was_armed": True,
                    "delayed_market_exit_status": "FAILED",
                    "delayed_market_exit_reason": "core_tp_place_failed",
                    "delayed_market_exit_executed_ts_ms": 0,
                    "delayed_market_exit_exit_attempt_count": 3,
                },
                position=PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
                current_position_key=None,
                cash=1000.0,
                equity=1000.0,
                flat_balance_confirm_attempts=1,
                flat_balance_confirm_interval_seconds=0.5,
                flat_balance_stable_delta_usdt=10.0,
                flat_balance_cash_equity_max_diff_usdt=100.0,
                last_logged_cash=0.0,
                last_logged_equity=0.0,
                last_logged_position_key=None,
            )

        # Do NOT manually clear execution_state — the production code must do it.
        return execution_state

    @pytest.mark.asyncio
    async def test_order_failure_dme_failed_clears_on_flat_with_rolling_loss_guard(self) -> None:
        """When rolling_loss_guard is not None and
        halt_reason=order_failure_delayed_market_exit_failed, flat must clear halt.
        """
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "order_failure_delayed_market_exit_failed"

        updated_state = await self._call_flat_settlement(
            execution_state,
            "order_failure_delayed_market_exit_failed",
            rolling_loss_guard=FakeRollingLossGuard(),
        )

        assert updated_state.trading_halted is False, (
            "DME failed halt must clear on flat even with rolling_loss_guard present"
        )
        assert updated_state.halt_reason is None

    @pytest.mark.asyncio
    async def test_near_tp_final_tp_failed_dme_armed_clears_on_flat_with_rolling_loss_guard(self) -> None:
        """When rolling_loss_guard is not None and
        halt_reason=near_tp_final_tp_failed_delayed_market_exit_armed, flat must clear halt.
        """
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "near_tp_final_tp_failed_delayed_market_exit_armed"

        updated_state = await self._call_flat_settlement(
            execution_state,
            "near_tp_final_tp_failed_delayed_market_exit_armed",
            rolling_loss_guard=FakeRollingLossGuard(),
        )

        assert updated_state.trading_halted is False, (
            "near_tp DME armed halt must clear on flat even with rolling_loss_guard present"
        )
        assert updated_state.halt_reason is None

    @pytest.mark.asyncio
    async def test_rolling_loss_guard_preserves_non_clearable_halt(self) -> None:
        """When rolling_loss_guard is not None and halt_reason is NOT in
        flat_clearable_halt_reasons, the halt must be preserved (not cleared).
        """
        execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
        execution_state.trading_halted = True
        execution_state.halt_reason = "some_critical_non_clearable_halt"

        updated_state = await self._call_flat_settlement(
            execution_state,
            "some_critical_non_clearable_halt",
            rolling_loss_guard=FakeRollingLossGuard(),
        )

        assert updated_state.trading_halted is True, (
            "Non-clearable halt must be preserved when rolling_loss_guard is present"
        )
        assert updated_state.halt_reason == "some_critical_non_clearable_halt"


# ── Test: finalize flat settlement with DME fields does not raise ───────

class TestFinalizeFlatSettlementRecordFlatPayloadWithDmeDoesNotRaise:
    """Integration test: finalize_account_sync_flat_settlement_phase with
    delayed_market_exit_* fields in record_flat_payload must not raise
    TypeError and must complete state_store.clear().
    """

    @pytest.mark.asyncio
    async def test_finalize_with_dme_fields_does_not_raise(self, tmp_path: Path) -> None:
        from src.reporting.live_state_store import LiveStateStore
        from src.reporting.trade_journal import LiveTradeJournal

        journal = LiveTradeJournal(
            path=tmp_path / "events.jsonl",
            summary_path=tmp_path / "summary.jsonl",
        )
        state_file = tmp_path / "live_state.json"
        state_file.write_text('{"position_id": "pos-1"}')
        state_store = LiveStateStore(path=state_file)

        record_flat_payload: dict[str, object] = {
            "position_id": "pos-finalize-1",
            "symbol": "ETH-USDT-SWAP",
            "side": "LONG",
            "cash_before_position": 1000.0,
            "cash_after": 1050.0,
            "equity_after": 1050.0,
            "reason": "tp_hit",
            "layers": 1,
            "avg_entry_price": 3000.0,
            "last_tp_price": 3100.0,
            "last_tp_plan": "SINGLE",
            "partial_tp_consumed": False,
            "trend_runner_exit_reason": None,
            # DME extra fields that caused TypeError before fix
            "delayed_market_exit_was_armed": True,
            "delayed_market_exit_reason": "core_tp_place_failed",
            "delayed_market_exit_status": "FAILED",
            "delayed_market_exit_executed_ts_ms": 123,
            "delayed_market_exit_exit_attempt_count": 2,
            "delayed_market_exit_cleared": True,
        }

        execution_state = live_runtime_types.ExecutionState("pos-finalize-1", 1000.0)

        # This must not raise TypeError
        await finalize_account_sync_flat_settlement_phase(
            state_lock=asyncio.Lock(),
            execution_state=execution_state,
            journal=journal,
            email_sender=None,
            state_store=state_store,
            rolling_loss_guard=None,
            record_flat_payload=record_flat_payload,
            pending_flat_payload=None,
            flat_previous_halt_reason=None,
            clear_state=True,
        )

        # state_store.clear() must have been called — file no longer exists
        assert not state_file.exists(), (
            "state_store.clear() must execute after record_flat, file should be deleted"
        )

        # Journal must contain FLAT event with DME fields
        events = journal.load_events()
        assert len(events) == 1
        assert events[0].event_type == "FLAT"
        assert events[0].position_id == "pos-finalize-1"

        payload = events[0].payload
        assert payload["delayed_market_exit_was_armed"] is True
        assert payload["delayed_market_exit_reason"] == "core_tp_place_failed"
        assert payload["delayed_market_exit_status"] == "FAILED"
        assert payload["delayed_market_exit_executed_ts_ms"] == 123
        assert payload["delayed_market_exit_exit_attempt_count"] == 2
        assert payload["delayed_market_exit_cleared"] is True


if __name__ == "__main__":
    pytest.main([__file__])
