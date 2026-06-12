from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.live import runtime_types as live_runtime_types
from src.live.account_sync.protective_orders_phase import run_account_sync_protective_orders_phase


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def append(self, event: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event, dict(payload), position_id))


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:
        self.saved.append(state)


class FakeTrader:
    symbol = "ETH-USDT-SWAP"

    def __init__(self) -> None:
        self.three_stage_result = (True, "new-post-sl", "ok")
        self.middle_runner_result = (True, "new-middle-sl", "ok")
        self.fast_result = (True, "new-fast-sl", "ok")
        self.calls: list[tuple[str, str | None]] = []

    async def place_three_stage_post_tp1_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds
    ):
        self.calls.append(("place_three_stage_post_tp1_protective_stop_with_retries", None))
        return self.three_stage_result

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str) -> bool:
        self.calls.append(("cancel_three_stage_post_tp1_protective_stop", order_id))
        return True

    async def place_middle_runner_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds
    ):
        self.calls.append(("place_middle_runner_protective_stop_with_retries", None))
        return self.middle_runner_result

    async def cancel_middle_runner_protective_stop(self, order_id: str) -> bool:
        self.calls.append(("cancel_middle_runner_protective_stop", order_id))
        return True

    async def place_middle_bucket_fast_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds
    ):
        self.calls.append(("place_middle_bucket_fast_protective_stop_with_retries", None))
        return self.fast_result

    async def cancel_middle_bucket_fast_protective_stop(self, order_id: str) -> bool:
        self.calls.append(("cancel_middle_bucket_fast_protective_stop", order_id))
        return True


def strategy_state(**overrides) -> SimpleNamespace:
    values = dict(
        avg_entry_price=100.0,
        three_stage_tp1_price=None,
        three_stage_tp1_ratio=0.0,
        three_stage_tp2_price=None,
        three_stage_tp2_ratio=0.0,
        three_stage_runner_ratio=0.0,
        three_stage_post_tp1_protective_sl_order_id=None,
        three_stage_post_tp1_protective_sl_price=None,
        three_stage_post_tp1_protected=False,
        middle_runner_protective_sl_order_id=None,
        middle_runner_protective_sl_price=None,
        middle_runner_size_mismatch_protected=False,
        middle_bucket_split_fast_sl_order_id=None,
        middle_bucket_split_fast_sl_price=None,
        middle_bucket_split_fast_sl_protected=False,
        middle_bucket_split_fast_sl_invalid_action_taken=None,
        delayed_market_exit_armed=False,
        delayed_market_exit_reason=None,
        delayed_market_exit_context=None,
        delayed_market_exit_side=None,
        delayed_market_exit_position_id=None,
        delayed_market_exit_source_event=None,
        delayed_market_exit_armed_ts_ms=None,
        delayed_market_exit_deadline_ts_ms=None,
        delayed_market_exit_manual_intervention_required=False,
        delayed_market_exit_last_error=None,
        delayed_market_exit_status=None,
        delayed_market_exit_executed_ts_ms=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_strategy(state: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(middle_bucket_split_fast_sl_fee_buffer_pct=0.01),
    )


async def run_phase(
    *,
    strategy,
    trader,
    journal,
    execution_state,
    three_stage_post_tp1_sl_payload=None,
    middle_runner_sl_payload=None,
    middle_bucket_split_fast_protection_payload=None,
):
    return await run_account_sync_protective_orders_phase(
        state_lock=asyncio.Lock(),
        execution_state=execution_state,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=FakeStateStore(),
        save_state_payload=None,
        three_stage_post_tp1_cancel_payload=None,
        three_stage_post_tp1_sl_payload=three_stage_post_tp1_sl_payload,
        middle_runner_sl_payload=middle_runner_sl_payload,
        middle_runner_activation_payload=None,
        middle_bucket_split_event_payload=None,
        middle_bucket_split_fast_protection_payload=middle_bucket_split_fast_protection_payload,
    )


def three_stage_payload(**overrides) -> dict:
    payload = {
        "position_id": "pos-1",
        "side": "LONG",
        "contracts": Decimal("4"),
        "core_contracts": Decimal("4"),
        "net_contracts": Decimal("4"),
        "protective_sl_price": 104.0,
        "old_sl_order_id": None,
        "old_sl_price": None,
        "old_protected": False,
    }
    payload.update(overrides)
    return payload


def middle_runner_payload(**overrides) -> dict:
    payload = {
        "position_id": "pos-1",
        "side": "LONG",
        "contracts": Decimal("4"),
        "core_contracts": Decimal("4"),
        "net_contracts": Decimal("4"),
        "protective_sl_price": 105.0,
        "old_sl_order_id": None,
        "old_sl_price": None,
        "old_protected": False,
        "reason": "partial_tp_filled",
    }
    payload.update(overrides)
    return payload


def fast_payload(**overrides) -> dict:
    payload = {
        "position_id": "pos-1",
        "side": "LONG",
        "avg_entry_price": 100.0,
        "current_price": 110.0,
        "net_contracts": Decimal("4"),
        "invalid_action": "MARKET_EXIT",
        "enabled": True,
        "old_sl_order_id": None,
        "old_sl_price": None,
        "old_protected": False,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_three_stage_failure_keeps_existing_stronger_sl_without_dme() -> None:
    state = strategy_state()
    strategy = fake_strategy(state)
    trader = FakeTrader()
    trader.three_stage_result = (False, None, "invalid_price")
    journal = FakeJournal()
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)

    await run_phase(
        strategy=strategy,
        trader=trader,
        journal=journal,
        execution_state=execution_state,
        three_stage_post_tp1_sl_payload=three_stage_payload(
            protective_sl_price=104.0,
            old_sl_order_id="old-sl",
            old_sl_price=105.0,
            old_protected=True,
        ),
    )

    assert execution_state.trading_halted is False
    assert execution_state.halt_reason is None
    assert state.three_stage_post_tp1_protective_sl_order_id == "old-sl"
    assert state.three_stage_post_tp1_protective_sl_price == 105.0
    assert state.three_stage_post_tp1_protected is True
    event_names = [event for event, _payload, _position_id in journal.events]
    assert "THREE_STAGE_POST_TP1_PROTECTIVE_SL_KEEP_EXISTING" in event_names
    assert "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED" not in event_names
    assert state.delayed_market_exit_armed is False


@pytest.mark.asyncio
async def test_three_stage_failure_without_old_sl_arms_dme() -> None:
    state = strategy_state()
    strategy = fake_strategy(state)
    trader = FakeTrader()
    trader.three_stage_result = (False, None, "invalid_price")
    journal = FakeJournal()
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)

    await run_phase(
        strategy=strategy,
        trader=trader,
        journal=journal,
        execution_state=execution_state,
        three_stage_post_tp1_sl_payload=three_stage_payload(),
    )

    assert execution_state.trading_halted is True
    assert execution_state.halt_reason == "three_stage_post_tp1_sl_failed_delayed_market_exit_armed"
    assert state.delayed_market_exit_armed is True
    assert "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED" in [
        event for event, _payload, _position_id in journal.events
    ]


@pytest.mark.asyncio
async def test_middle_runner_failure_keeps_existing_stronger_short_sl_without_dme() -> None:
    state = strategy_state()
    strategy = fake_strategy(state)
    trader = FakeTrader()
    trader.middle_runner_result = (False, None, "invalid_price")
    journal = FakeJournal()
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)

    await run_phase(
        strategy=strategy,
        trader=trader,
        journal=journal,
        execution_state=execution_state,
        middle_runner_sl_payload=middle_runner_payload(
            side="SHORT",
            protective_sl_price=96.0,
            old_sl_order_id="old-middle-sl",
            old_sl_price=95.0,
            old_protected=True,
        ),
    )

    assert execution_state.trading_halted is False
    assert execution_state.halt_reason is None
    assert state.middle_runner_protective_sl_order_id == "old-middle-sl"
    assert state.middle_runner_protective_sl_price == 95.0
    assert state.delayed_market_exit_armed is False
    assert "MIDDLE_RUNNER_PROTECTIVE_SL_KEEP_EXISTING" in [
        event for event, _payload, _position_id in journal.events
    ]


@pytest.mark.asyncio
async def test_middle_runner_failure_with_weaker_old_sl_arms_dme() -> None:
    state = strategy_state()
    strategy = fake_strategy(state)
    trader = FakeTrader()
    trader.middle_runner_result = (False, None, "invalid_price")
    journal = FakeJournal()
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)

    await run_phase(
        strategy=strategy,
        trader=trader,
        journal=journal,
        execution_state=execution_state,
        middle_runner_sl_payload=middle_runner_payload(
            side="LONG",
            protective_sl_price=105.0,
            old_sl_order_id="old-middle-sl",
            old_sl_price=103.0,
            old_protected=True,
        ),
    )

    assert execution_state.trading_halted is True
    assert execution_state.halt_reason == "middle_runner_sl_failed_delayed_market_exit_armed"
    assert state.delayed_market_exit_armed is True
    assert "MIDDLE_RUNNER_ORDER_WARNING" in [event for event, _payload, _position_id in journal.events]


@pytest.mark.asyncio
async def test_middle_bucket_fast_failure_keeps_existing_stronger_fast_sl_without_dme() -> None:
    state = strategy_state()
    strategy = fake_strategy(state)
    trader = FakeTrader()
    trader.fast_result = (False, None, "invalid_price")
    journal = FakeJournal()
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)

    await run_phase(
        strategy=strategy,
        trader=trader,
        journal=journal,
        execution_state=execution_state,
        middle_bucket_split_fast_protection_payload=fast_payload(
            old_sl_order_id="old-fast-sl",
            old_sl_price=102.0,
            old_protected=True,
        ),
    )

    assert execution_state.trading_halted is False
    assert execution_state.halt_reason is None
    assert state.middle_bucket_split_fast_sl_order_id == "old-fast-sl"
    assert state.middle_bucket_split_fast_sl_price == 102.0
    assert state.middle_bucket_split_fast_sl_protected is True
    assert state.delayed_market_exit_armed is False
    assert "MIDDLE_BUCKET_FAST_PROTECTIVE_SL_KEEP_EXISTING" in [
        event for event, _payload, _position_id in journal.events
    ]


@pytest.mark.asyncio
async def test_three_stage_success_places_new_before_canceling_old() -> None:
    state = strategy_state()
    strategy = fake_strategy(state)
    trader = FakeTrader()
    trader.three_stage_result = (True, "new-sl", "ok")
    journal = FakeJournal()
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)

    await run_phase(
        strategy=strategy,
        trader=trader,
        journal=journal,
        execution_state=execution_state,
        three_stage_post_tp1_sl_payload=three_stage_payload(
            old_sl_order_id="old-sl",
            old_sl_price=103.0,
            old_protected=True,
        ),
    )

    assert trader.calls[:2] == [
        ("place_three_stage_post_tp1_protective_stop_with_retries", None),
        ("cancel_three_stage_post_tp1_protective_stop", "old-sl"),
    ]
    assert state.three_stage_post_tp1_protective_sl_order_id == "new-sl"
    assert state.three_stage_post_tp1_protective_sl_price == 104.0
