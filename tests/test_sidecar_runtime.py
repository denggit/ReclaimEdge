from __future__ import annotations

from decimal import Decimal

import pytest

from scripts.run_boll_cvd_live import (
    ExecutionState,
    apply_sidecar_startup_recovery,
    execute_sidecar_after_core_entry,
    force_close_sidecar_after_core_flat,
    monitor_sidecar_orders_once,
    refresh_sidecar_state_totals,
    sidecar_position_mismatch,
)
from src.execution.trader import PositionSnapshot
from src.risk.simple_position_sizer import PositionSize
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent


def intent(intent_type: str = "OPEN_LONG", layer_index: int = 1) -> TradeIntent:
    return TradeIntent(
        intent_type=intent_type,  # type: ignore[arg-type]
        side="LONG",
        price=3000.0,
        layer_index=layer_index,
        tp_price=3100.0,
        reason="test",
        size=PositionSize(30.0, 1500.0, 0.5, layer_index, 1.0 if layer_index == 1 else 1.15),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=1000 + layer_index,
        avg_entry_price=3000.0,
        breakeven_price=3003.0,
        tp_mode="MIDDLE",
    )


class Journal:
    def __init__(self) -> None:
        self.events = []

    def append(self, event, payload, position_id=None):  # type: ignore[no-untyped-def]
        self.events.append((event, dict(payload), position_id))


class Store:
    def __init__(self) -> None:
        self.saved = []
        self.cleared = 0

    def save(self, state):  # type: ignore[no-untyped-def]
        self.saved.append(state)

    def clear(self) -> None:
        self.cleared += 1


class Trader:
    symbol = "ETH-USDT-SWAP"
    account_equity_usdt = 1000.0
    leverage = "50"

    def __init__(self) -> None:
        self.sidecar_market_orders = []
        self.sidecar_tps = []
        self.cancelled_sidecar_tps = []
        self.market_exits = []
        self.status_by_order = {}

    async def place_sidecar_market_order(self, *, side, eth_qty):  # type: ignore[no-untyped-def]
        self.sidecar_market_orders.append((side, eth_qty))
        return {"order_id": "sc-market", "contracts": "1.66", "qty": 0.166}

    async def place_sidecar_fixed_take_profit(self, *, side, contracts, tp_price, client_order_id=None):  # type: ignore[no-untyped-def]
        self.sidecar_tps.append((side, contracts, tp_price, client_order_id))
        return "sidecar-tp"

    async def fetch_sidecar_order_status(self, order_id: str):  # type: ignore[no-untyped-def]
        return {"order_id": order_id, "status": self.status_by_order.get(order_id, "OPEN"), "filled_qty": None, "avg_fill_price": None}

    async def cancel_sidecar_take_profit(self, order_id: str):  # type: ignore[no-untyped-def]
        self.cancelled_sidecar_tps.append(order_id)
        return True

    async def market_exit_remaining_position_with_retries(self, side, retry_count):  # type: ignore[no-untyped-def]
        self.market_exits.append((side, retry_count))
        return True, "ok"


def sidecar_state() -> StrategyPositionState:
    return StrategyPositionState(
        side="LONG",
        layers=1,
        total_entry_qty=0.5,
        avg_entry_price=3000,
        sidecar_enabled_for_position=True,
        sidecar_margin_pct=0.01,
        sidecar_tp_pct=0.004,
    )


@pytest.mark.asyncio
async def test_open_long_success_creates_sidecar_leg_and_tp() -> None:
    state = sidecar_state()
    execution = ExecutionState("pos-1", 1000.0)
    trader = Trader()
    journal = Journal()
    store = Store()

    ok = await execute_sidecar_after_core_entry(
        trader=trader, strategy_state=state, execution_state=execution, intent=intent(), journal=journal, state_store=store, trader_symbol="ETH-USDT-SWAP"
    )

    assert ok
    assert state.sidecar_legs[0]["status"] == "OPEN"
    assert state.sidecar_legs[0]["tp_order_id"] == "sidecar-tp"
    assert state.sidecar_legs[0]["tp_price"] == pytest.approx(3012.0)
    assert [event[0] for event in journal.events] == ["SIDECAR_LEG_OPENED", "SIDECAR_TP_PLACED"]
    assert store.saved[-1].sidecar_legs[0]["tp_order_id"] == "sidecar-tp"


@pytest.mark.asyncio
async def test_add_long_creates_layer_sidecar_leg() -> None:
    state = sidecar_state()
    execution = ExecutionState("pos-1", 1000.0)
    trader = Trader()

    await execute_sidecar_after_core_entry(
        trader=trader, strategy_state=state, execution_state=execution, intent=intent("ADD_LONG", 2), journal=Journal(), state_store=Store(), trader_symbol="ETH-USDT-SWAP"
    )

    assert state.sidecar_legs[0]["layer_index"] == 2
    assert trader.sidecar_market_orders


@pytest.mark.asyncio
async def test_position_level_disabled_does_not_create_sidecar_mid_position() -> None:
    state = sidecar_state()
    state.sidecar_enabled_for_position = False
    trader = Trader()

    ok = await execute_sidecar_after_core_entry(
        trader=trader, strategy_state=state, execution_state=ExecutionState("pos-1", 1000.0), intent=intent("ADD_LONG", 2), journal=Journal(), state_store=Store(), trader_symbol="ETH-USDT-SWAP"
    )

    assert ok
    assert state.sidecar_legs == []
    assert trader.sidecar_market_orders == []


@pytest.mark.asyncio
async def test_sidecar_tp_filled_updates_state() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    refresh_sidecar_state_totals(state)
    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
    journal = Journal()

    await monitor_sidecar_orders_once(
        trader=trader,
        strategy_state=state,
        execution_state=ExecutionState("pos-1", 1000.0),
        journal=journal,
        state_store=Store(),
        trader_symbol="ETH-USDT-SWAP",
        core_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        position_id="pos-1",
        cash_before_position=1000.0,
        ts_ms=2,
    )

    assert state.sidecar_legs[0]["status"] == "TP_FILLED"
    assert state.sidecar_open_qty == 0
    assert journal.events[0][0] == "SIDECAR_TP_FILLED"


@pytest.mark.asyncio
async def test_sidecar_tp_missing_while_core_active_halts() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    trader = Trader()
    trader.status_by_order["tp-1"] = "UNKNOWN"
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()

    await monitor_sidecar_orders_once(
        trader=trader,
        strategy_state=state,
        execution_state=execution,
        journal=journal,
        state_store=Store(),
        trader_symbol="ETH-USDT-SWAP",
        core_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        position_id="pos-1",
        cash_before_position=1000.0,
        ts_ms=2,
    )

    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_tp_order_missing_or_unknown"
    assert journal.events[0][0] == "SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN"


@pytest.mark.asyncio
async def test_core_flat_force_closes_sidecar() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    refresh_sidecar_state_totals(state)
    trader = Trader()
    journal = Journal()

    ok = await force_close_sidecar_after_core_flat(
        trader=trader,
        strategy_state=state,
        execution_state=ExecutionState("pos-1", 1000.0),
        journal=journal,
        state_store=Store(),
        trader_symbol="ETH-USDT-SWAP",
        position_id="pos-1",
        cash_before_position=1000.0,
        ts_ms=2,
    )

    assert ok
    assert trader.cancelled_sidecar_tps == ["tp-1"]
    assert trader.market_exits
    assert state.sidecar_legs[0]["status"] == "FORCE_CLOSED"
    assert journal.events[0][0] == "SIDECAR_FORCE_CLOSED_AFTER_CORE_FLAT"


def test_core_sidecar_position_mismatch_detected() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"status": "OPEN", "qty": 0.2, "contracts": "2", "tp_order_id": "tp"}]

    assert sidecar_position_mismatch(PositionSnapshot("LONG", Decimal("1"), 3000, 0.1, Decimal("1")), state)


@pytest.mark.asyncio
async def test_startup_recovery_open_order_continues() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    trader = Trader()
    execution = ExecutionState("pos-1", 1000.0)

    await apply_sidecar_startup_recovery(
        strategy=type("S", (), {"state": state})(),
        execution_state=execution,
        saved_state=type("Saved", (), {"sidecar_legs": state.sidecar_legs})(),
        startup_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        trader=trader,
        journal=Journal(),
        state_store=Store(),
    )

    assert not execution.trading_halted
    assert state.sidecar_legs[0]["status"] == "OPEN"


@pytest.mark.asyncio
async def test_startup_recovery_unknown_order_halts() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    trader = Trader()
    trader.status_by_order["tp-1"] = "UNKNOWN"
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()

    await apply_sidecar_startup_recovery(
        strategy=type("S", (), {"state": state})(),
        execution_state=execution,
        saved_state=type("Saved", (), {"sidecar_legs": state.sidecar_legs})(),
        startup_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        trader=trader,
        journal=journal,
        state_store=Store(),
    )

    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_startup_order_state_unknown"
    assert journal.events[0][0] == "SIDECAR_STARTUP_ORDER_STATE_UNKNOWN"


@pytest.mark.asyncio
async def test_recovered_position_without_saved_sidecar_does_not_backfill(monkeypatch) -> None:
    monkeypatch.setenv("SIDECAR_ENABLED", "true")
    state = sidecar_state()
    state.sidecar_legs = []
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()

    await apply_sidecar_startup_recovery(
        strategy=type("S", (), {"state": state})(),
        execution_state=execution,
        saved_state=type("Saved", (), {"sidecar_legs": []})(),
        startup_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        trader=Trader(),
        journal=journal,
        state_store=Store(),
    )

    assert state.sidecar_enabled_for_position is False
    assert journal.events[0][0] == "SIDECAR_DISABLED_FOR_RECOVERED_POSITION"


def test_flat_cleanup_resets_sidecar_fields() -> None:
    state = StrategyPositionState()

    assert state.sidecar_enabled_for_position is False
    assert state.sidecar_legs == []
    assert state.sidecar_open_qty == 0
