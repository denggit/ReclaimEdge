from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_DOWN

import pytest

import asyncio

from scripts.run_boll_cvd_live import (
    ExecutionState,
    SidecarPreCoreReconcileResult,
    apply_main_tp_startup_recovery,
    apply_sidecar_startup_recovery,
    execute_sidecar_after_core_entry,
    force_close_sidecar_after_core_flat,
    monitor_sidecar_orders_once,
    reconcile_sidecar_orders_before_core_view,
    refresh_sidecar_state_totals,
    sidecar_open_contracts,
    sidecar_position_mismatch,
)
from src.execution.trader import PositionSnapshot
from src.position_management.sidecar.model import sidecar_open_qty
from src.position_management.sidecar.reconciler import build_core_position_view
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
        self.position_snapshot = PositionSnapshot("LONG", Decimal("1"), 3000, 0.1, Decimal("1"))
        self.pending_orders = []

    async def place_sidecar_market_order(self, *, side, eth_qty):  # type: ignore[no-untyped-def]
        self.sidecar_market_orders.append((side, eth_qty))
        return {"order_id": "sc-market", "contracts": "1.66", "qty": 0.166}

    async def place_sidecar_fixed_take_profit(self, *, side, contracts, tp_price, client_order_id=None):  # type: ignore[no-untyped-def]
        self.sidecar_tps.append((side, contracts, tp_price, client_order_id))
        return "sidecar-tp"

    async def fetch_sidecar_order_status(self, order_id: str):  # type: ignore[no-untyped-def]
        return {"order_id": order_id, "status": self.status_by_order.get(order_id, "OPEN"), "filled_qty": None, "avg_fill_price": None}

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return self.position_snapshot

    async def fetch_pending_orders(self):  # type: ignore[no-untyped-def]
        return list(self.pending_orders)

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


@pytest.mark.asyncio
async def test_sidecar_force_close_mismatch_halts_without_market_exit() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    refresh_sidecar_state_totals(state)
    trader = Trader()
    trader.position_snapshot = PositionSnapshot("LONG", Decimal("2"), 3000, 0.2, Decimal("2"))
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()

    ok = await force_close_sidecar_after_core_flat(
        trader=trader,
        strategy_state=state,
        execution_state=execution,
        journal=journal,
        state_store=Store(),
        trader_symbol="ETH-USDT-SWAP",
        position_id="pos-1",
        cash_before_position=1000.0,
        ts_ms=2,
    )

    assert not ok
    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_force_close_position_mismatch"
    assert trader.cancelled_sidecar_tps == []
    assert trader.market_exits == []
    assert journal.events[0][0] == "SIDECAR_FORCE_CLOSE_POSITION_MISMATCH"


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


@pytest.mark.asyncio
async def test_startup_recovery_tp_filled_sidecar_position_stays_enabled() -> None:
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "TP_FILLED", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 2}]
    execution = ExecutionState("pos-1", 1000.0)
    saved_state = type(
        "Saved",
        (),
        {
            "sidecar_enabled_for_position": True,
            "sidecar_margin_pct": 0.01,
            "sidecar_tp_pct": 0.004,
            "sidecar_legs": state.sidecar_legs,
        },
    )()

    await apply_sidecar_startup_recovery(
        strategy=type("S", (), {"state": state})(),
        execution_state=execution,
        saved_state=saved_state,
        startup_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        trader=Trader(),
        journal=Journal(),
        state_store=Store(),
    )

    assert state.sidecar_enabled_for_position is True


@pytest.mark.asyncio
async def test_startup_restores_main_tp_order_id() -> None:
    trader = Trader()
    execution = ExecutionState("pos-1", 1000.0)
    saved_state = type("Saved", (), {"tp_order_id": "core-tp", "tp_order_ids": []})()

    await apply_main_tp_startup_recovery(
        execution_state=execution,
        saved_state=saved_state,
        startup_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        trader=trader,
        journal=Journal(),
    )

    assert getattr(trader, "tp_order_id") == "core-tp"
    assert not execution.trading_halted


@pytest.mark.asyncio
async def test_startup_missing_main_tp_order_id_halts_when_reduce_only_exists() -> None:
    trader = Trader()
    trader.pending_orders = [{"instId": "ETH-USDT-SWAP", "reduceOnly": "true", "ordId": "unknown-tp"}]
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()

    await apply_main_tp_startup_recovery(
        execution_state=execution,
        saved_state=type("Saved", (), {"tp_order_id": None, "tp_order_ids": []})(),
        startup_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        trader=trader,
        journal=journal,
    )

    assert execution.trading_halted
    assert execution.halt_reason == "main_tp_order_id_missing_on_startup"
    assert journal.events[0][0] == "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP"


def test_flat_cleanup_resets_sidecar_fields() -> None:
    state = StrategyPositionState()

    assert state.sidecar_enabled_for_position is False
    assert state.sidecar_legs == []
    assert state.sidecar_open_qty == 0


# ---------------------------------------------------------------------------
# Tests for with_entry_add_managed_core_contracts
# ---------------------------------------------------------------------------

from scripts.run_boll_cvd_live import with_entry_add_managed_core_contracts  # noqa: E402


class FakeTraderForManagedCore:
    """Minimal fake trader for with_entry_add_managed_core_contracts tests."""
    symbol = "ETH-USDT-SWAP"
    contract_multiplier = Decimal("0.1")
    contract_precision = Decimal("0.01")
    min_contracts = Decimal("0.01")

    def eth_qty_to_contracts(self, eth_qty: Decimal) -> Decimal:
        raw = eth_qty / self.contract_multiplier
        lots = (raw / self.contract_precision).to_integral_value(rounding=ROUND_DOWN)
        return lots * self.contract_precision


def _make_open_intent(eth_qty: float, intent_type: str = "OPEN_LONG") -> TradeIntent:
    return TradeIntent(
        intent_type=intent_type,
        side="LONG",
        price=3000.0,
        layer_index=1,
        tp_price=3100.0,
        reason="test",
        size=PositionSize(30.0, 1500.0, eth_qty, 1, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=1000,
        avg_entry_price=3000.0,
        breakeven_price=3003.0,
        tp_mode="MIDDLE",
    )


def test_open_long_sidecar_enabled_core_flat_populates_managed_core() -> None:
    """OPEN_LONG sidecar enabled, core flat: managed_core_contracts = 10 (1 ETH → 10 contracts)."""
    state = StrategyPositionState(sidecar_enabled_for_position=True)
    intent_t = _make_open_intent(eth_qty=1.0, intent_type="OPEN_LONG")
    trader = FakeTraderForManagedCore()
    # core flat = None position
    core_position = PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

    result = with_entry_add_managed_core_contracts(
        intent=intent_t,
        strategy_state=state,
        account_core_position=core_position,
        trader=trader,
    )

    assert Decimal(result.managed_core_contracts) == Decimal("10")
    assert float(result.managed_core_eth_qty) == pytest.approx(1.0)


def test_add_long_sidecar_enabled_core_has_position_populates_managed_core() -> None:
    """ADD_LONG sidecar enabled: core=10 contracts, add 1 ETH→10 contracts -> managed_core_contracts=20."""
    state = StrategyPositionState(sidecar_enabled_for_position=True)
    intent_t = _make_open_intent(eth_qty=1.0, intent_type="ADD_LONG")
    trader = FakeTraderForManagedCore()
    # Core position has 10 contracts (1 ETH), side is LONG matching intent
    # account_core_position is already the core view (not OKX net)
    core_position = PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"))

    result = with_entry_add_managed_core_contracts(
        intent=intent_t,
        strategy_state=state,
        account_core_position=core_position,
        trader=trader,
    )

    # existing 10 + new 10 = 20
    assert Decimal(result.managed_core_contracts) == Decimal("20")
    # existing 1.0 + new 1.0 = 2.0
    assert float(result.managed_core_eth_qty) == pytest.approx(2.0)


def test_add_long_sidecar_enabled_core_side_mismatch_uses_zero() -> None:
    """ADD_LONG sidecar enabled but core has SHORT position: current_core_contracts=0."""
    state = StrategyPositionState(sidecar_enabled_for_position=True)
    intent_t = _make_open_intent(eth_qty=1.0, intent_type="ADD_LONG")
    trader = FakeTraderForManagedCore()
    # Core position is SHORT, intent is LONG → side mismatch
    core_position = PositionSnapshot("SHORT", Decimal("5"), 3000.0, 0.5, Decimal("5"))

    result = with_entry_add_managed_core_contracts(
        intent=intent_t,
        strategy_state=state,
        account_core_position=core_position,
        trader=trader,
    )

    # side mismatch → current_core_contracts = 0, expected = 0 + 10 = 10
    assert Decimal(result.managed_core_contracts) == Decimal("10")
    assert float(result.managed_core_eth_qty) == pytest.approx(1.0)


def test_add_long_sidecar_disabled_returns_unchanged() -> None:
    """ADD_LONG with sidecar disabled: managed_core_contracts remains None (old logic)."""
    state = StrategyPositionState(sidecar_enabled_for_position=False)
    intent_t = _make_open_intent(eth_qty=1.0, intent_type="ADD_LONG")
    trader = FakeTraderForManagedCore()
    core_position = PositionSnapshot("LONG", Decimal("5"), 3000.0, 0.5, Decimal("5"))

    result = with_entry_add_managed_core_contracts(
        intent=intent_t,
        strategy_state=state,
        account_core_position=core_position,
        trader=trader,
    )

    assert result is intent_t
    assert result.managed_core_contracts is None


def test_update_tp_intent_not_modified() -> None:
    """UPDATE_TP intent is not modified by with_entry_add_managed_core_contracts."""
    state = StrategyPositionState(sidecar_enabled_for_position=True)
    intent_t = _make_open_intent(eth_qty=1.0, intent_type="UPDATE_TP")
    trader = FakeTraderForManagedCore()
    core_position = PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"))

    result = with_entry_add_managed_core_contracts(
        intent=intent_t,
        strategy_state=state,
        account_core_position=core_position,
        trader=trader,
    )

    assert result is intent_t


def test_intent_already_has_managed_core_contracts_not_overwritten() -> None:
    """If managed_core_contracts is already set, it is not overwritten."""
    state = StrategyPositionState(sidecar_enabled_for_position=True)
    intent_t = _make_open_intent(eth_qty=1.0, intent_type="ADD_LONG")
    intent_t = replace(intent_t, managed_core_contracts="99")
    trader = FakeTraderForManagedCore()
    core_position = PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"))

    result = with_entry_add_managed_core_contracts(
        intent=intent_t,
        strategy_state=state,
        account_core_position=core_position,
        trader=trader,
    )

    assert result is intent_t
    assert result.managed_core_contracts == "99"


# ---------------------------------------------------------------------------
# Tests: Sidecar TP_FILLED + active global SL halt
# ---------------------------------------------------------------------------
_GLOBAL_SL_FIELDS = [
    "near_tp_protective_sl_order_id",
    "middle_runner_protective_sl_order_id",
    "three_stage_post_tp1_protective_sl_order_id",
    "trend_runner_sl_order_id",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sl_field", _GLOBAL_SL_FIELDS)
async def test_sidecar_tp_filled_with_active_global_sl_halts(sl_field: str) -> None:
    """Sidecar TP_FILLED + any active global protective SL order -> halt for manual reconcile."""
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    refresh_sidecar_state_totals(state)
    setattr(state, sl_field, "old-sl-001")

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()
    store = Store()

    await monitor_sidecar_orders_once(
        trader=trader,
        strategy_state=state,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        core_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        position_id="pos-1",
        cash_before_position=1000.0,
        ts_ms=2,
    )

    # Leg must be marked TP_FILLED
    assert state.sidecar_legs[0]["status"] == "TP_FILLED"
    # Trading must be halted with the correct reason
    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_tp_filled_requires_global_sl_reconcile"
    # Journal must record the event
    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_FILLED" in event_names
    assert "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE" in event_names
    reconcile_entry = journal.events[event_names.index("SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE")]
    assert sl_field in str(reconcile_entry[1].get("active_global_sl_orders", []))
    # state_store.save must be called
    assert len(store.saved) > 0


@pytest.mark.asyncio
async def test_sidecar_tp_filled_without_global_sl_does_not_halt() -> None:
    """Sidecar TP_FILLED without any active global SL: no halt, just update leg status."""
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    refresh_sidecar_state_totals(state)
    # No global SL order_ids set

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
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

    assert state.sidecar_legs[0]["status"] == "TP_FILLED"
    assert not execution.trading_halted
    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_FILLED" in event_names
    assert "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE" not in event_names


@pytest.mark.asyncio
async def test_sidecar_tp_filled_global_sl_on_trader_also_halts() -> None:
    """SL order_id on trader instance (not just strategy_state) also triggers halt."""
    state = sidecar_state()
    state.sidecar_legs = [{"leg_id": "leg-1", "status": "OPEN", "tp_order_id": "tp-1", "qty": 0.1, "contracts": "1", "created_ts_ms": 1, "updated_ts_ms": 1}]
    refresh_sidecar_state_totals(state)

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
    # Set SL on trader (monitor_sidecar_orders_once checks getattr(trader, sl_field, None) as fallback)
    trader.trend_runner_sl_order_id = "trader-sl-001"
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()
    store = Store()

    await monitor_sidecar_orders_once(
        trader=trader,
        strategy_state=state,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        core_position=PositionSnapshot("LONG", Decimal("5"), 3000, 0.5, Decimal("5")),
        position_id="pos-1",
        cash_before_position=1000.0,
        ts_ms=2,
    )

    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_tp_filled_requires_global_sl_reconcile"
    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE" in event_names
    assert len(store.saved) > 0


# ---------------------------------------------------------------------------
# Tests: reconcile_sidecar_orders_before_core_view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sidecar_tp_filled_reconciled_before_core_view() -> None:
    """Pre-core reconciliation discovers sidecar TP_FILLED so that
    core_position = OKX_net - sidecar_open_qty uses fresh (zero) sidecar open qty.

    Scenario:
    - OKX net position = 3 ETH (sidecar TP already filled, only core remains)
    - Local sidecar_open_qty = 1 ETH (stale — still counts the leg as OPEN)
    - Without pre-core reconcile, core_position = 3 - 1 = 2 ETH (wrong)
    - With pre-core reconcile, sidecar_open_qty = 0, core_position = 3 ETH (correct)
    """
    state = sidecar_state()
    state.sidecar_legs = [
        {
            "leg_id": "leg-1",
            "status": "OPEN",
            "tp_order_id": "tp-1",
            "qty": 1.0,
            "contracts": "10",
            "entry_price": 3000.0,
            "created_ts_ms": 1,
            "updated_ts_ms": 1,
        }
    ]
    refresh_sidecar_state_totals(state)
    # Verify stale state: sidecar_open_qty = 1 ETH
    assert sidecar_open_qty(state.sidecar_legs) == pytest.approx(1.0)

    # OKX net position = 3 ETH / 30 contracts (only core remains after sidecar TP)
    okx_position = PositionSnapshot("LONG", Decimal("30"), 3000.0, 3.0, Decimal("30"))

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()
    store = Store()
    state_lock = asyncio.Lock()

    strategy = type("S", (), {"state": state})()

    # Act: pre-core reconciliation
    result = await reconcile_sidecar_orders_before_core_view(
        trader=trader,
        strategy=strategy,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        ts_ms=2,
        state_lock=state_lock,
    )

    # Assert: sidecar state was updated
    assert isinstance(result, SidecarPreCoreReconcileResult)
    assert result.queried
    assert result.changed
    assert state.sidecar_legs[0]["status"] == "TP_FILLED"
    assert sidecar_open_qty(state.sidecar_legs) == pytest.approx(0.0)

    # Now compute core_position with fresh sidecar_open_qty
    core_position = build_core_position_view(
        okx_position,
        sidecar_open_qty(state.sidecar_legs),
        sidecar_open_contracts(state.sidecar_legs),
    )

    # Core position should be 3 ETH (OKX net), NOT 2 ETH (3 - stale 1)
    assert float(core_position.eth_qty) == pytest.approx(3.0)
    assert core_position.contracts == Decimal("30")

    # Journal should record SIDECAR_TP_FILLED
    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_FILLED" in event_names

    # state_store.save must have been called
    assert len(store.saved) > 0


def test_stale_sidecar_open_qty_would_understate_core_without_pre_reconcile() -> None:
    """Pure-function test: lock in the risk that stale sidecar_open_qty
    understates core_position.

    Without pre-core reconciliation:
      core = OKX_net(3) - stale_sidecar_open_qty(1) = 2 ETH  ← WRONG
    With pre-core reconciliation:
      core = OKX_net(3) - fresh_sidecar_open_qty(0) = 3 ETH  ← CORRECT
    """
    okx_position = PositionSnapshot("LONG", Decimal("30"), 3000.0, 3.0, Decimal("30"))

    # Stale: sidecar TP filled but local state still has open_qty = 1 ETH
    stale_core = build_core_position_view(okx_position, 1.0, Decimal("10"))
    assert float(stale_core.eth_qty) == pytest.approx(2.0)
    assert stale_core.contracts == Decimal("20")

    # Fresh: after pre-core reconcile, sidecar_open_qty = 0
    fresh_core = build_core_position_view(okx_position, 0.0, Decimal("0"))
    assert float(fresh_core.eth_qty) == pytest.approx(3.0)
    assert fresh_core.contracts == Decimal("30")


@pytest.mark.asyncio
async def test_sidecar_tp_filled_with_active_global_sl_still_halts_pre_core() -> None:
    """Pre-core reconciliation must still halt when sidecar TP_FILLED is
    discovered and any active global SL order exists.

    Even though we are reconciling early (before core_position calculation),
    the conservative safety rule must still apply:
      trading_halted = True
      halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
    """
    state = sidecar_state()
    state.sidecar_legs = [
        {
            "leg_id": "leg-1",
            "status": "OPEN",
            "tp_order_id": "tp-1",
            "qty": 0.1,
            "contracts": "1",
            "created_ts_ms": 1,
            "updated_ts_ms": 1,
        }
    ]
    refresh_sidecar_state_totals(state)
    # Set an active global protective SL on strategy state
    state.three_stage_post_tp1_protective_sl_order_id = "old-sl"

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()
    store = Store()
    state_lock = asyncio.Lock()

    strategy = type("S", (), {"state": state})()

    result = await reconcile_sidecar_orders_before_core_view(
        trader=trader,
        strategy=strategy,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        ts_ms=2,
        state_lock=state_lock,
    )

    assert isinstance(result, SidecarPreCoreReconcileResult)
    assert result.queried
    assert result.changed
    # Leg must be marked TP_FILLED
    assert state.sidecar_legs[0]["status"] == "TP_FILLED"
    # Trading must be halted
    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_tp_filled_requires_global_sl_reconcile"
    # Journal must record the event
    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_FILLED" in event_names
    assert "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE" in event_names
    reconcile_entry = journal.events[
        event_names.index("SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE")
    ]
    assert "three_stage_post_tp1_protective_sl_order_id" in str(
        reconcile_entry[1].get("active_global_sl_orders", [])
    )
    # state_store.save must be called
    assert len(store.saved) > 0


# ---------------------------------------------------------------------------
# Tests: Problem 1 — sidecar leg append delayed until after TP placement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sidecar_leg_not_in_state_before_tp_placement() -> None:
    """The leg must not appear in strategy_state.sidecar_legs with
    status=OPEN + tp_order_id=None while TP is being placed.

    After TP succeeds the leg is appended with tp_order_id set.
    """
    state = sidecar_state()
    execution = ExecutionState("pos-1", 1000.0)
    trader = Trader()
    journal = Journal()
    store = Store()

    ok = await execute_sidecar_after_core_entry(
        trader=trader, strategy_state=state, execution_state=execution,
        intent=intent(), journal=journal, state_store=store,
        trader_symbol="ETH-USDT-SWAP",
    )

    assert ok
    # After successful execution the leg must be OPEN with tp_order_id set
    assert len(state.sidecar_legs) == 1
    assert state.sidecar_legs[0]["status"] == "OPEN"
    assert state.sidecar_legs[0]["tp_order_id"] == "sidecar-tp"
    assert state.sidecar_legs[0]["tp_order_id"] is not None
    # No OPEN leg should ever have tp_order_id=None
    for leg in state.sidecar_legs:
        if leg["status"] == "OPEN":
            assert leg.get("tp_order_id") is not None, \
                "OPEN leg must have non-None tp_order_id"


class FailingTpTrader(Trader):
    """Trader whose place_sidecar_fixed_take_profit always raises."""

    async def place_sidecar_fixed_take_profit(self, *, side, contracts, tp_price, client_order_id=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("tp place simulated failure")


@pytest.mark.asyncio
async def test_sidecar_tp_failure_appends_unknown_halted_not_open() -> None:
    """When TP placement fails, the leg must be appended as UNKNOWN_HALTED,
    not OPEN.  An OPEN leg with tp_order_id=None must never be exposed.
    """
    state = sidecar_state()
    execution = ExecutionState("pos-1", 1000.0)
    trader = FailingTpTrader()
    journal = Journal()
    store = Store()

    ok = await execute_sidecar_after_core_entry(
        trader=trader, strategy_state=state, execution_state=execution,
        intent=intent(), journal=journal, state_store=store,
        trader_symbol="ETH-USDT-SWAP",
    )

    assert not ok
    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_tp_place_failed"
    assert state.sidecar_dirty
    assert state.sidecar_halt_reason == "sidecar_tp_place_failed"

    # The leg must be UNKNOWN_HALTED, not OPEN
    assert len(state.sidecar_legs) == 1
    assert state.sidecar_legs[0]["status"] == "UNKNOWN_HALTED"
    assert state.sidecar_legs[0]["warning_recorded"] is True

    # No OPEN leg should exist with tp_order_id=None
    for leg in state.sidecar_legs:
        assert leg["status"] != "OPEN", \
            "No OPEN leg should be appended when TP placement fails"
        if leg["status"] == "OPEN":
            assert leg.get("tp_order_id") is not None

    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_PLACE_FAILED" in event_names
    assert "SIDECAR_LEG_OPENED" not in event_names


# ---------------------------------------------------------------------------
# Tests: Problem 2 — pre-core reconcile queried vs changed semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_core_reconcile_queried_true_when_no_changes() -> None:
    """When OPEN sidecar legs exist and we fetch order status, but all
    orders are still OPEN (no state changes), the pre-core reconcile
    returns queried=True, changed=False.

    This prevents account_position_sync_worker from calling
    monitor_sidecar_orders_once again in the same sync cycle.
    """
    state = sidecar_state()
    state.sidecar_legs = [
        {
            "leg_id": "leg-1",
            "status": "OPEN",
            "tp_order_id": "tp-1",
            "qty": 0.1,
            "contracts": "1",
            "created_ts_ms": 1,
            "updated_ts_ms": 1,
        }
    ]
    refresh_sidecar_state_totals(state)

    trader = Trader()
    trader.status_by_order["tp-1"] = "OPEN"  # still OPEN, no change
    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()
    store = Store()
    state_lock = asyncio.Lock()

    strategy = type("S", (), {"state": state})()

    result = await reconcile_sidecar_orders_before_core_view(
        trader=trader,
        strategy=strategy,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        ts_ms=2,
        state_lock=state_lock,
    )

    assert isinstance(result, SidecarPreCoreReconcileResult)
    assert result.queried is True, \
        "pre-core reconcile queried OPEN sidecar orders → queried must be True"
    assert result.changed is False, \
        "No order status changed → changed must be False"
    # No journal events should be emitted
    assert len(journal.events) == 0
    # No state save should happen
    assert len(store.saved) == 0


@pytest.mark.asyncio
async def test_pre_core_reconcile_skips_when_pending_orders() -> None:
    """When execution_state.pending_order_count > 0, pre-core reconcile
    must return queried=False, changed=False and not query any orders.
    """
    state = sidecar_state()
    state.sidecar_legs = [
        {
            "leg_id": "leg-1",
            "status": "OPEN",
            "tp_order_id": "tp-1",
            "qty": 0.1,
            "contracts": "1",
            "created_ts_ms": 1,
            "updated_ts_ms": 1,
        }
    ]
    refresh_sidecar_state_totals(state)

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"  # would be detected if queried
    execution = ExecutionState("pos-1", 1000.0)
    execution.pending_order_count = 1  # pending order should block
    journal = Journal()
    store = Store()
    state_lock = asyncio.Lock()

    strategy = type("S", (), {"state": state})()

    result = await reconcile_sidecar_orders_before_core_view(
        trader=trader,
        strategy=strategy,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        ts_ms=2,
        state_lock=state_lock,
    )

    assert isinstance(result, SidecarPreCoreReconcileResult)
    assert result.queried is False, \
        "pending_order_count > 0 → queried must be False"
    assert result.changed is False
    # The TP_FILLED must NOT have been detected
    assert state.sidecar_legs[0]["status"] == "OPEN"


# ---------------------------------------------------------------------------
# Tests: Problem 3 — pre-core SL check uses trader fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_core_reconcile_sl_on_trader_triggers_halt() -> None:
    """When sidecar TP_FILLED is discovered and the only active global SL
    order is on the trader instance (not strategy.state), the pre-core
    reconcile must still halt — matching monitor_sidecar_orders_once
    behaviour.
    """
    state = sidecar_state()
    state.sidecar_legs = [
        {
            "leg_id": "leg-1",
            "status": "OPEN",
            "tp_order_id": "tp-1",
            "qty": 0.1,
            "contracts": "1",
            "created_ts_ms": 1,
            "updated_ts_ms": 1,
        }
    ]
    refresh_sidecar_state_totals(state)
    # strategy.state has NO SL order ids
    for sl_field in (
        "near_tp_protective_sl_order_id",
        "middle_runner_protective_sl_order_id",
        "three_stage_post_tp1_protective_sl_order_id",
        "trend_runner_sl_order_id",
    ):
        setattr(state, sl_field, None)

    trader = Trader()
    trader.status_by_order["tp-1"] = "FILLED"
    # SL is only on the trader
    trader.trend_runner_sl_order_id = "trader-sl"

    execution = ExecutionState("pos-1", 1000.0)
    journal = Journal()
    store = Store()
    state_lock = asyncio.Lock()

    strategy = type("S", (), {"state": state})()

    result = await reconcile_sidecar_orders_before_core_view(
        trader=trader,
        strategy=strategy,
        execution_state=execution,
        journal=journal,
        state_store=store,
        trader_symbol="ETH-USDT-SWAP",
        ts_ms=2,
        state_lock=state_lock,
    )

    assert result.changed
    assert result.queried
    # Leg must be marked TP_FILLED
    assert state.sidecar_legs[0]["status"] == "TP_FILLED"
    # Trading must be halted
    assert execution.trading_halted
    assert execution.halt_reason == "sidecar_tp_filled_requires_global_sl_reconcile"
    # Journal must record the halt
    event_names = [e[0] for e in journal.events]
    assert "SIDECAR_TP_FILLED" in event_names
    assert "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE" in event_names
    # state_store.save must be called
    assert len(store.saved) > 0
