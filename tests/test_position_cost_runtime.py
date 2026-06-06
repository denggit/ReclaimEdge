from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.trader import PositionSnapshot
from src.position_management.cost_runtime import (
    record_core_position_reduction_exit,
    record_remaining_entry_notional,
    record_remaining_exit_notional,
    record_sidecar_tp_fill_exit,
    refresh_net_remaining_breakeven,
    sync_strategy_cost_from_position,
)
from src.position_management.sidecar.model import SidecarLegStatus
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)


def position(side: str = "LONG", qty: float = 1.0, avg_entry: float = 100.0) -> PositionSnapshot:
    return PositionSnapshot(side, Decimal("1"), avg_entry, qty, Decimal("1"))  # type: ignore[arg-type]


def strategy() -> BollCvdReclaimStrategy:
    return BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


def test_refresh_net_remaining_breakeven_resets_unknown_side() -> None:
    state = StrategyPositionState(side=None, net_remaining_breakeven_price=123.0)

    refresh_net_remaining_breakeven(state)

    assert state.net_remaining_breakeven_price == 0.0


def test_record_remaining_entry_notional_adds_entry_notional_and_remaining_qty() -> None:
    state = StrategyPositionState(side="LONG")

    record_remaining_entry_notional(state, qty=2.0, price=100.0)

    assert state.position_cost_entry_notional == 200.0
    assert state.position_cost_remaining_qty == 2.0
    assert state.net_remaining_breakeven_price == pytest.approx(100.1)


def test_record_remaining_exit_notional_adds_exit_notional_and_reduces_remaining_qty() -> None:
    state = StrategyPositionState(
        side="LONG",
        position_cost_entry_notional=300.0,
        position_cost_remaining_qty=3.0,
    )

    record_remaining_exit_notional(state, qty=1.0, price=110.0)

    assert state.position_cost_exit_notional == 110.0
    assert state.position_cost_remaining_qty == 2.0


def test_record_core_position_reduction_exit_uses_expected_remaining_qty_branch() -> None:
    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=1.0,
        position_cost_entry_notional=100.0,
        position_cost_remaining_qty=1.0,
    )

    record_core_position_reduction_exit(
        state,
        position(qty=0.2),
        exit_price=110.0,
        expected_remaining_qty=0.6,
    )

    assert state.position_cost_exit_notional == pytest.approx(44.0)
    assert state.position_cost_remaining_qty == pytest.approx(0.6)


def test_record_sidecar_tp_fill_exit_uses_status_fill_values() -> None:
    state = StrategyPositionState(
        side="LONG",
        position_cost_entry_notional=200.0,
        position_cost_remaining_qty=2.0,
    )
    leg = {"qty": 0.5, "tp_price": 120.0, "status": SidecarLegStatus.OPEN.value}
    status = {"filled_qty": "0.4", "avg_fill_price": "115"}

    record_sidecar_tp_fill_exit(state, leg, status)

    assert state.position_cost_exit_notional == pytest.approx(46.0)
    assert state.position_cost_remaining_qty == pytest.approx(1.6)


def test_record_sidecar_tp_fill_exit_falls_back_to_leg_qty_and_tp_price() -> None:
    state = StrategyPositionState(
        side="LONG",
        position_cost_entry_notional=200.0,
        position_cost_remaining_qty=2.0,
    )
    leg = {"qty": 0.5, "tp_price": 120.0, "status": SidecarLegStatus.OPEN.value}
    status = {"filled_qty": None, "avg_fill_price": None}

    record_sidecar_tp_fill_exit(state, leg, status)

    assert state.position_cost_exit_notional == pytest.approx(60.0)
    assert state.position_cost_remaining_qty == pytest.approx(1.5)


def test_sync_strategy_cost_from_position_three_stage_keeps_total_entry_cost() -> None:
    strat = strategy()
    strat.state = StrategyPositionState(
        side="LONG",
        layers=1,
        total_entry_qty=10.0,
        total_entry_notional=1000.0,
        avg_entry_price=100.0,
        three_stage_runner_enabled_for_position=True,
    )

    sync_strategy_cost_from_position(strat, position(qty=5.0, avg_entry=120.0))

    assert strat.state.total_entry_qty == 10.0
    assert strat.state.total_entry_notional == 1000.0
    assert strat.state.avg_entry_price == 120.0
    assert strat.state.last_entry_price == 120.0


def test_sync_strategy_cost_from_position_calls_restore_callback_on_state_mismatch() -> None:
    strat = strategy()
    strat.state = StrategyPositionState(side=None, layers=0)
    restored: list[tuple[BollCvdReclaimStrategy, PositionSnapshot]] = []

    sync_strategy_cost_from_position(
        strat,
        position(side="LONG", qty=1.0, avg_entry=100.0),
        restore_from_position=lambda strategy_arg, position_arg: restored.append((strategy_arg, position_arg)),
    )

    assert restored == [(strat, position(side="LONG", qty=1.0, avg_entry=100.0))]
