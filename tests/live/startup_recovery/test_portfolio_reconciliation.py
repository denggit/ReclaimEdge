from __future__ import annotations

from decimal import Decimal

from src.execution.trader_types import PositionSnapshot
from src.live.startup_recovery.portfolio_reconciliation import reconcile_startup_state
from src.portfolio.capital_ledger import (
    LEDGER_VERSION,
    CapitalLedgerSnapshot,
    SymbolCapitalState,
)
from src.reporting.live_state_store import LivePositionState


INST_ID = "ETH-USDT-SWAP"


def _position(*, side: str | None = None, contracts: str = "0") -> PositionSnapshot:
    return PositionSnapshot(
        side=side,  # type: ignore[arg-type]
        contracts=Decimal(contracts),
        avg_entry_price=3000.0 if side is not None else 0.0,
        eth_qty=1.0 if side is not None else 0.0,
        raw_pos=Decimal(contracts),
    )


def _ledger(
    state: SymbolCapitalState | None = None,
) -> CapitalLedgerSnapshot:
    return CapitalLedgerSnapshot(
        version=LEDGER_VERSION,
        updated_ms=1,
        leader_symbol=None,
        global_no_new_entry=False,
        symbols={INST_ID: state or SymbolCapitalState()},
    )


def _active_ledger(
    *,
    side: str = "LONG",
    used_layers: int = 1,
    planned_main_contracts: tuple[str, ...] = ("1",),
    plan_max_layers: int = 1,
    position_plan_id: str | None = "plan-1",
) -> SymbolCapitalState:
    return SymbolCapitalState(
        state="OPEN",
        side=side,
        used_layers=used_layers,
        position_plan_id=position_plan_id,
        planned_main_contracts=planned_main_contracts,
        plan_max_layers=plan_max_layers,
    )


def _saved(*, side: str = "LONG", layers: int = 1) -> LivePositionState:
    return LivePositionState(
        position_id="pos-1",
        symbol=INST_ID,
        side=side,
        layers=layers,
    )


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_all_flat_ok_none() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(),
        saved_state=None,
        ledger_snapshot=_ledger(),
    )

    assert result.severity == "OK"
    assert result.action == "NONE"
    assert result.issues == ()


def test_okx_position_without_saved_state_warn_halts_new_risk() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="1"),
        saved_state=None,
        ledger_snapshot=_ledger(_active_ledger()),
    )

    assert result.severity == "WARN"
    assert result.action == "HALT_NEW_RISK"
    assert _codes(result) == {"OKX_POSITION_WITHOUT_SAVED_STATE"}


def test_okx_position_with_ledger_flat_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="1"),
        saved_state=_saved(),
        ledger_snapshot=_ledger(),
    )

    assert result.severity == "CRITICAL"
    assert result.action == "HALT_NEW_RISK"
    assert "OKX_POSITION_LEDGER_FLAT" in _codes(result)


def test_ledger_active_but_okx_flat_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(),
        saved_state=None,
        ledger_snapshot=_ledger(_active_ledger()),
    )

    assert result.severity == "CRITICAL"
    assert "LEDGER_ACTIVE_BUT_OKX_FLAT" in _codes(result)


def test_saved_active_but_okx_flat_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(),
        saved_state=_saved(),
        ledger_snapshot=_ledger(),
    )

    assert result.severity == "CRITICAL"
    assert "SAVED_STATE_ACTIVE_BUT_OKX_FLAT" in _codes(result)


def test_side_mismatch_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="1"),
        saved_state=_saved(side="SHORT"),
        ledger_snapshot=_ledger(_active_ledger(side="LONG")),
    )

    assert result.severity == "CRITICAL"
    assert "SIDE_MISMATCH" in _codes(result)


def test_layer_mismatch_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="2"),
        saved_state=_saved(layers=2),
        ledger_snapshot=_ledger(_active_ledger(used_layers=1)),
    )

    assert result.severity == "CRITICAL"
    assert "LAYER_MISMATCH" in _codes(result)


def test_ledger_active_missing_plan_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="1"),
        saved_state=_saved(),
        ledger_snapshot=_ledger(
            _active_ledger(
                position_plan_id=None,
                planned_main_contracts=(),
                plan_max_layers=0,
            )
        ),
    )

    assert result.severity == "CRITICAL"
    assert "LEDGER_ACTIVE_MISSING_PLAN" in _codes(result)


def test_ledger_plan_length_mismatch_is_critical() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="1"),
        saved_state=_saved(),
        ledger_snapshot=_ledger(
            _active_ledger(
                planned_main_contracts=("1",),
                plan_max_layers=2,
            )
        ),
    )

    assert result.severity == "CRITICAL"
    assert "LEDGER_PLAN_LENGTH_MISMATCH" in _codes(result)


def test_okx_saved_and_ledger_consistent_ok_none() -> None:
    result = reconcile_startup_state(
        inst_id=INST_ID,
        position=_position(side="LONG", contracts="1"),
        saved_state=_saved(),
        ledger_snapshot=_ledger(_active_ledger()),
    )

    assert result.severity == "OK"
    assert result.action == "NONE"
    assert result.issues == ()
