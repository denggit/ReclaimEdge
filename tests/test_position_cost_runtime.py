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


# ── TP1 increment validation (ETH qty correctness) ─────────────────────


def test_record_core_position_reduction_exit_uses_eth_qty_for_tp1_increment() -> None:
    """Verify the TP1 exit_notional increment uses ETH qty, not contracts.

    This test only validates the TP1 exit notional increment — that
    reduced_qty is computed correctly as 0.41682721 ETH and produces
    exit_notional ≈ 651.0 USDT.

    Full net_remaining_breakeven validation is covered by the dedicated
    tests below (with and without sidecar).
    """
    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=0.59482721,
        position_cost_entry_notional=1361.68678471,
        position_cost_remaining_qty=0.59482721,
        sidecar_legs=[],
    )
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), 1529.0053, 0.17800000, Decimal("1.78")),
        exit_price=1561.5685,
    )
    expected_exit_notional = 0.41682721 * 1561.5685  # ≈ 651.0
    assert state.position_cost_exit_notional == pytest.approx(expected_exit_notional, rel=1e-7)
    assert state.position_cost_remaining_qty == pytest.approx(0.17800000, rel=1e-7)


def test_record_core_position_reduction_exit_clamps_inflated_state() -> None:
    """Verify the clamp prevents inflation when position_cost_remaining_qty is stale.

    Simulates drifted state where position_cost_remaining_qty is 8x the actual
    remaining (mimicking the observed bug).
    """
    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=0.59482721,
        position_cost_entry_notional=1361.68678471,
        position_cost_remaining_qty=8.0,  # drifted high!
        sidecar_legs=[],
    )
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), 1529.0053, 0.17800000, Decimal("1.78")),
        exit_price=1561.5685,
    )
    # Without clamp: qty = 8.0 - 0.178 = 7.822, exit_notional ≈ 12215
    # With clamp: clamped to total_entry - core.eth = 0.5948 - 0.178 = 0.4168 => exit_notional ≈ 651
    expected_exit_notional = 0.41682721 * 1561.5685
    assert state.position_cost_exit_notional == pytest.approx(expected_exit_notional, rel=1e-7)
    # Drift is corrected: remaining_qty overwritten from position state
    assert state.position_cost_remaining_qty == pytest.approx(0.17800000, rel=1e-7)


def test_record_core_position_reduction_exit_with_sidecar() -> None:
    """Verify sidecar qty is preserved in remaining_qty, excluded from reduced_qty."""
    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=0.59482721,
        position_cost_entry_notional=1361.68678471,
        position_cost_remaining_qty=0.59482721 + 0.05,  # core + sidecar
        sidecar_legs=[{"qty": 0.05, "contracts": "0.5", "status": "OPEN"}],
    )
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), 1529.0053, 0.17800000, Decimal("1.78")),
        exit_price=1561.5685,
    )
    # reduced_qty should be core reduction only: 0.5948 - 0.178 = 0.4168 (excludes sidecar 0.05)
    expected_core_reduction = 0.41682721 * 1561.5685
    assert state.position_cost_exit_notional == pytest.approx(expected_core_reduction, rel=1e-7)
    # remaining_qty = core(0.178) + sidecar(0.05) = 0.228
    assert state.position_cost_remaining_qty == pytest.approx(0.22800000, rel=1e-7)


def test_record_core_position_reduction_exit_tp2_chaining() -> None:
    """TP2 call after TP1 in same cycle should chain correctly."""
    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=0.59482721,
        position_cost_entry_notional=1361.68678471,
        position_cost_remaining_qty=0.59482721,
    )
    # TP1: use expected_remaining_qty to simulate simultaneous TP1+TP2 scenario
    tp1_price = 1561.5685
    expected_after_tp1_core = 0.59482721 * 0.3  # 1 - tp1_ratio
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), 1529.0053, 0.17800000, Decimal("1.78")),
        exit_price=tp1_price,
        expected_remaining_qty=expected_after_tp1_core,
    )
    # After TP1: exit_notional records core reduction from entry to after_tp1
    tp1_reduction = 0.59482721 - expected_after_tp1_core
    expected_tp1_exit = tp1_reduction * tp1_price
    assert state.position_cost_exit_notional == pytest.approx(expected_tp1_exit, rel=1e-7)
    assert state.position_cost_remaining_qty == pytest.approx(expected_after_tp1_core, rel=1e-7)

    # TP2: position further reduced
    tp2_price = 1607.9163
    state.position_cost_remaining_qty = expected_after_tp1_core  # simulate state after TP1
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), 1529.0053, 0.17800000, Decimal("1.78")),
        exit_price=tp2_price,
    )
    # TP2 should chain: old_remaining=0.1784, new_remaining=0.178, reduced=0.0004
    # But clamp: total_entry - core.eth = 0.5948 - 0.178 = 0.4168, TP1 already consumed 0.4164
    # reduced_qty = min(0.0004, 0.4168) = 0.0004 (normal chaining, very small because same snapshot)
    tp2_reduction = expected_after_tp1_core - 0.17800000
    expected_tp2_exit = tp2_reduction * tp2_price
    assert state.position_cost_exit_notional == pytest.approx(expected_tp1_exit + expected_tp2_exit, rel=1e-6)


def test_record_core_position_reduction_exit_clamp_fallback_when_qty_zero() -> None:
    """When tracked qty is zero/negative, clamp provides the fallback from total_entry_qty."""
    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=0.59482721,
        position_cost_entry_notional=1361.68678471,
        position_cost_remaining_qty=0.0,  # stale — was already set to 0
        sidecar_legs=[],
    )
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), 1529.0053, 0.17800000, Decimal("1.78")),
        exit_price=1561.5685,
    )
    # qty = 0 - 0.178 = -0.178 -> reduced_qty=0 -> falls through to clamp: 0.4168
    expected_exit_notional = 0.41682721 * 1561.5685
    assert state.position_cost_exit_notional == pytest.approx(expected_exit_notional, rel=1e-7)
    assert state.position_cost_remaining_qty == pytest.approx(0.17800000, rel=1e-7)


# ── Full net_remaining_breakeven validation after TP1 ───────────────────


def test_three_stage_tp1_without_sidecar_remaining_breakeven_reasonable() -> None:
    """After TP1 at a price above avg_entry, net_remaining_breakeven
    must be significantly below avg_entry_price (locked-in profit).

    Construction:
      - core_total_qty = 0.59482721 ETH
      - avg_entry_price = 1529.0053
      - tp1_price = 1561.5685 (higher → profit locked)
      - after TP1: core.eth_qty = 0.17800000 ETH (30% remaining, tp1_ratio=0.7)
    """
    avg_entry = 1529.0053
    core_qty = 0.59482721
    tp1_price = 1561.5685
    remaining_qty = 0.17800000

    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=core_qty,
        position_cost_entry_notional=core_qty * avg_entry,
        position_cost_exit_notional=0.0,
        position_cost_remaining_qty=core_qty,
        sidecar_legs=[],
    )
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), avg_entry, remaining_qty, Decimal("1.78")),
        exit_price=tp1_price,
        fee_buffer_pct=0.001,
    )
    # TP1 increment: (0.59482721 - 0.178) * 1561.5685 ≈ 650.90
    tp1_increment = (core_qty - remaining_qty) * tp1_price
    raw_breakeven = (state.position_cost_entry_notional - tp1_increment) / remaining_qty
    # raw_breakeven ≈ (909.474 - 650.904) / 0.178 ≈ 1452.64
    # buffered = raw * (1 + fee_buffer_pct) = 1452.64 * 1.001 ≈ 1454.10
    assert state.net_remaining_breakeven_price > 0
    assert state.net_remaining_breakeven_price < avg_entry, (
        f"net_remaining_breakeven_price={state.net_remaining_breakeven_price:.4f} "
        f"should be < avg_entry={avg_entry:.4f} because TP1 locked in profit"
    )
    assert state.net_remaining_breakeven_price == pytest.approx(raw_breakeven * 1.001, rel=1e-6)


def test_three_stage_tp1_after_sidecar_realized_keeps_remaining_breakeven_realistic() -> None:
    """TP1 after sidecar already realized: remaining breakeven should fall further.

    Construction:
      - core_total_qty = 0.59482721 ETH
      - sidecar_qty = 0.25 ETH (already filled at ~avg_entry*1.004)
      - avg_entry_price = 1529.0053
      - sidecar_tp_price = avg_entry * 1.004 ≈ 1535.12
      - tp1_price = 1561.5685
      - after TP1: core.eth_qty = 0.17800000 ETH

    Pre-TP1 state:
      - entry_notional = core_qty * avg_entry + sidecar_qty * avg_entry
                        = 0.59482721*1529.0053 + 0.25*1529.0053 ≈ 1291.74
      - exit_notional = sidecar_qty * sidecar_tp_price
                        (sidecar already realized) ≈ 383.78
      - remaining_qty = core_qty (sidecar already closed)

    After TP1:
      - exit_notional += (core_qty - 0.178) * tp1_price ≈ 650.90
      - total_exit_notional ≈ 383.78 + 650.90 = 1034.68
      - remaining_qty = 0.178
      - raw ≈ (1291.74 - 1034.68) / 0.178 ≈ 1444.16
      - buffered ≈ raw * 1.001 ≈ 1445.60
    """
    avg_entry = 1529.0053
    core_qty = 0.59482721
    sidecar_qty = 0.25
    tp1_price = 1561.5685
    remaining_qty = 0.17800000
    sidecar_tp_price = avg_entry * 1.004
    fee_buffer_pct = 0.001

    # Pre-TP1: sidecar already realized
    entry_notional = core_qty * avg_entry + sidecar_qty * avg_entry
    pre_tp1_exit_notional = sidecar_qty * sidecar_tp_price

    state = StrategyPositionState(
        side="LONG",
        total_entry_qty=core_qty,
        position_cost_entry_notional=entry_notional,
        position_cost_exit_notional=pre_tp1_exit_notional,
        position_cost_remaining_qty=core_qty,
        sidecar_legs=[],
    )
    record_core_position_reduction_exit(
        state,
        PositionSnapshot("LONG", Decimal("1.78"), avg_entry, remaining_qty, Decimal("1.78")),
        exit_price=tp1_price,
        fee_buffer_pct=fee_buffer_pct,
    )
    # Verify TP1 increment is correct
    tp1_increment = (core_qty - remaining_qty) * tp1_price
    total_exit_notional = pre_tp1_exit_notional + tp1_increment
    assert state.position_cost_exit_notional == pytest.approx(total_exit_notional, rel=1e-7)
    assert state.position_cost_remaining_qty == pytest.approx(remaining_qty, rel=1e-7)

    # Net breakeven should reflect both sidecar profit and TP1 profit
    raw = (entry_notional - total_exit_notional) / remaining_qty
    expected_buffered = raw * (1.0 + fee_buffer_pct)
    assert state.net_remaining_breakeven_price > 0
    assert state.net_remaining_breakeven_price < avg_entry, (
        f"net_remaining_breakeven_price={state.net_remaining_breakeven_price:.4f} "
        f"should be < avg_entry={avg_entry:.4f} — sidecar + TP1 both locked profit"
    )
    # With sidecar realized, breakeven should be lower than without sidecar
    # (more profit already locked)
    assert state.net_remaining_breakeven_price == pytest.approx(expected_buffered, rel=1e-6)


# ── JSON-safe tests ─────────────────────────────────────────────────────

def test_to_json_safe_converts_decimal() -> None:
    """Verify to_json_safe recursively converts Decimal to float for JSON."""
    import json

    from src.utils import to_json_safe

    # Simple Decimal
    assert to_json_safe(Decimal("1.5")) == 1.5
    assert isinstance(to_json_safe(Decimal("1.5")), float)

    # Dict with nested Decimal
    d = {"a": Decimal("2.5"), "b": "hello", "c": [Decimal("1"), Decimal("2")]}
    result = to_json_safe(d)
    assert result == {"a": 2.5, "b": "hello", "c": [1.0, 2.0]}
    json.dumps(result)  # must not raise

    # List of Decimals
    assert to_json_safe([Decimal("1"), Decimal("2")]) == [1.0, 2.0]

    # Nested structure
    nested = {"outer": {"inner": Decimal("3.14")}}
    assert to_json_safe(nested) == {"outer": {"inner": 3.14}}

    # Non-Decimal pass-through
    assert to_json_safe(42) == 42
    assert to_json_safe(None) is None
    assert to_json_safe("hello") == "hello"
    assert to_json_safe(True) is True
    assert to_json_safe(3.14) == 3.14


def test_three_stage_tp1_payload_is_json_serializable() -> None:
    """Verify typical TP1 payload dicts are JSON-serializable."""
    import json

    # Simulate a payload like three_stage_post_tp1_sl_payload
    payload = {
        "position_id": "test123",
        "side": "LONG",
        "contracts": float(Decimal("1.5")),
        "core_contracts": float(Decimal("1.5")),
        "net_contracts": float(Decimal("1.5")),
        "protective_sl_price": 1535.4725,
        "old_sl_order_id": "algo-old",
        "current_price": 1561.0,
        "current_price_source": "latest_market_price",
        "reason": "three_stage_tp1_filled",
    }
    result = json.dumps(payload)
    assert isinstance(result, str)
    assert '"contracts": 1.5' in result

    # Verify inline GLOBAL_SL payload
    global_sl_payload = {
        "position_id": "test123",
        "core_side": "LONG",
        "core_contracts": float(Decimal("1.5")),
        "net_side": "LONG",
        "net_contracts": float(Decimal("1.5")),
        "trading_halted": True,
        "halt_reason": "three_stage_post_tp1_global_sl_net_position_missing",
        "manual_intervention_required": True,
    }
    result2 = json.dumps(global_sl_payload)
    assert isinstance(result2, str)
