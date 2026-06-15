from __future__ import annotations

import pytest

from src.position_management.sidecar.core_exit_safety import (
    SidecarCoreExitRisk,
    active_sidecar_tp_order_ids,
    classify_sidecar_core_final_exit_risk,
    core_tp_is_loss_for_position,
    open_sidecar_legs,
    sidecar_core_exit_client_order_id,
    sidecar_leg_tp_is_beyond_core_exit,
)
from src.position_management.sidecar.model import SidecarLegStatus


# ── helpers ──────────────────────────────────────────────────────────────

def _open_leg(leg_id: str = "leg-1", tp_price: float = 105.0) -> dict:
    return {
        "leg_id": leg_id,
        "status": SidecarLegStatus.OPEN.value,
        "tp_price": tp_price,
        "qty": 0.1,
        "contracts": "1",
        "entry_price": 3000.0,
        "side": "LONG",
        "layer_index": 1,
        "tp_pct": 0.004,
        "margin_pct": 0.01,
        "layer_multiplier": 1.0,
        "position_id": "pos-1",
        "created_ts_ms": 1000,
        "updated_ts_ms": 1000,
    }


def _open_unprotected_leg(leg_id: str = "leg-2", tp_price: float = 106.0) -> dict:
    leg = _open_leg(leg_id, tp_price)
    leg["status"] = SidecarLegStatus.OPEN_UNPROTECTED.value
    leg["tp_order_id"] = None
    return leg


def _filled_leg(leg_id: str = "leg-3") -> dict:
    return {
        "leg_id": leg_id,
        "status": SidecarLegStatus.TP_FILLED.value,
        "tp_price": 105.0,
        "qty": 0.1,
        "contracts": "1",
        "entry_price": 3000.0,
        "side": "LONG",
        "layer_index": 1,
        "tp_pct": 0.004,
        "margin_pct": 0.01,
        "layer_multiplier": 1.0,
        "position_id": "pos-1",
        "created_ts_ms": 1000,
        "updated_ts_ms": 1000,
    }


# ── test: open_sidecar_legs ──────────────────────────────────────────────

def test_open_sidecar_legs_returns_only_open_and_open_unprotected() -> None:
    legs = [
        _open_leg("open-1"),
        _open_unprotected_leg("open-unprotected"),
        _filled_leg("filled"),
    ]
    result = open_sidecar_legs(legs)
    ids = {leg["leg_id"] for leg in result}
    assert ids == {"open-1", "open-unprotected"}


def test_open_sidecar_legs_empty_list() -> None:
    assert open_sidecar_legs([]) == []


# ── test: core_tp_is_loss_for_position ───────────────────────────────────

def test_long_core_tp_below_breakeven_is_loss() -> None:
    assert core_tp_is_loss_for_position("LONG", 99.0, 100.0) is True


def test_long_core_tp_equal_breakeven_is_loss() -> None:
    assert core_tp_is_loss_for_position("LONG", 100.0, 100.0) is True


def test_long_core_tp_above_breakeven_is_not_loss() -> None:
    assert core_tp_is_loss_for_position("LONG", 101.0, 100.0) is False


def test_short_core_tp_above_breakeven_is_loss() -> None:
    assert core_tp_is_loss_for_position("SHORT", 101.0, 100.0) is True


def test_short_core_tp_equal_breakeven_is_loss() -> None:
    assert core_tp_is_loss_for_position("SHORT", 100.0, 100.0) is True


def test_short_core_tp_below_breakeven_is_not_loss() -> None:
    assert core_tp_is_loss_for_position("SHORT", 99.0, 100.0) is False


def test_breakeven_none_not_loss() -> None:
    assert core_tp_is_loss_for_position("LONG", 99.0, None) is False


def test_breakeven_zero_not_loss() -> None:
    assert core_tp_is_loss_for_position("LONG", 99.0, 0.0) is False


# ── test: sidecar_leg_tp_is_beyond_core_exit ─────────────────────────────

def test_long_sidecar_tp_beyond_core() -> None:
    # LONG: price rises to TP. sidecar tp=106 > core_tp=105 → risky
    assert sidecar_leg_tp_is_beyond_core_exit("LONG", 105.0, 106.0) is True


def test_long_sidecar_tp_before_core() -> None:
    # LONG: sidecar tp=106 < core_tp=107 → safe
    assert sidecar_leg_tp_is_beyond_core_exit("LONG", 107.0, 106.0) is False


def test_long_sidecar_tp_equal_core() -> None:
    assert sidecar_leg_tp_is_beyond_core_exit("LONG", 105.0, 105.0) is False


def test_short_sidecar_tp_beyond_core() -> None:
    # SHORT: price falls to TP. sidecar tp=94 < core_tp=95 → risky
    assert sidecar_leg_tp_is_beyond_core_exit("SHORT", 95.0, 94.0) is True


def test_short_sidecar_tp_before_core() -> None:
    # SHORT: sidecar tp=94 > core_tp=93 → safe
    assert sidecar_leg_tp_is_beyond_core_exit("SHORT", 93.0, 94.0) is False


def test_short_sidecar_tp_equal_core() -> None:
    assert sidecar_leg_tp_is_beyond_core_exit("SHORT", 95.0, 95.0) is False


# ── test: classify_sidecar_core_final_exit_risk ──────────────────────────

# Test 1: LONG loss TP
def test_long_loss_tp_risky() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=99.0,
        breakeven_price=100.0,
        sidecar_legs=[_open_leg("leg-1", 105.0)],
    )
    assert risk.risky is True
    assert risk.reason == "core_tp_loss_vs_breakeven"
    assert "leg-1" in risk.risky_leg_ids


# Test 2: SHORT loss TP
def test_short_loss_tp_risky() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="SHORT",
        core_tp_price=101.0,
        breakeven_price=100.0,
        sidecar_legs=[_open_leg("leg-1", 95.0)],
    )
    assert risk.risky is True
    assert risk.reason == "core_tp_loss_vs_breakeven"
    assert "leg-1" in risk.risky_leg_ids


# Test 3: LONG sidecar TP beyond core
def test_long_sidecar_tp_beyond_core_risky() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=105.0,
        breakeven_price=100.0,
        sidecar_legs=[_open_leg("leg-1", 106.0)],
    )
    assert risk.risky is True
    assert risk.reason == "sidecar_tp_beyond_core_final_exit"
    assert "leg-1" in risk.risky_leg_ids


# Test 4: LONG sidecar TP before core
def test_long_sidecar_tp_before_core_safe() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=107.0,
        breakeven_price=100.0,
        sidecar_legs=[_open_leg("leg-1", 106.0)],
    )
    assert risk.risky is False
    assert risk.reason == "sidecar_tp_reaches_before_or_at_core_exit"


# Test 5: SHORT sidecar TP beyond core
def test_short_sidecar_tp_beyond_core_risky() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="SHORT",
        core_tp_price=95.0,
        breakeven_price=100.0,
        sidecar_legs=[_open_leg("leg-1", 94.0)],
    )
    assert risk.risky is True
    assert risk.reason == "sidecar_tp_beyond_core_final_exit"
    assert "leg-1" in risk.risky_leg_ids


# Test 6: SHORT sidecar TP before core
def test_short_sidecar_tp_before_core_safe() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="SHORT",
        core_tp_price=93.0,
        breakeven_price=100.0,
        sidecar_legs=[_open_leg("leg-1", 94.0)],
    )
    assert risk.risky is False
    assert risk.reason == "sidecar_tp_reaches_before_or_at_core_exit"


# Test 7: no open legs → not risky
def test_no_open_legs_safe() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=99.0,
        breakeven_price=100.0,
        sidecar_legs=[_filled_leg("filled")],
    )
    assert risk.risky is False
    assert risk.reason == "no_open_sidecar_legs"


def test_empty_legs_list_safe() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=99.0,
        breakeven_price=100.0,
        sidecar_legs=[],
    )
    assert risk.risky is False
    assert risk.reason == "no_open_sidecar_legs"


def test_loss_tp_takes_priority_over_beyond() -> None:
    # When core TP is a loss, all open legs are risky regardless of individual tp_prices
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=99.0,
        breakeven_price=100.0,
        sidecar_legs=[
            _open_leg("leg-1", 105.0),
            _open_leg("leg-2", 98.0),
        ],
    )
    assert risk.risky is True
    assert risk.reason == "core_tp_loss_vs_breakeven"
    assert set(risk.risky_leg_ids) == {"leg-1", "leg-2"}


def test_multiple_open_legs_only_some_beyond() -> None:
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=105.0,
        breakeven_price=100.0,
        sidecar_legs=[
            _open_leg("leg-1", 106.0),  # beyond core
            _open_leg("leg-2", 104.0),  # before core
            _open_unprotected_leg("leg-3", 107.0),  # beyond core, unprotected
        ],
    )
    assert risk.risky is True
    assert risk.reason == "sidecar_tp_beyond_core_final_exit"
    assert set(risk.risky_leg_ids) == {"leg-1", "leg-3"}


def test_leg_with_invalid_tp_price_treated_as_risky() -> None:
    leg = _open_leg("leg-bad")
    leg["tp_price"] = None  # type: ignore[assignment]
    risk = classify_sidecar_core_final_exit_risk(
        side="LONG",
        core_tp_price=105.0,
        breakeven_price=100.0,
        sidecar_legs=[leg],
    )
    assert risk.risky is True
    assert "leg-bad" in risk.risky_leg_ids


# ── test: sidecar_core_exit_client_order_id ──────────────────────────────

def test_client_order_id_is_within_32_chars() -> None:
    cid = sidecar_core_exit_client_order_id(
        position_id="pos-test",
        leg_id="leg-1",
        old_tp_order_id="old123",
        ts_ms=5_000,
    )
    assert len(cid) <= 32
    assert cid.startswith("SCE")


def test_different_leg_ids_produce_different_clord_ids() -> None:
    cid1 = sidecar_core_exit_client_order_id(
        position_id="pos-test",
        leg_id="leg-ETHUSDTSWAPSHORT178079055281199a",
        old_tp_order_id="3633692596170973184",
        ts_ms=5_000,
    )
    cid2 = sidecar_core_exit_client_order_id(
        position_id="pos-test",
        leg_id="leg-ETHUSDTSWAPSHORT178079055281199b",
        old_tp_order_id="3633692596170974000",
        ts_ms=5_000,
    )
    assert cid1 != cid2


def test_different_ts_produces_different_clord_ids() -> None:
    cid1 = sidecar_core_exit_client_order_id(
        position_id="pos-test",
        leg_id="leg-1",
        old_tp_order_id="old123",
        ts_ms=5_000,
    )
    cid2 = sidecar_core_exit_client_order_id(
        position_id="pos-test",
        leg_id="leg-1",
        old_tp_order_id="old123",
        ts_ms=6_000,
    )
    assert cid1 != cid2


def test_none_position_id_and_order_id_still_produces_valid_id() -> None:
    cid = sidecar_core_exit_client_order_id(
        position_id=None,
        leg_id="leg-1",
        old_tp_order_id=None,
        ts_ms=5_000,
    )
    assert len(cid) <= 32
    assert cid.startswith("SCE")


def test_similar_leg_ids_but_different_old_tp_produce_different_ids() -> None:
    """Regression: two legs whose leg_id has the same long prefix must not collide."""
    # Simulate the real-world scenario where sanitized leg_ids become identical
    cid1 = sidecar_core_exit_client_order_id(
        position_id="pos-1",
        leg_id="ETHUSDTSWAPSHORT178079055281199f",
        old_tp_order_id="3633692596170973184",
        ts_ms=1780790552811,
    )
    cid2 = sidecar_core_exit_client_order_id(
        position_id="pos-1",
        leg_id="ETHUSDTSWAPSHORT178079055281199f",
        old_tp_order_id="3633692596170974000",
        ts_ms=1780790552811,
    )
    assert cid1 != cid2


def test_client_order_id_only_contains_alphanumeric() -> None:
    cid = sidecar_core_exit_client_order_id(
        position_id="pos|test",
        leg_id="leg-1",
        old_tp_order_id="old|123",
        ts_ms=5_000,
    )
    assert all(c.isalnum() for c in cid)
    assert cid.startswith("SCE")


# ── test: active_sidecar_tp_order_ids ───────────────────────────────────

def test_active_sidecar_tp_order_ids_collects_open_and_open_unprotected() -> None:
    legs = [
        {**_open_leg("leg-1", 105.0), "tp_order_id": "tp-aaa"},
        {**_open_unprotected_leg("leg-2", 106.0), "tp_order_id": "tp-bbb"},
    ]
    result = active_sidecar_tp_order_ids(legs)
    assert result == ("tp-aaa", "tp-bbb")


def test_active_sidecar_tp_order_ids_ignores_closed_missing_and_duplicates() -> None:
    legs = [
        {**_open_leg("leg-1", 105.0), "tp_order_id": "tp-aaa"},
        {**_open_leg("leg-1b", 106.0), "tp_order_id": "tp-aaa"},  # duplicate id
        {"leg_id": "leg-2", "status": "TP_FILLED", "tp_order_id": "tp-filled"},
        {"leg_id": "leg-3", "status": "FORCE_CLOSED", "tp_order_id": "tp-force"},
        {"leg_id": "leg-4", "status": "CANCELLED", "tp_order_id": "tp-canc"},
        {"leg_id": "leg-5", "status": "UNKNOWN_HALTED", "tp_order_id": "tp-halt"},
        {**_open_leg("leg-6", 107.0), "tp_order_id": None},
        {**_open_leg("leg-7", 108.0), "tp_order_id": ""},
        {**_open_leg("leg-8", 109.0)},  # no tp_order_id key
    ]
    result = active_sidecar_tp_order_ids(legs)
    # Only "tp-aaa" from the two OPEN legs (deduped), all other statuses ignored
    assert result == ("tp-aaa",)


def test_active_sidecar_tp_order_ids_empty_and_no_matches() -> None:
    assert active_sidecar_tp_order_ids([]) == ()
    assert active_sidecar_tp_order_ids(
        [{"leg_id": "x", "status": "TP_FILLED", "tp_order_id": "x"}]
    ) == ()
