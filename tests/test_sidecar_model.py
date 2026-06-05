from decimal import Decimal

import pytest

from src.execution.trader import PositionSnapshot
from src.position_management.sidecar.model import (
    SidecarLegStatus,
    calculate_core_margin_pct,
    calculate_sidecar_margin,
    calculate_sidecar_qty,
    calculate_sidecar_tp_price,
    sanitize_okx_client_order_id,
    sidecar_open_contracts,
    sidecar_open_qty,
    trim_sidecar_legs_for_state,
)
from src.position_management.sidecar.planner import sidecar_client_order_id
from src.position_management.sidecar.reconciler import build_core_position_view
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


def test_sidecar_disabled_keeps_core_margin_pct_and_sizing() -> None:
    config = SimplePositionSizerConfig(dry_run_equity_usdt=1000, layer_margin_pct=0.04, leverage=50, sidecar_enabled=False)
    sizer = SimplePositionSizer(config)

    size = sizer.calculate(100.0, layer_index=1)

    assert calculate_core_margin_pct(0.04, False, 0.01) == 0.04
    assert size.margin_usdt == pytest.approx(40.0)
    assert size.notional_usdt == pytest.approx(2000.0)
    assert size.eth_qty == pytest.approx(20.0)


def test_sidecar_enabled_core_and_sidecar_sizing() -> None:
    config = SimplePositionSizerConfig(
        dry_run_equity_usdt=1000,
        layer_margin_pct=0.04,
        leverage=50,
        layer_multiplier_step=0.15,
        sidecar_enabled=True,
        sidecar_margin_pct=0.01,
    )
    sizer = SimplePositionSizer(config)

    core = sizer.calculate(100.0, layer_index=2)
    sidecar_qty = calculate_sidecar_qty(
        account_equity_usdt=1000,
        price=100.0,
        leverage=50,
        layer_margin_pct=0.04,
        sidecar_margin_pct=0.01,
        layer_multiplier=1.15,
    )

    assert config.core_margin_pct == pytest.approx(0.03)
    assert core.margin_usdt == pytest.approx(1000 * 0.03 * 1.15)
    assert calculate_sidecar_margin(0.04, 0.01, 1.15) == pytest.approx(0.0115)
    assert sidecar_qty == pytest.approx((1000 * 0.01 * 1.15 * 50) / 100.0)


def test_sidecar_tp_price() -> None:
    assert calculate_sidecar_tp_price("LONG", 3000, 0.004) == pytest.approx(3012)
    assert calculate_sidecar_tp_price("SHORT", 3000, 0.004) == pytest.approx(2988)


def test_trim_and_open_qty_keep_small_state() -> None:
    legs = [
        {"leg_id": "1", "qty": 1, "status": SidecarLegStatus.TP_FILLED.value, "created_ts_ms": 1, "updated_ts_ms": 1},
        {"leg_id": "2", "qty": 2, "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 2, "updated_ts_ms": 2},
        {"leg_id": "3", "qty": 3, "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 3, "updated_ts_ms": 3},
    ]

    assert sidecar_open_qty(legs) == pytest.approx(5.0)
    assert [leg["leg_id"] for leg in trim_sidecar_legs_for_state(legs, 2)] == ["2", "3"]


def test_open_unprotected_counts_as_sidecar_exposure() -> None:
    legs = [
        {"leg_id": "1", "qty": 0.69, "contracts": "0.69", "status": SidecarLegStatus.OPEN_UNPROTECTED.value, "created_ts_ms": 1, "updated_ts_ms": 1},
        {"leg_id": "2", "qty": 2, "contracts": "2", "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 2, "updated_ts_ms": 2},
    ]

    assert sidecar_open_qty(legs) == pytest.approx(2.69)
    assert sidecar_open_contracts(legs) == Decimal("2.69")
    assert [leg["leg_id"] for leg in trim_sidecar_legs_for_state(legs, 1)] == ["1", "2"]


def test_sidecar_client_order_id_is_okx_safe_for_live_failure_sample() -> None:
    got = sidecar_client_order_id(
        position_id="ETH-USDT-SWAP:LONG:1780610847229:348f76f9",
        layer_index=1,
        ts_ms=1780610847229,
    )

    assert got.startswith("SC")
    assert got.isalnum()
    assert len(got) <= 32
    assert "-" not in got
    assert ":" not in got
    assert "_" not in got


def test_sanitize_okx_client_order_id_drops_special_chars_and_allows_empty() -> None:
    assert sanitize_okx_client_order_id("SC-97644895de-L1-47229") == "SC97644895deL147229"
    assert sanitize_okx_client_order_id("___---:::") == ""


def test_trim_never_drops_open_legs_even_over_limit() -> None:
    legs = [
        {"leg_id": "1", "qty": 1, "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 1, "updated_ts_ms": 1},
        {"leg_id": "2", "qty": 1, "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 2, "updated_ts_ms": 2},
        {"leg_id": "3", "qty": 1, "status": SidecarLegStatus.OPEN.value, "created_ts_ms": 3, "updated_ts_ms": 3},
    ]

    trimmed = trim_sidecar_legs_for_state(legs, 2)

    assert [leg["leg_id"] for leg in trimmed] == ["1", "2", "3"]


def test_build_core_position_view_subtracts_sidecar() -> None:
    okx = PositionSnapshot("LONG", Decimal("12"), 3000.0, 1.2, Decimal("12"))

    core = build_core_position_view(okx, 0.2, Decimal("2"))

    assert core.side == "LONG"
    assert core.contracts == Decimal("10")
    assert core.eth_qty == pytest.approx(1.0)


def test_build_core_position_view_subtracts_open_unprotected_sidecar() -> None:
    okx = PositionSnapshot("LONG", Decimal("2.77"), 3000.0, 2.77, Decimal("2.77"))
    legs = [
        {"leg_id": "sc", "qty": 0.69, "contracts": "0.69", "status": SidecarLegStatus.OPEN_UNPROTECTED.value},
    ]

    core = build_core_position_view(okx, sidecar_open_qty(legs), sidecar_open_contracts(legs))

    assert core.contracts == Decimal("2.08")
    assert core.eth_qty == pytest.approx(2.08)
