import pytest

from src.position_management.cost_basis import calculate_remaining_breakeven_price


def test_long_sidecar_tp_lowers_remaining_breakeven() -> None:
    basis = calculate_remaining_breakeven_price(
        side="LONG",
        entry_notional=6925.0,
        exit_notional=1731.9,
        remaining_qty=2.08,
        fee_buffer_pct=0.001,
    )

    raw = (6925.0 - 1731.9) / 2.08
    assert basis.raw_breakeven_price == pytest.approx(raw)
    assert basis.buffered_breakeven_price == pytest.approx(raw * 1.001)
    assert basis.buffered_breakeven_price < 2500.0 * 1.001


def test_short_sidecar_tp_lowers_remaining_breakeven_with_short_fee_buffer() -> None:
    basis = calculate_remaining_breakeven_price(
        side="SHORT",
        entry_notional=6925.0,
        exit_notional=1725.0,
        remaining_qty=2.08,
        fee_buffer_pct=0.001,
    )

    raw = (6925.0 - 1725.0) / 2.08
    assert basis.raw_breakeven_price == pytest.approx(raw)
    assert basis.buffered_breakeven_price == pytest.approx(raw * 0.999)
    assert basis.buffered_breakeven_price > 2500.0 * 0.999


def test_remaining_qty_zero_returns_none_prices() -> None:
    basis = calculate_remaining_breakeven_price(
        side="LONG",
        entry_notional=100.0,
        exit_notional=100.0,
        remaining_qty=0.0,
        fee_buffer_pct=0.001,
    )

    assert basis.raw_breakeven_price is None
    assert basis.buffered_breakeven_price is None
