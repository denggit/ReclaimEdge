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
    # SHORT sidecar fixed TP buys back below entry, lowering the remaining cost basis.
    # entry=2500, total_qty=2.77, sidecar_qty=0.69, sidecar TP=0.4% → TP price=2490.
    entry_price = 2500.0
    total_qty = 2.77
    sidecar_qty = 0.69
    remaining_qty = 2.08
    sidecar_tp_price = entry_price * (1 - 0.004)

    entry_notional = entry_price * total_qty
    exit_notional = sidecar_qty * sidecar_tp_price

    basis = calculate_remaining_breakeven_price(
        side="SHORT",
        entry_notional=entry_notional,
        exit_notional=exit_notional,
        remaining_qty=remaining_qty,
        fee_buffer_pct=0.001,
    )

    raw = (entry_notional - exit_notional) / remaining_qty
    assert basis.raw_breakeven_price == pytest.approx(raw)
    assert basis.buffered_breakeven_price == pytest.approx(raw * 0.999)
    assert basis.buffered_breakeven_price > entry_price * 0.999


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
