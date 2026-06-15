from __future__ import annotations

import pytest

from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


def test_default_leverage_is_20x() -> None:
    cfg = SimplePositionSizerConfig()
    assert cfg.leverage == 20.0


def test_calculate_uses_stop_distance_to_cap_risk() -> None:
    sizer = SimplePositionSizer(
        SimplePositionSizerConfig(
            dry_run_equity_usdt=10_000,
            leverage=20,
            trade_risk_pct=0.01,
            fee_slippage_buffer_pct=0.001,
        )
    )

    size = sizer.calculate(price=1800, stop_price=1782)

    assert size.sizing_mode == "risk"
    assert size.risk_usdt == pytest.approx(100.0)
    assert size.stop_distance_pct == pytest.approx(0.01)
    assert size.effective_risk_pct == pytest.approx(0.011)
    assert size.notional_usdt == pytest.approx(100 / 0.011)
    assert size.margin_usdt == pytest.approx((100 / 0.011) / 20)
    assert size.eth_qty == pytest.approx((100 / 0.011) / 1800)


def test_legacy_margin_sizing_remains_fallback_without_stop() -> None:
    sizer = SimplePositionSizer(
        SimplePositionSizerConfig(
            dry_run_equity_usdt=1_000,
            layer_margin_pct=0.03,
            leverage=20,
        )
    )

    size = sizer.calculate(price=2000)

    assert size.sizing_mode == "margin"
    assert size.margin_usdt == pytest.approx(30.0)
    assert size.notional_usdt == pytest.approx(600.0)
