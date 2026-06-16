from __future__ import annotations

import os

import pytest

from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


def test_default_leverage_is_20x() -> None:
    cfg = SimplePositionSizerConfig()
    assert cfg.leverage == 20.0


def test_default_trade_risk_pct_is_003() -> None:
    cfg = SimplePositionSizerConfig()
    assert cfg.trade_risk_pct == 0.003


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


def test_default_003_pct_risk_sizing() -> None:
    """With default trade_risk_pct=0.003 (0.3%), verify the risk-based formula.
    equity=10_000; risk_usdt = 10_000 * 0.003 = 30;
    stop_distance_pct = (1800-1782)/1800 = 0.01;
    effective_risk_pct = 0.01 + 0.001 = 0.011;
    notional = 30 / 0.011; margin = notional / 20; eth_qty = notional / 1800
    """
    sizer = SimplePositionSizer(
        SimplePositionSizerConfig(
            dry_run_equity_usdt=10_000,
            leverage=20,
            trade_risk_pct=0.003,
            fee_slippage_buffer_pct=0.001,
        )
    )

    size = sizer.calculate(price=1800, stop_price=1782)

    assert size.sizing_mode == "risk"
    assert size.risk_usdt == pytest.approx(30.0)
    assert size.stop_distance_pct == pytest.approx(0.01)
    assert size.effective_risk_pct == pytest.approx(0.011)
    assert size.notional_usdt == pytest.approx(30 / 0.011)
    assert size.margin_usdt == pytest.approx((30 / 0.011) / 20)
    assert size.eth_qty == pytest.approx((30 / 0.011) / 1800)


def test_env_trade_risk_pct_overrides_default() -> None:
    """TRADE_RISK_PCT=0.01 should override the default 0.003."""
    os.environ["TRADE_RISK_PCT"] = "0.01"
    try:
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.01
    finally:
        del os.environ["TRADE_RISK_PCT"]


def test_env_entry_risk_pct_overrides_default() -> None:
    """ENTRY_RISK_PCT=0.005 should override the default 0.003."""
    os.environ["ENTRY_RISK_PCT"] = "0.005"
    try:
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.005
    finally:
        del os.environ["ENTRY_RISK_PCT"]


def test_env_trade_risk_pct_priority_over_entry_risk_pct() -> None:
    """TRADE_RISK_PCT takes priority over ENTRY_RISK_PCT when both are set."""
    os.environ["TRADE_RISK_PCT"] = "0.02"
    os.environ["ENTRY_RISK_PCT"] = "0.005"
    try:
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.02
    finally:
        del os.environ["TRADE_RISK_PCT"]
        del os.environ["ENTRY_RISK_PCT"]


def test_env_entry_risk_pct_fallback_when_trade_risk_pct_absent() -> None:
    """When only ENTRY_RISK_PCT is set, it is used as the fallback."""
    # ensure TRADE_RISK_PCT is not set
    os.environ.pop("TRADE_RISK_PCT", None)
    os.environ["ENTRY_RISK_PCT"] = "0.005"
    try:
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.005
    finally:
        del os.environ["ENTRY_RISK_PCT"]


# ── ENTRY_MAX_STOP_DISTANCE_PCT default guard ─────────────────────────


def test_entry_max_stop_distance_pct_default_is_0012() -> None:
    """Default config must use 0.012 (1.2%) as the safety ceiling."""
    from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig

    cfg = BollCvdReclaimStrategyConfig()
    assert cfg.entry_max_stop_distance_pct == pytest.approx(0.012)


def test_entry_max_stop_distance_pct_env_override() -> None:
    """ENTRY_MAX_STOP_DISTANCE_PCT=0.02 should override the default 0.012."""
    from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig

    os.environ["ENTRY_MAX_STOP_DISTANCE_PCT"] = "0.02"
    try:
        cfg = BollCvdReclaimStrategyConfig.from_env()
        assert cfg.entry_max_stop_distance_pct == pytest.approx(0.02)
    finally:
        del os.environ["ENTRY_MAX_STOP_DISTANCE_PCT"]


def test_entry_max_stop_distance_pct_explicit_zero_disables_guard() -> None:
    """ENTRY_MAX_STOP_DISTANCE_PCT=0 can still explicitly disable the ceiling."""
    from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig

    os.environ["ENTRY_MAX_STOP_DISTANCE_PCT"] = "0"
    try:
        cfg = BollCvdReclaimStrategyConfig.from_env()
        assert cfg.entry_max_stop_distance_pct == 0.0
    finally:
        del os.environ["ENTRY_MAX_STOP_DISTANCE_PCT"]


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
