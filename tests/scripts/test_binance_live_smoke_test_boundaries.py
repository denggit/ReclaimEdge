#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_live_smoke_test_boundaries.py
@Description: Boundary tests — the smoke test must not import strategy,
              CVD, live loop, OKX, or other forbidden modules.
"""

from __future__ import annotations

from pathlib import Path


_SMOKE_TEST_PATH = Path(__file__).resolve().parents[2] / "scripts" / "binance_live_smoke_test.py"


def _read_smoke_test_text() -> str:
    return _SMOKE_TEST_PATH.read_text(encoding="utf-8")


def test_smoke_test_file_exists() -> None:
    assert _SMOKE_TEST_PATH.exists(), f"Smoke test script not found at {_SMOKE_TEST_PATH}"
    assert _SMOKE_TEST_PATH.is_file()


def test_smoke_test_does_not_import_strategy_or_live_loop() -> None:
    text = _read_smoke_test_text()

    forbidden = [
        "src.strategies",
        "CvdTracker",
        "BollBandBreakoutMonitor",
        "ExecutionCommandProcessor",
        "scripts.run_boll_cvd_live",
        "OKX_CONFIG",
        "ETH-USDT-SWAP",
        "BTCUSDT",
        "BTC-USDT",
        "from src.live",
        "from src.position_management",
        "from src.risk",
        "from src.reporting",
        "from src.account_sync",
        "from src.startup_recovery",
        "from src.execution.trader",
        "from src.execution.tp_sl",
        "from src.data_feed",
        "from config",
    ]

    for token in forbidden:
        assert token not in text, f"'{token}' must not appear in smoke test script"


def test_smoke_test_allows_binance_and_models_imports() -> None:
    text = _read_smoke_test_text()

    allowed = [
        "BinanceBrokerClient",
        "AiohttpBinanceTransport",
        "BinanceTransportResponse",
        "ExchangeName",
        "BrokerOrderRequest",
        "BrokerOrderSide",
        "BrokerOrderType",
        "BrokerPositionSide",
        "BrokerQuantityUnit",
        "build_signed_request",
    ]

    for token in allowed:
        assert token in text, f"'{token}' should appear in smoke test script"


def test_smoke_test_has_ethusdt_only() -> None:
    text = _read_smoke_test_text()

    # Must contain ETHUSDT
    assert '"ETHUSDT"' in text or "'ETHUSDT'" in text or 'BINANCE_SYMBOL' in text

    # Must NOT reference BTC
    assert "BTCUSDT" not in text
    assert "BTC-USDT" not in text


def test_smoke_test_has_client_order_id_prefix() -> None:
    text = _read_smoke_test_text()
    assert "RE_SMOKE_" in text


def test_smoke_test_has_cleanup_function() -> None:
    text = _read_smoke_test_text()
    assert "async def cleanup" in text
    assert "cancel_smoke_orders" in text or "cancel" in text


def test_smoke_test_has_safety_gates() -> None:
    text = _read_smoke_test_text()
    assert "require_live_confirmation" in text
    assert "validate_unified_config_for_binance" in text
    assert "load_unified_runtime_config" in text
    assert "require_one_way_position_mode" in text
    assert "require_isolated_margin" in text
    assert CONFIRM_ENV in text


CONFIRM_ENV = "BINANCE_LIVE_SMOKE_TEST_CONFIRM"


def test_smoke_test_uses_unified_runtime_config() -> None:
    """The smoke test must load and validate the unified runtime config."""
    text = _read_smoke_test_text()
    assert "load_unified_runtime_config()" in text
    assert "validate_unified_config_for_binance" in text


def test_smoke_test_does_not_read_okx_legacy_env_vars() -> None:
    """The smoke test must not read any OKX_* legacy env vars."""
    text = _read_smoke_test_text()
    okx_env_vars = [
        "OKX_INST_ID",
        "OKX_BAR",
        "OKX_TD_MODE",
        "OKX_POS_SIDE_MODE",
    ]
    import_lines = [
        line for line in text.splitlines()
        if "OKX_" in line and not line.strip().startswith("#") and "Does NOT" not in line
    ]
    for var in okx_env_vars:
        for line in import_lines:
            assert var not in line, (
                f"OKX_* env var {var} must not be read by smoke test: {line.strip()}"
            )


def test_smoke_test_file_compiles() -> None:
    """Ensure the script compiles without syntax errors."""
    text = _read_smoke_test_text()
    compile(text, str(_SMOKE_TEST_PATH), "exec")


def test_smoke_test_does_not_have_dry_run_path() -> None:
    text = _read_smoke_test_text()
    assert "run_boll_cvd_dry_run" not in text
    assert "dry_run" not in text


def test_smoke_test_does_not_have_hedge_mode_tokens() -> None:
    """Smoke test must not contain hedge-mode tokens."""
    text = _read_smoke_test_text()
    assert "POSITION_SIDE" not in text
    assert "require_hedge_position_mode" not in text
    assert '"hedge"' not in text and "'hedge'" not in text


def test_smoke_test_has_one_way_mode_tokens() -> None:
    """Smoke test must use one-way mode."""
    text = _read_smoke_test_text()
    assert "require_one_way_position_mode" in text
    assert "one-way/net" in text or "One-way" in text or "one_way" in text


def test_smoke_test_has_reduce_only() -> None:
    """Smoke test TP/SL/close must use reduce_only=True."""
    text = _read_smoke_test_text()
    assert "reduce_only=True" in text


# ---------------------------------------------------------------------------
# Leverage-aware margin check boundaries (20B-FIX-2)
# ---------------------------------------------------------------------------


def test_smoke_test_contains_leverage_endpoint() -> None:
    """Script must reference /fapi/v1/leverage."""
    text = _read_smoke_test_text()
    assert "/fapi/v1/leverage" in text


def test_smoke_test_no_longer_uses_notional_balance_check() -> None:
    """The old balance check (available >= notional) must be gone."""
    text = _read_smoke_test_text()
    assert "available_balance < calculated_notional" not in text


def test_smoke_test_no_longer_uses_old_insufficient_balance_message() -> None:
    """The old 'need ≈ {calculated_notional}' error message must be gone."""
    text = _read_smoke_test_text()
    assert "need ≈ {calculated_notional}" not in text


def test_smoke_test_contains_required_margin_with_buffer() -> None:
    """New margin check must reference required_margin_with_buffer."""
    text = _read_smoke_test_text()
    assert "required_margin_with_buffer" in text


def test_smoke_test_contains_margin_buffer_multiplier() -> None:
    """Script must reference margin_buffer_multiplier."""
    text = _read_smoke_test_text()
    assert "margin_buffer_multiplier" in text


def test_smoke_test_contains_set_initial_leverage() -> None:
    """Script must contain set_initial_leverage function."""
    text = _read_smoke_test_text()
    assert "set_initial_leverage" in text


def test_smoke_test_contains_calculate_required_margin_with_buffer() -> None:
    """Script must contain the margin calculation function."""
    text = _read_smoke_test_text()
    assert "calculate_required_margin_with_buffer" in text


def test_smoke_test_still_has_ethusdt_only() -> None:
    """Still only ETHUSDT — no BTC."""
    text = _read_smoke_test_text()
    assert "BTCUSDT" not in text
    assert "BTC-USDT" not in text


def test_smoke_test_still_one_way_mode() -> None:
    """Still One-way / net mode, not hedge."""
    text = _read_smoke_test_text()
    assert "require_one_way_position_mode" in text
    assert "Hedge Mode" in text or "one-way/net" in text or "One-way" in text


def test_smoke_test_still_isolated() -> None:
    """Still isolated margin mode."""
    text = _read_smoke_test_text()
    assert "require_isolated_margin" in text
    assert "isolated" in text
