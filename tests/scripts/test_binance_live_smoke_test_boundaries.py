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
        # Live runtime imports are forbidden, but binance_live_preflight is
        # explicitly allowed — it is a no-network safety gate.
        "from src.live.binance_market_data_bridge",
        "from src.live.binance_signal_only_runtime",
        "from src.live.live_runtime_selector",
        "from src.live.workers",
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


CONFIRM_ENV = "LIVE_SMOKE_TEST_CONFIRM"


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


# ---------------------------------------------------------------------------
# 20B-FIX-4: Algo Order API + clientOrderId length boundaries
# ---------------------------------------------------------------------------


def test_stop_market_not_sent_via_broker_client_place_order() -> None:
    """STOP_MARKET must NOT appear as a type passed to BinanceBrokerClient.place_order.

    The script should use the Algo Order API, not the regular order endpoint.
    """
    text = _read_smoke_test_text()

    # "STOP_MARKET" may appear in comments / docstrings / type hints,
    # but must NOT appear in a call to client.place_order.
    # We check that there is no pattern like:
    #   order_type=BrokerOrderType.STOP_MARKET
    #   followed later by client.place_order(request)
    # A simpler check: STOP_MARKET should only appear in comments/docstrings
    # and in the algo order params, not in _make_order_request calls.
    lines = text.splitlines()
    stop_market_in_code = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
            continue
        if "STOP_MARKET" in stripped:
            stop_market_in_code.append((i + 1, stripped))

    # STOP_MARKET should only appear in:
    # 1. Comments / docstrings
    # 2. The algo order params (type=STOP_MARKET in the raw dict)
    # It should NOT be in _make_order_request calls (via BrokerOrderType.STOP_MARKET)
    for lineno, code in stop_market_in_code:
        if "BrokerOrderType.STOP_MARKET" in code:
            assert False, (
                f"Line {lineno}: STOP_MARKET via BrokerOrderType found — "
                f"this should use algo order API instead: {code.strip()}"
            )


def test_client_order_id_length_check() -> None:
    """Script must enforce clientOrderId length <= 36."""
    text = _read_smoke_test_text()
    # The generator function must have the 36-char limit
    assert "[:36]" in text or "<= 36" in text or "len(cid) <= 36" in text
    assert "short_label" in text or "short_labels" in text


def test_smoke_test_contains_algo_order_endpoint() -> None:
    """Script must reference /fapi/v1/algoOrder."""
    text = _read_smoke_test_text()
    assert "/fapi/v1/algoOrder" in text


def test_smoke_test_contains_algo_type_conditional() -> None:
    """Algo order must set algoType=CONDITIONAL."""
    text = _read_smoke_test_text()
    assert '"CONDITIONAL"' in text or "'CONDITIONAL'" in text


def test_smoke_test_contains_client_algo_id_param() -> None:
    """Algo order must use clientAlgoId parameter name."""
    text = _read_smoke_test_text()
    assert "clientAlgoId" in text


def test_cleanup_contains_fallback_close_without_client_order_id() -> None:
    """cleanup must contain fallback close without clientOrderId."""
    text = _read_smoke_test_text()
    assert "fallback close" in text or "fallback" in text
    # The fallback close should pass client_order_id=None
    assert "client_order_id=None" in text


def test_cleanup_contains_cancel_smoke_algo_orders() -> None:
    """cleanup must attempt to cancel algo smoke orders."""
    text = _read_smoke_test_text()
    assert "cancel_smoke_algo_orders" in text


def test_smoke_test_does_not_send_position_side() -> None:
    """Script must not set positionSide in order params (One-way mode)."""
    text = _read_smoke_test_text()
    # positionSide should not appear in order params dicts
    # (it's only used in BrokerOrderRequest model with NET value)
    assert '"positionSide"' not in text
    assert "'positionSide'" not in text


def test_no_position_side_in_algo_params() -> None:
    """Algo order params must not include positionSide (One-way mode uses BOTH default).

    Some string appearances of 'positionSide' are fine (e.g. in the
    /fapi/v1/positionSide/dual path).  We only assert that positionSide is
    NOT set as a literal dict key like '"positionSide"' or "'positionSide'"
    in the code (which would indicate it's being passed as a parameter).
    """
    text = _read_smoke_test_text()
    # Check for positionSide used as a dict key (quoted string key)
    assert '"positionSide"' not in text, "algo params must not contain positionSide key"
    assert "'positionSide'" not in text, "algo params must not contain positionSide key"


# ---------------------------------------------------------------------------
# 20C-4C-PREP: Harden live smoke test safety gates
# ---------------------------------------------------------------------------


def test_smoke_test_contains_preflight_import() -> None:
    """Script must import from src.exchanges.binance.live_preflight."""
    text = _read_smoke_test_text()
    assert "src.exchanges.binance.live_preflight" in text


def test_smoke_test_contains_live_confirmation() -> None:
    """Script must reference LIVE_CONFIRMATION env var."""
    text = _read_smoke_test_text()
    assert "LIVE_CONFIRMATION" in text


def test_smoke_test_contains_allow_set_leverage_env() -> None:
    """Script must reference LIVE_SMOKE_TEST_ALLOW_SET_LEVERAGE."""
    text = _read_smoke_test_text()
    assert "LIVE_SMOKE_TEST_ALLOW_SET_LEVERAGE" in text


def test_smoke_test_contains_allow_set_leverage_value() -> None:
    """Script must reference I_UNDERSTAND_THIS_CHANGES_EXCHANGE_LEVERAGE."""
    text = _read_smoke_test_text()
    assert "I_UNDERSTAND_THIS_CHANGES_EXCHANGE_LEVERAGE" in text


def test_smoke_test_contains_require_no_existing_position() -> None:
    """Script must contain require_no_existing_position."""
    text = _read_smoke_test_text()
    assert "require_no_existing_position" in text


def test_smoke_test_contains_require_requested_notional_cap() -> None:
    """Script must contain require_requested_notional_cap."""
    text = _read_smoke_test_text()
    assert "require_requested_notional_cap" in text


def test_smoke_test_contains_require_calculated_notional_cap() -> None:
    """Script must contain require_calculated_notional_cap."""
    text = _read_smoke_test_text()
    assert "require_calculated_notional_cap" in text


def test_smoke_test_contains_require_binance_live_preflight_for_smoke() -> None:
    """Script must contain require_binance_live_preflight_for_smoke."""
    text = _read_smoke_test_text()
    assert "require_binance_live_preflight_for_smoke" in text


def test_smoke_test_contains_require_existing_leverage() -> None:
    """Script must contain require_existing_leverage."""
    text = _read_smoke_test_text()
    assert "require_existing_leverage" in text


def test_smoke_test_contains_allow_set_leverage() -> None:
    """Script must contain allow_set_leverage function."""
    text = _read_smoke_test_text()
    assert "allow_set_leverage" in text


def test_smoke_test_no_btc_or_spot() -> None:
    """Script must not reference BTC or spot trading."""
    text = _read_smoke_test_text()
    assert "BTCUSDT" not in text
    assert "BTC-USDT" not in text
    assert "BrokerMarketType.SPOT" not in text
    assert 'market_type="SPOT"' not in text
    assert "market_type='SPOT'" not in text


def test_smoke_test_still_does_not_import_strategy() -> None:
    """Script must not import strategy modules."""
    text = _read_smoke_test_text()
    assert "src.strategies" not in text


def test_smoke_test_still_does_not_import_execution() -> None:
    """Script must not import execution modules."""
    text = _read_smoke_test_text()
    assert "src.execution" not in text


def test_smoke_test_still_does_not_import_live_workers() -> None:
    """Script must not import live workers."""
    text = _read_smoke_test_text()
    assert "src.live.workers" not in text


def test_smoke_test_still_does_not_import_run_boll() -> None:
    """Script must not import run_boll_cvd_live."""
    text = _read_smoke_test_text()
    assert "scripts.run_boll_cvd_live" not in text
    assert "run_boll_cvd_live" not in text


def test_smoke_test_does_not_contain_position_side_dict_key() -> None:
    """Script must not send positionSide as a dict key (One-way mode)."""
    text = _read_smoke_test_text()
    assert '"positionSide"' not in text
    assert "'positionSide'" not in text


# ---------------------------------------------------------------------------
# 20C-4C-PREP-FIX: Enforce live smoke position notional cap
# ---------------------------------------------------------------------------


def test_smoke_test_does_not_contain_hard_max_position_notional() -> None:
    """Script must NOT reference BINANCE_LIVE_HARD_MAX_POSITION_NOTIONAL_USDT (removed)."""
    text = _read_smoke_test_text()
    assert "BINANCE_LIVE_HARD_MAX_POSITION_NOTIONAL_USDT" not in text


def test_smoke_test_contains_preflight_max_position() -> None:
    """Script must compute preflight_max_position from preflight config."""
    text = _read_smoke_test_text()
    assert "preflight_max_position" in text


def test_smoke_test_contains_max_position_notional() -> None:
    """Script must reference max_position_notional_usdt from preflight config."""
    text = _read_smoke_test_text()
    assert "max_position_notional_usdt" in text
