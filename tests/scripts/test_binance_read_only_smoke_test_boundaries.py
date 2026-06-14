#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_read_only_smoke_test_boundaries.py
@Description: Boundary tests — scan the read-only smoke script source for
              required and forbidden tokens/imports.
"""

from __future__ import annotations

from pathlib import Path


def _smoke_script_source() -> str:
    """Read the full source of the read-only smoke script."""
    path = Path("scripts/binance_read_only_smoke_test.py")
    return path.read_text(encoding="utf-8")


# ======================================================================
# Required tokens
# ======================================================================


def test_contains_read_only_confirmation_env() -> None:
    source = _smoke_script_source()
    assert "READ_ONLY_SMOKE_CONFIRM" in source
    # Legacy alias must also still be present.
    assert "BINANCE_READ_ONLY_SMOKE_CONFIRM" in source


def test_contains_read_only_confirmation_value() -> None:
    source = _smoke_script_source()
    assert "I_UNDERSTAND_THIS_READS_EXCHANGE_PRIVATE_ACCOUNT" in source
    # Legacy value must also still be present.
    assert "I_UNDERSTAND_THIS_READS_BINANCE_PRIVATE_ACCOUNT" in source


def test_contains_fetch_position() -> None:
    source = _smoke_script_source()
    assert "fetch_position" in source


def test_contains_fetch_open_orders() -> None:
    source = _smoke_script_source()
    assert "fetch_open_orders" in source


def test_contains_binance_broker_client() -> None:
    source = _smoke_script_source()
    assert "BinanceBrokerClient" in source


def test_contains_aiohttp_binance_transport() -> None:
    source = _smoke_script_source()
    assert "AiohttpBinanceTransport" in source


def test_contains_no_orders_were_placed_message() -> None:
    source = _smoke_script_source()
    assert "no orders were placed" in source


# ======================================================================
# Forbidden tokens — write/order operations
# ======================================================================


FORBIDDEN_WRITE_TOKENS = [
    "place_order",
    "cancel_order",
    "BrokerOrderRequest",
    "BrokerOrderType.MARKET",
    "BrokerOrderType.LIMIT",
    "BrokerOrderType.STOP_MARKET",
    "TAKE_PROFIT",
    "CHANGE_LEVERAGE",
    "change_leverage",
]


def test_does_not_contain_place_order() -> None:
    source = _smoke_script_source()
    assert "place_order" not in source


def test_does_not_contain_cancel_order() -> None:
    source = _smoke_script_source()
    assert "cancel_order" not in source


def test_does_not_contain_broker_order_request() -> None:
    source = _smoke_script_source()
    assert "BrokerOrderRequest" not in source


def test_does_not_contain_broker_order_type_market() -> None:
    source = _smoke_script_source()
    assert "BrokerOrderType.MARKET" not in source


def test_does_not_contain_broker_order_type_limit() -> None:
    source = _smoke_script_source()
    assert "BrokerOrderType.LIMIT" not in source


def test_does_not_contain_broker_order_type_stop_market() -> None:
    source = _smoke_script_source()
    assert "BrokerOrderType.STOP_MARKET" not in source


def test_does_not_contain_take_profit() -> None:
    source = _smoke_script_source()
    assert "TAKE_PROFIT" not in source


def test_does_not_contain_stop_market() -> None:
    source = _smoke_script_source()
    # "STOP_MARKET" appears inside BrokerOrderType fully qualified
    # or as a standalone token; we check standalone usage.
    assert "STOP_MARKET" not in source


def test_does_not_contain_change_leverage() -> None:
    source = _smoke_script_source()
    assert "CHANGE_LEVERAGE" not in source
    assert "change_leverage" not in source


# ======================================================================
# Forbidden tokens — Binance endpoints
# ======================================================================


def test_does_not_contain_fapi_order_path() -> None:
    source = _smoke_script_source()
    assert "/fapi/v1/order" not in source


def test_does_not_contain_fapi_leverage_path() -> None:
    source = _smoke_script_source()
    assert "/fapi/v1/leverage" not in source


def test_does_not_contain_fapi_algo_order_path() -> None:
    source = _smoke_script_source()
    assert "/fapi/v1/algoOrder" not in source


def test_does_not_contain_fapi_open_algo_orders_path() -> None:
    source = _smoke_script_source()
    assert "/fapi/v1/openAlgoOrders" not in source


# ======================================================================
# Forbidden tokens — live smoke confirmation (must be independent)
# ======================================================================


def test_does_not_contain_live_smoke_test_confirm_env() -> None:
    source = _smoke_script_source()
    assert "LIVE_SMOKE_TEST_CONFIRM" not in source
    # Neither the primary nor the alias form should appear.
    assert "BINANCE_LIVE_SMOKE_TEST_CONFIRM" not in source


def test_does_not_contain_i_understand_this_places_real_orders() -> None:
    source = _smoke_script_source()
    assert "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS" not in source
    assert "I_UNDERSTAND_THIS_PLACES_REAL_EXCHANGE_ORDERS" not in source


# ======================================================================
# Forbidden imports — strategy / execution / live runtime
# ======================================================================


FORBIDDEN_IMPORTS = [
    "src.strategies",
    "src.execution",
    "src.live.workers",
]


def test_does_not_import_src_strategies() -> None:
    source = _smoke_script_source()
    assert "src.strategies" not in source


def test_does_not_import_src_execution() -> None:
    source = _smoke_script_source()
    assert "src.execution" not in source


def test_does_not_import_src_live_workers() -> None:
    source = _smoke_script_source()
    assert "src.live.workers" not in source


# ======================================================================
# Allowed imports — must be present
# ======================================================================


def test_imports_binance_client() -> None:
    source = _smoke_script_source()
    assert "src.exchanges.binance.client" in source


def test_imports_aiohttp_transport() -> None:
    source = _smoke_script_source()
    assert "src.exchanges.binance.aiohttp_transport" in source


def test_imports_runtime_config() -> None:
    source = _smoke_script_source()
    assert "src.exchanges.runtime_config" in source


# ======================================================================
# Forbidden: Binance signing direct usage
# ======================================================================


def test_does_not_import_binance_signing() -> None:
    source = _smoke_script_source()
    assert "src.exchanges.binance.signing" not in source


def test_does_not_import_binance_semantic_executor() -> None:
    source = _smoke_script_source()
    assert "src.exchanges.binance.semantic_executor" not in source
