#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_open_orders_trading_client_port_boundaries.py
@Description: Boundary scan — cancel_existing_reduce_only_orders must use
              TradingClientPort exclusively for open orders reads and cancels.
"""

from __future__ import annotations

import ast
import pathlib


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
TP_SL_FILE = HERE.parent.parent / "src" / "execution" / "tp_sl_execution_manager.py"


def _read_source() -> str:
    return TP_SL_FILE.read_text()


def _extract_method_source(source: str, method_name: str) -> str:
    """Extract the source lines of *method_name* from *source* using ast."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                return ast.get_source_segment(source, node)
    raise AssertionError(f"Method {method_name!r} not found in source")


# ===================================================================
# Tests
# ===================================================================


class TestCancelExistingReduceOnlyOrdersBoundary:
    """Method-level boundary scan for cancel_existing_reduce_only_orders."""

    @staticmethod
    def _method_source() -> str:
        source = _read_source()
        return _extract_method_source(source, "cancel_existing_reduce_only_orders")

    # ------------------------------------------------------------------
    # MUST contain
    # ------------------------------------------------------------------

    def test_contains_fetch_open_orders(self) -> None:
        src = self._method_source()
        assert "self.trading_client.fetch_open_orders(" in src, (
            "cancel_existing_reduce_only_orders must use "
            "self.trading_client.fetch_open_orders()"
        )

    def test_contains_trading_client_cancel_order(self) -> None:
        src = self._method_source()
        assert "self.trading_client.cancel_order(" in src, (
            "cancel_existing_reduce_only_orders must cancel via "
            "self.trading_client.cancel_order()"
        )

    # ------------------------------------------------------------------
    # MUST NOT contain
    # ------------------------------------------------------------------

    def test_does_not_contain_fetch_broker_open_orders(self) -> None:
        src = self._method_source()
        assert "fetch_broker_open_orders(" not in src, (
            "cancel_existing_reduce_only_orders must not call "
            "fetch_broker_open_orders()"
        )

    def test_does_not_contain_trader_fetch_broker_open_orders(self) -> None:
        src = self._method_source()
        assert "self.trader.fetch_broker_open_orders(" not in src, (
            "cancel_existing_reduce_only_orders must not call "
            "self.trader.fetch_broker_open_orders()"
        )

    def test_does_not_contain_trader_request(self) -> None:
        src = self._method_source()
        assert "self.trader.request(" not in src, (
            "cancel_existing_reduce_only_orders must not call "
            "self.trader.request()"
        )

    def test_does_not_contain_orders_pending(self) -> None:
        src = self._method_source()
        assert '"/api/v5/trade/orders-pending"' not in src, (
            "cancel_existing_reduce_only_orders must not reference "
            "/api/v5/trade/orders-pending"
        )

    def test_does_not_contain_cancel_order_endpoint(self) -> None:
        src = self._method_source()
        assert '"/api/v5/trade/cancel-order"' not in src, (
            "cancel_existing_reduce_only_orders must not reference "
            "/api/v5/trade/cancel-order"
        )


class TestFileLevelNoNewForbiddenImports:
    """Whole-file scan: no forbidden identifiers added.

    NOTE: ``os.getenv`` is already present in the file (inside
    ``_broker_semantic_*`` helpers) so it is NOT checked at file level.
    The file‑level check only covers identifiers that are guaranteed not
    to exist anywhere in the file pre‑change.
    """

    @staticmethod
    def _file_source() -> str:
        return _read_source()

    def test_no_binance(self) -> None:
        src = self._file_source()
        assert "Binance" not in src

    def test_no_exchange_runtime_bundle(self) -> None:
        src = self._file_source()
        assert "ExchangeRuntimeBundle" not in src

    def test_no_broker_semantic_executor(self) -> None:
        src = self._file_source()
        assert "BrokerSemanticExecutor" not in src

    def test_no_three_stage_adapter(self) -> None:
        src = self._file_source()
        assert "ThreeStageAdapter" not in src

    def test_no_middle_runner_adapter(self) -> None:
        src = self._file_source()
        assert "MiddleRunnerAdapter" not in src

    def test_no_sidecar_adapter(self) -> None:
        src = self._file_source()
        assert "SidecarAdapter" not in src

    def test_no_trader_instantiation(self) -> None:
        src = self._file_source()
        assert "Trader()" not in src

    def test_no_okx_private_client_instantiation(self) -> None:
        src = self._file_source()
        assert "OkxPrivateClient()" not in src

    def test_no_load_dotenv(self) -> None:
        src = self._file_source()
        assert "load_dotenv" not in src
