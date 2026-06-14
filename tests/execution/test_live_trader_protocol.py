#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_live_trader_protocol.py
@Description: Tests for LiveTraderProtocol — structural checks, no runtime side-effects.
"""

from __future__ import annotations

from pathlib import Path

from src.execution.live_trader_protocol import LiveTraderProtocol
from src.execution.trader import Trader


# ======================================================================
# Structural satisfaction
# ======================================================================


class TestTraderSatisfiesProtocol:
    """Existing Trader class structurally satisfies LiveTraderProtocol."""

    _REQUIRED_ATTRS: tuple[str, ...] = (
        "symbol",
        "account_equity_usdt",
        "position_contracts",
        "contract_multiplier",
        "contract_precision",
        "min_contracts",
        "leverage",
        "broker_exchange_name",
    )

    _REQUIRED_METHODS: tuple[str, ...] = (
        "start",
        "close",
        "initialize",
        "fetch_position_snapshot",
        "fetch_usdt_equity",
        "execute_intent",
        "fetch_broker_open_orders",
        "fetch_broker_algo_orders",
        "recover_broker_open_orders",
        "fetch_broker_position",
        "fetch_broker_open_order_raws",
        "fetch_broker_algo_order_raws",
        "recover_broker_open_order_raws",
    )

    def test_trader_init_sets_required_attributes(self) -> None:
        """Trader.__init__ assigns all required instance attributes."""
        import inspect
        source = inspect.getsource(Trader.__init__)
        for attr in self._REQUIRED_ATTRS:
            # broker_exchange_name is a @property, not set in __init__
            if attr == "broker_exchange_name":
                continue
            assert f"self.{attr}" in source, \
                f"Trader.__init__ missing assignment: self.{attr}"

    def test_trader_has_broker_exchange_name_property(self) -> None:
        """Trader has broker_exchange_name as a property."""
        import inspect
        assert hasattr(Trader, "broker_exchange_name")
        assert isinstance(
            inspect.getattr_static(Trader, "broker_exchange_name"), property
        )

    def test_trader_has_required_methods(self) -> None:
        """Trader class has all methods required by the protocol."""
        for method in self._REQUIRED_METHODS:
            assert hasattr(Trader, method), f"Trader missing method: {method}"

    def test_protocol_is_runtime_checkable(self) -> None:
        """The protocol is decorated with @runtime_checkable."""
        assert getattr(LiveTraderProtocol, "_is_runtime_protocol", False), \
            "LiveTraderProtocol must be decorated with @runtime_checkable"


# ======================================================================
# Source-level safety checks
# ======================================================================


class TestProtocolFileSafety:
    """The protocol file must not import exchange clients or secrets."""

    _PROTOCOL_PATH: Path = Path("src/execution/live_trader_protocol.py")

    def test_protocol_file_has_no_exchange_client_imports(self) -> None:
        text = self._PROTOCOL_PATH.read_text()
        assert "src.exchanges.binance" not in text
        assert "src.exchanges.okx" not in text
        assert "OkxPrivateClient" not in text
        assert "BinanceBrokerClient" not in text

    def test_protocol_file_has_no_api_secret_strings(self) -> None:
        text = self._PROTOCOL_PATH.read_text()
        assert "api_key" not in text.lower()
        assert "secret_key" not in text.lower()

    def test_protocol_file_has_no_order_methods(self) -> None:
        text = self._PROTOCOL_PATH.read_text()
        assert "place_order" not in text
        assert "cancel_order" not in text
