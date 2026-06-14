#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_market_data_client_port_boundaries.py
@Description: Boundary tests for MarketDataClientPort — the port must NOT
              import concrete exchanges or specific symbols.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

# ======================================================================
# Paths
# ======================================================================

_PORT_PATH = Path(__file__).resolve().parents[2] / "src" / "data_feed" / "market_data_client_port.py"


def _read_port_text() -> str:
    return _PORT_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_port_file_exists() -> None:
    assert _PORT_PATH.exists(), f"MarketDataClientPort file not found at {_PORT_PATH}"
    assert _PORT_PATH.is_file()


def test_port_file_compiles() -> None:
    text = _read_port_text()
    compile(text, str(_PORT_PATH), "exec")


# ======================================================================
# Import-ability
# ======================================================================


def test_port_can_be_imported() -> None:
    from src.data_feed.market_data_client_port import MarketDataClientPort  # noqa: F401


def test_dtos_can_be_imported() -> None:
    from src.data_feed.market_data_client_port import (
        CandleSnapshot,
        MarketDataEvent,
        MarketTradeSnapshot,
    )  # noqa: F401


# ======================================================================
# Forbidden tokens — no concrete exchange references
# ======================================================================


class TestNoConcreteExchangeReferences:
    """The port source must NOT reference any concrete exchange."""

    def test_no_okx_reference(self) -> None:
        text = _read_port_text()
        assert "OKX" not in text
        assert "okx" not in text

    def test_no_binance_reference(self) -> None:
        text = _read_port_text()
        assert "Binance" not in text
        assert "binance" not in text

    def test_no_ethusdt_symbol(self) -> None:
        text = _read_port_text()
        assert "ETH-USDT-SWAP" not in text
        assert "ETHUSDT" not in text

    def test_no_api_paths(self) -> None:
        text = _read_port_text()
        assert "/api/v5" not in text
        assert "/fapi" not in text

    def test_no_binance_feed_import(self) -> None:
        text = _read_port_text()
        assert "src.data_feed.binance" not in text

    def test_no_okx_feed_import(self) -> None:
        text = _read_port_text()
        assert "src.data_feed.okx" not in text

    def test_no_exchange_import(self) -> None:
        text = _read_port_text()
        assert "src.exchanges" not in text


# ======================================================================
# DTO behaviour
# ======================================================================


class TestDtoBehaviour:
    """DTOs must behave as expected."""

    def test_candle_snapshot_defaults(self) -> None:
        from src.data_feed.market_data_client_port import CandleSnapshot

        c = CandleSnapshot(
            open_time_ms=1000,
            close_time_ms=2000,
            open_price=Decimal("3000"),
            high_price=Decimal("3100"),
            low_price=Decimal("2900"),
            close_price=Decimal("3050"),
            volume=Decimal("100.5"),
            is_closed=True,
        )
        assert c.open_time_ms == 1000
        assert c.is_closed is True
        assert c.raw == {}

    def test_market_trade_snapshot_defaults(self) -> None:
        from src.data_feed.market_data_client_port import MarketTradeSnapshot

        t = MarketTradeSnapshot(
            event_time_ms=5000,
            price=Decimal("3000.5"),
            qty=Decimal("2.0"),
        )
        assert t.side is None
        assert t.raw == {}

    def test_dtos_are_frozen(self) -> None:
        from src.data_feed.market_data_client_port import CandleSnapshot

        c = CandleSnapshot(
            open_time_ms=0,
            close_time_ms=0,
            open_price=Decimal("0"),
            high_price=Decimal("0"),
            low_price=Decimal("0"),
            close_price=Decimal("0"),
            volume=Decimal("0"),
            is_closed=False,
        )
        with pytest.raises(Exception):
            c.close_price = Decimal("100")  # type: ignore[misc]

    def test_market_data_event_union(self) -> None:
        from src.data_feed.market_data_client_port import (
            CandleSnapshot,
            MarketDataEvent,
            MarketTradeSnapshot,
        )

        candle = CandleSnapshot(
            open_time_ms=0,
            close_time_ms=0,
            open_price=Decimal("0"),
            high_price=Decimal("0"),
            low_price=Decimal("0"),
            close_price=Decimal("0"),
            volume=Decimal("0"),
            is_closed=False,
        )
        trade = MarketTradeSnapshot(event_time_ms=0, price=Decimal("0"), qty=Decimal("0"))
        # Both should be assignable to MarketDataEvent
        _ev1: MarketDataEvent = candle
        _ev2: MarketDataEvent = trade


# Local import for mypy's sake
import pytest  # noqa: E402
