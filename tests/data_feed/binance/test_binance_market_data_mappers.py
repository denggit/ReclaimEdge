#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_market_data_mappers.py
@Description: Tests for Binance mappers that convert raw Binance data
              to unified CandleSnapshot and MarketTradeSnapshot DTOs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.data_feed.binance.mappers import (
    map_binance_agg_trade_to_market_trade_snapshot,
    map_binance_rest_kline_to_candle_snapshot,
)
from src.data_feed.market_data_client_port import (
    CandleSnapshot,
    MarketTradeSnapshot,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_kline_row(
    open_time: int = 1710000000000,
    open_p: str = "3100.00",
    high: str = "3130.00",
    low: str = "3090.00",
    close: str = "3120.00",
    volume: str = "123.45",
    close_time: int = 1710000899999,
    **extra: object,
) -> list:
    """Build a raw Binance REST kline row matching the /fapi/v1/klines format.

    Binance format: [openTime, open, high, low, close, volume, closeTime, ...]
    """
    return [
        str(open_time),
        open_p,
        high,
        low,
        close,
        volume,
        str(close_time),
    ]


def _make_agg_trade_payload(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "e": "aggTrade",
        "E": 1710000000123,
        "s": "ETHUSDT",
        "a": 5933014,
        "p": "3100.50",
        "q": "1.25",
        "f": 100,
        "l": 105,
        "T": 1710000000111,
        "m": True,
    }
    payload.update(overrides)
    return payload


# ======================================================================
# map_binance_rest_kline_to_candle_snapshot
# ======================================================================


class TestMapBinanceRestKlineToCandleSnapshot:
    def test_returns_candle_snapshot(self) -> None:
        row = _make_kline_row()
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert isinstance(result, CandleSnapshot)

    def test_open_time_ms(self) -> None:
        row = _make_kline_row(open_time=1710000000000)
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.open_time_ms == 1710000000000

    def test_close_time_ms(self) -> None:
        row = _make_kline_row(close_time=1710000899999)
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.close_time_ms == 1710000899999

    def test_open_price_decimal(self) -> None:
        row = _make_kline_row(open_p="3100.00")
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.open_price == Decimal("3100.00")

    def test_high_price_decimal(self) -> None:
        row = _make_kline_row(high="3130.00")
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.high_price == Decimal("3130.00")

    def test_low_price_decimal(self) -> None:
        row = _make_kline_row(low="3090.00")
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.low_price == Decimal("3090.00")

    def test_close_price_decimal(self) -> None:
        row = _make_kline_row(close="3120.00")
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.close_price == Decimal("3120.00")

    def test_volume_decimal(self) -> None:
        row = _make_kline_row(volume="123.45")
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.volume == Decimal("123.45")

    def test_is_closed_when_close_time_in_past(self) -> None:
        row = _make_kline_row(close_time=1000)
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m", now_ms=2000
        )
        assert result.is_closed is True

    def test_is_not_closed_when_close_time_in_future(self) -> None:
        row = _make_kline_row(close_time=2000)
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m", now_ms=1000
        )
        assert result.is_closed is False

    def test_is_closed_default_when_now_ms_not_provided(self) -> None:
        row = _make_kline_row(close_time=1000)
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        # When now_ms is not provided, defaults to close_time_ms → is_closed=True
        assert result.is_closed is True

    def test_raw_contains_symbol_and_interval(self) -> None:
        row = _make_kline_row()
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.raw["symbol"] == "ETHUSDT"
        assert result.raw["interval"] == "15m"

    def test_raw_contains_open_and_close_time(self) -> None:
        row = _make_kline_row(open_time=101, close_time=202)
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.raw["open_time_ms"] == 101
        assert result.raw["close_time_ms"] == 202

    def test_too_short_row_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            map_binance_rest_kline_to_candle_snapshot(
                ["1", "2"], symbol="ETHUSDT", interval="15m"
            )

    def test_numeric_price_as_string(self) -> None:
        row = _make_kline_row(open_p="3100")
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.open_price == Decimal("3100")

    def test_numeric_price_as_float_in_row(self) -> None:
        row = ["1710000000000", 3100.0, 3130.0, 3090.0, 3120.0, "123.45", "1710000899999"]
        result = map_binance_rest_kline_to_candle_snapshot(
            row, symbol="ETHUSDT", interval="15m"
        )
        assert result.open_price == Decimal("3100.0")
        assert result.high_price == Decimal("3130.0")


# ======================================================================
# map_binance_agg_trade_to_market_trade_snapshot
# ======================================================================


class TestMapBinanceAggTradeToMarketTradeSnapshot:
    def test_returns_market_trade_snapshot(self) -> None:
        payload = _make_agg_trade_payload()
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert isinstance(result, MarketTradeSnapshot)

    def test_price_decimal(self) -> None:
        payload = _make_agg_trade_payload(p="3100.50")
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.price == Decimal("3100.50")

    def test_qty_decimal(self) -> None:
        payload = _make_agg_trade_payload(q="1.25")
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.qty == Decimal("1.25")

    def test_event_time_ms(self) -> None:
        payload = _make_agg_trade_payload(E=1710000000123)
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.event_time_ms == 1710000000123

    def test_m_true_maps_to_sell(self) -> None:
        payload = _make_agg_trade_payload(m=True)
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.side == "SELL"

    def test_m_false_maps_to_buy(self) -> None:
        payload = _make_agg_trade_payload(m=False)
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.side == "BUY"

    def test_m_absent_side_none(self) -> None:
        payload = _make_agg_trade_payload()
        del payload["m"]
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.side is None

    def test_m_none_side_none(self) -> None:
        payload = _make_agg_trade_payload(m=None)
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.side is None

    def test_m_not_bool_side_none(self) -> None:
        payload = _make_agg_trade_payload(m="not_a_bool")
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.side is None

    def test_raw_preserved(self) -> None:
        payload = _make_agg_trade_payload()
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.raw == payload
        assert result.raw["e"] == "aggTrade"

    def test_price_as_integer(self) -> None:
        payload = _make_agg_trade_payload(p=310050)
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.price == Decimal("310050")

    def test_qty_as_integer(self) -> None:
        payload = _make_agg_trade_payload(q=125)
        result = map_binance_agg_trade_to_market_trade_snapshot(payload)
        assert result.qty == Decimal("125")

    def test_missing_E_raises(self) -> None:
        payload = _make_agg_trade_payload()
        del payload["E"]
        with pytest.raises(ValueError, match="missing E"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_missing_p_raises(self) -> None:
        payload = _make_agg_trade_payload()
        del payload["p"]
        with pytest.raises(ValueError, match="missing p"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_missing_q_raises(self) -> None:
        payload = _make_agg_trade_payload()
        del payload["q"]
        with pytest.raises(ValueError, match="missing q"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_invalid_E_raises(self) -> None:
        payload = _make_agg_trade_payload(E="not_a_number")
        with pytest.raises(ValueError, match="invalid int E"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_invalid_p_raises(self) -> None:
        payload = _make_agg_trade_payload(p="not_a_number")
        with pytest.raises(ValueError, match="invalid decimal p"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_invalid_q_raises(self) -> None:
        payload = _make_agg_trade_payload(q="not_a_number")
        with pytest.raises(ValueError, match="invalid decimal q"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_empty_E_raises(self) -> None:
        payload = _make_agg_trade_payload(E="")
        with pytest.raises(ValueError, match="missing E"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)

    def test_none_E_raises(self) -> None:
        payload = _make_agg_trade_payload(E=None)
        with pytest.raises(ValueError, match="missing E"):
            map_binance_agg_trade_to_market_trade_snapshot(payload)
