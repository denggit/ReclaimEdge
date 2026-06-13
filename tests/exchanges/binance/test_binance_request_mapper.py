#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_request_mapper.py
@Description: Unit tests for Binance request mapper (One-way / net mode).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.request_mapper import (
    BINANCE_ETH_CONTRACT_SIZE_BASE,
    _format_decimal,
    _normalize_binance_position_mode,
    broker_order_request_to_binance_params,
    broker_order_side_to_binance,
    broker_position_side_to_binance,
    broker_quantity_to_binance_base_quantity,
)
from src.exchanges.models import (
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _request(**kwargs) -> BrokerOrderRequest:
    defaults = dict(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        side=BrokerOrderSide.BUY,
        position_side=BrokerPositionSide.NET,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("0.1"),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
    )
    defaults.update(kwargs)
    return BrokerOrderRequest(**defaults)


# ---------------------------------------------------------------------------
# 1. MARKET + BASE_ASSET quantity direct
# ---------------------------------------------------------------------------


def test_market_base_asset_quantity_direct() -> None:
    req = _request(
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("0.5"),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.5"
    assert "positionSide" not in params
    assert "reduceOnly" not in params
    assert "price" not in params
    assert "stopPrice" not in params
    assert "timeInForce" not in params


# ---------------------------------------------------------------------------
# 2. MARKET + CONTRACTS converts 2 contracts -> 0.2 ETH
# ---------------------------------------------------------------------------


def test_market_contracts_converts_to_base_quantity() -> None:
    req = _request(
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["quantity"] == "0.2"
    assert params["type"] == "MARKET"
    assert "positionSide" not in params


def test_broker_quantity_to_binance_base_quantity_contracts() -> None:
    result = broker_quantity_to_binance_base_quantity(
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )
    assert result == Decimal("0.2")


def test_broker_quantity_to_binance_base_quantity_base_asset() -> None:
    result = broker_quantity_to_binance_base_quantity(
        quantity=Decimal("1.5"),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
    )
    assert result == Decimal("1.5")


# ---------------------------------------------------------------------------
# 3. LIMIT requires price and adds price/timeInForce=GTC
# ---------------------------------------------------------------------------


def test_limit_order_requires_price_and_adds_time_in_force_gtc() -> None:
    req = _request(
        order_type=BrokerOrderType.LIMIT,
        price=Decimal("3100.50"),
        quantity=Decimal("0.3"),
    )

    params = broker_order_request_to_binance_params(req)

    assert params["type"] == "LIMIT"
    assert params["price"] == "3100.5"
    assert params["timeInForce"] == "GTC"
    assert "positionSide" not in params


def test_limit_order_without_price_raises_value_error() -> None:
    req = _request(
        order_type=BrokerOrderType.LIMIT,
        price=None,
    )

    with pytest.raises(ValueError, match="LIMIT order requires price"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 4. STOP_MARKET requires trigger_price and adds stopPrice (no positionSide)
# ---------------------------------------------------------------------------


def test_stop_market_requires_trigger_price_and_adds_stop_price() -> None:
    req = _request(
        order_type=BrokerOrderType.STOP_MARKET,
        side=BrokerOrderSide.SELL,
        trigger_price=Decimal("2900.00"),
    )

    params = broker_order_request_to_binance_params(req)

    assert params["type"] == "STOP_MARKET"
    assert params["stopPrice"] == "2900"
    assert "positionSide" not in params
    assert "price" not in params
    assert "timeInForce" not in params


def test_stop_market_without_trigger_price_raises_value_error() -> None:
    req = _request(
        order_type=BrokerOrderType.STOP_MARKET,
        trigger_price=None,
    )

    with pytest.raises(ValueError, match="STOP_MARKET order requires trigger_price"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 5. TAKE_PROFIT_MARKET requires trigger_price and adds stopPrice (no positionSide)
# ---------------------------------------------------------------------------


def test_take_profit_market_requires_trigger_price_and_adds_stop_price() -> None:
    req = _request(
        order_type=BrokerOrderType.TAKE_PROFIT_MARKET,
        side=BrokerOrderSide.SELL,
        trigger_price=Decimal("3200.00"),
    )

    params = broker_order_request_to_binance_params(req)

    assert params["type"] == "TAKE_PROFIT_MARKET"
    assert params["stopPrice"] == "3200"
    assert "positionSide" not in params
    assert "price" not in params
    assert "timeInForce" not in params


def test_take_profit_market_without_trigger_price_raises_value_error() -> None:
    req = _request(
        order_type=BrokerOrderType.TAKE_PROFIT_MARKET,
        trigger_price=None,
    )

    with pytest.raises(ValueError, match="TAKE_PROFIT_MARKET order requires trigger_price"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 6. client_order_id -> newClientOrderId
# ---------------------------------------------------------------------------


def test_client_order_id_maps_to_new_client_order_id() -> None:
    req = _request(client_order_id="my-custom-id-001")

    params = broker_order_request_to_binance_params(req)

    assert params["newClientOrderId"] == "my-custom-id-001"


def test_no_client_order_id_omits_new_client_order_id() -> None:
    req = _request(client_order_id=None)

    params = broker_order_request_to_binance_params(req)

    assert "newClientOrderId" not in params


# ---------------------------------------------------------------------------
# 7. reduce_only=True emits reduceOnly="true" in One-way mode
# ---------------------------------------------------------------------------


def test_reduce_only_true_emits_reduce_only_in_net_mode() -> None:
    req = _request(
        side=BrokerOrderSide.SELL,
        reduce_only=True,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["reduceOnly"] == "true"
    assert params["side"] == "SELL"
    assert "positionSide" not in params


def test_reduce_only_false_does_not_emit_reduce_only() -> None:
    req = _request(reduce_only=False)

    params = broker_order_request_to_binance_params(req)

    assert "reduceOnly" not in params


# ---------------------------------------------------------------------------
# 8. QUOTE_ASSET rejects
# ---------------------------------------------------------------------------


def test_quote_asset_quantity_unit_rejects() -> None:
    req = _request(quantity_unit=BrokerQuantityUnit.QUOTE_ASSET)

    with pytest.raises(ValueError, match="Unsupported Binance quantity unit"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 9. quantity <= 0 rejects
# ---------------------------------------------------------------------------


def test_quantity_zero_rejects() -> None:
    req = _request(quantity=Decimal("0"))

    with pytest.raises(ValueError, match="quantity must be positive"):
        broker_order_request_to_binance_params(req)


def test_quantity_negative_rejects() -> None:
    req = _request(quantity=Decimal("-0.5"))

    with pytest.raises(ValueError, match="quantity must be positive"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 10. non-BINANCE request rejects
# ---------------------------------------------------------------------------


def test_non_binance_request_rejects() -> None:
    req = _request(exchange=ExchangeName.OKX)

    with pytest.raises(ValueError, match="BrokerOrderRequest.exchange must be BINANCE"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 11. non-ETHUSDT symbol rejects
# ---------------------------------------------------------------------------


def test_non_ethusdt_symbol_rejects() -> None:
    req = _request(symbol="BTCUSDT")

    with pytest.raises(ValueError, match="Unsupported Binance symbol"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 12. broker_position_side_to_binance always raises in One-way mode
# ---------------------------------------------------------------------------


def test_broker_position_side_to_binance_rejects_all_values() -> None:
    with pytest.raises(ValueError, match="One-way mode does not use positionSide"):
        broker_position_side_to_binance(BrokerPositionSide.LONG)

    with pytest.raises(ValueError, match="One-way mode does not use positionSide"):
        broker_position_side_to_binance(BrokerPositionSide.SHORT)

    with pytest.raises(ValueError, match="One-way mode does not use positionSide"):
        broker_position_side_to_binance(BrokerPositionSide.NET)


# ---------------------------------------------------------------------------
# 13. unsupported order type rejects
# ---------------------------------------------------------------------------


def test_unsupported_order_type_rejects() -> None:
    req = _request(order_type=BrokerOrderType.UNKNOWN)

    with pytest.raises(ValueError, match="Unsupported Binance order type"):
        broker_order_request_to_binance_params(req)


# ---------------------------------------------------------------------------
# 14. position_mode validation (net / one_way / one-way accepted; hedge rejected)
# ---------------------------------------------------------------------------


def test_position_mode_net_works() -> None:
    req = _request()
    params = broker_order_request_to_binance_params(req, position_mode="net")
    assert "positionSide" not in params


def test_position_mode_one_way_works() -> None:
    req = _request()
    params = broker_order_request_to_binance_params(req, position_mode="one_way")
    assert "positionSide" not in params


def test_position_mode_oneway_works() -> None:
    req = _request()
    params = broker_order_request_to_binance_params(req, position_mode="oneway")
    assert "positionSide" not in params


def test_position_mode_one_way_dash_works() -> None:
    req = _request()
    params = broker_order_request_to_binance_params(req, position_mode="one-way")
    assert "positionSide" not in params


def test_position_mode_hedge_rejects() -> None:
    req = _request()
    with pytest.raises(ValueError, match="Unsupported Binance position mode"):
        broker_order_request_to_binance_params(req, position_mode="hedge")


def test_position_mode_dual_rejects() -> None:
    req = _request()
    with pytest.raises(ValueError, match="Unsupported Binance position mode"):
        broker_order_request_to_binance_params(req, position_mode="dual")


# ---------------------------------------------------------------------------
# 15. _normalize_binance_position_mode
# ---------------------------------------------------------------------------


class TestNormalizePositionMode:
    def test_net(self) -> None:
        assert _normalize_binance_position_mode("net") == "net"

    def test_one_way(self) -> None:
        assert _normalize_binance_position_mode("one_way") == "net"

    def test_oneway(self) -> None:
        assert _normalize_binance_position_mode("oneway") == "net"

    def test_one_dash_way(self) -> None:
        assert _normalize_binance_position_mode("one-way") == "net"

    def test_uppercase(self) -> None:
        assert _normalize_binance_position_mode("NET") == "net"

    def test_hedge_rejects(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Binance position mode"):
            _normalize_binance_position_mode("hedge")

    def test_dual_rejects(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Binance position mode"):
            _normalize_binance_position_mode("dual")

    def test_dual_side_rejects(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Binance position mode"):
            _normalize_binance_position_mode("dual_side")


# ---------------------------------------------------------------------------
# 16. Decimal formatting tests
# ---------------------------------------------------------------------------


def test_format_decimal_trailing_zeros() -> None:
    assert _format_decimal(Decimal("0.100")) == "0.1"


def test_format_decimal_integer() -> None:
    assert _format_decimal(Decimal("1.000")) == "1"


def test_format_decimal_with_digits() -> None:
    assert _format_decimal(Decimal("3100.50")) == "3100.5"


def test_format_decimal_zero() -> None:
    assert _format_decimal(Decimal("0")) == "0"


def test_format_decimal_large_number() -> None:
    assert _format_decimal(Decimal("1000.00")) == "1000"


# ---------------------------------------------------------------------------
# 17. contract size constant
# ---------------------------------------------------------------------------


def test_contract_size_is_0_1_eth() -> None:
    assert BINANCE_ETH_CONTRACT_SIZE_BASE == Decimal("0.1")


# ---------------------------------------------------------------------------
# 18. full param shape for open long buy market (One-way)
# ---------------------------------------------------------------------------


def test_full_open_long_buy_market_params() -> None:
    req = _request(
        side=BrokerOrderSide.BUY,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("1.0"),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        client_order_id="long-open-001",
    )

    params = broker_order_request_to_binance_params(req)

    assert params == {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "type": "MARKET",
        "quantity": "1",
        "newClientOrderId": "long-open-001",
    }


# ---------------------------------------------------------------------------
# 19. TP long (SELL LIMIT reduceOnly="true")
# ---------------------------------------------------------------------------


def test_tp_long_sell_limit_reduce_only() -> None:
    req = _request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.LIMIT,
        quantity=Decimal("0.5"),
        price=Decimal("3200"),
        reduce_only=True,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["side"] == "SELL"
    assert params["type"] == "LIMIT"
    assert params["price"] == "3200"
    assert params["timeInForce"] == "GTC"
    assert params["reduceOnly"] == "true"
    assert "positionSide" not in params


# ---------------------------------------------------------------------------
# 20. SL long (SELL STOP_MARKET reduceOnly="true")
# ---------------------------------------------------------------------------


def test_sl_long_sell_stop_market_reduce_only() -> None:
    req = _request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.STOP_MARKET,
        quantity=Decimal("0.5"),
        trigger_price=Decimal("2900"),
        reduce_only=True,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["side"] == "SELL"
    assert params["type"] == "STOP_MARKET"
    assert params["stopPrice"] == "2900"
    assert params["reduceOnly"] == "true"
    assert "positionSide" not in params


# ---------------------------------------------------------------------------
# 21. market close long (SELL MARKET reduceOnly="true")
# ---------------------------------------------------------------------------


def test_close_long_via_sell_market_reduce_only() -> None:
    req = _request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("0.5"),
        reduce_only=True,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["side"] == "SELL"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.5"
    assert params["reduceOnly"] == "true"
    assert "positionSide" not in params


# ---------------------------------------------------------------------------
# 22. open short (SELL, no reduceOnly)
# ---------------------------------------------------------------------------


def test_open_short_sell_no_reduce_only() -> None:
    req = _request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("0.2"),
        reduce_only=False,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["side"] == "SELL"
    assert params["type"] == "MARKET"
    assert "reduceOnly" not in params
    assert "positionSide" not in params


# ---------------------------------------------------------------------------
# 23. short TP (BUY LIMIT reduceOnly="true")
# ---------------------------------------------------------------------------


def test_short_tp_buy_limit_reduce_only() -> None:
    req = _request(
        side=BrokerOrderSide.BUY,
        order_type=BrokerOrderType.LIMIT,
        quantity=Decimal("0.3"),
        price=Decimal("2800"),
        reduce_only=True,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["side"] == "BUY"
    assert params["type"] == "LIMIT"
    assert params["price"] == "2800"
    assert params["reduceOnly"] == "true"
    assert "positionSide" not in params


# ---------------------------------------------------------------------------
# 24. short SL (BUY STOP_MARKET reduceOnly="true")
# ---------------------------------------------------------------------------


def test_short_sl_buy_stop_market_reduce_only() -> None:
    req = _request(
        side=BrokerOrderSide.BUY,
        order_type=BrokerOrderType.STOP_MARKET,
        quantity=Decimal("0.3"),
        trigger_price=Decimal("3200"),
        reduce_only=True,
    )

    params = broker_order_request_to_binance_params(req)

    assert params["side"] == "BUY"
    assert params["type"] == "STOP_MARKET"
    assert params["stopPrice"] == "3200"
    assert params["reduceOnly"] == "true"
    assert "positionSide" not in params
