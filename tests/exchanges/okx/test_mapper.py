"""Tests for src.exchanges.okx.mapper — pure mapping functions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerTimeInForce,
    ExchangeName,
)
from src.exchanges.okx.mapper import (
    broker_instrument_from_trader,
    broker_order_from_okx_pending_order,
    broker_position_from_snapshot,
    parse_okx_swap_symbol,
    unsupported_okx_order_request_error,
)


# ---------------------------------------------------------------------------
# Fake types for mapper tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeMetadata:
    contract_multiplier: Decimal = Decimal("0.1")
    contract_precision: Decimal = Decimal("0.01")
    min_contracts: Decimal = Decimal("0.01")


class _FakeTrader:
    symbol: str = "ETH-USDT-SWAP"
    instrument_metadata = _FakeMetadata()
    tick_size: Decimal = Decimal("0.01")


@dataclass(frozen=True)
class _FakePositionSnapshot:
    side: str | None
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal


# ---------------------------------------------------------------------------
# parse_okx_swap_symbol
# ---------------------------------------------------------------------------

class TestParseOkxSwapSymbol:
    def test_eth_usdt_swap(self):
        assert parse_okx_swap_symbol("ETH-USDT-SWAP") == ("ETH", "USDT")

    def test_btc_usdt_swap(self):
        assert parse_okx_swap_symbol("BTC-USDT-SWAP") == ("BTC", "USDT")

    def test_lowercase(self):
        assert parse_okx_swap_symbol("eth-usdt-swap") == ("ETH", "USDT")

    def test_no_swap_suffix(self):
        assert parse_okx_swap_symbol("ETH-USDT") == ("ETH-USDT", "USDT")

    def test_empty_string(self):
        assert parse_okx_swap_symbol("") == ("", "USDT")

    def test_malformed_single_hyphen(self):
        assert parse_okx_swap_symbol("ETH-SWAP") == ("ETH-SWAP", "USDT")

    def test_malformed_extra_parts(self):
        result = parse_okx_swap_symbol("ETH-USDT-SWAP-EXTRA")
        assert result[1] == "USDT"


# ---------------------------------------------------------------------------
# broker_instrument_from_trader
# ---------------------------------------------------------------------------

class TestBrokerInstrumentFromTrader:
    def test_basic_mapping(self):
        trader = _FakeTrader()
        inst = broker_instrument_from_trader(trader)
        assert inst.exchange == ExchangeName.OKX
        assert inst.symbol == "ETH-USDT-SWAP"
        assert inst.base_asset == "ETH"
        assert inst.quote_asset == "USDT"
        assert inst.contract_type == "SWAP"
        assert inst.margin_asset == "USDT"

    def test_contract_multiplier(self):
        trader = _FakeTrader()
        trader.instrument_metadata = _FakeMetadata(contract_multiplier=Decimal("0.1"))
        inst = broker_instrument_from_trader(trader)
        assert inst.contract_size == Decimal("0.1")

    def test_contract_precision_as_qty_step(self):
        trader = _FakeTrader()
        trader.instrument_metadata = _FakeMetadata(contract_precision=Decimal("0.01"))
        inst = broker_instrument_from_trader(trader)
        assert inst.qty_step == Decimal("0.01")

    def test_min_contracts(self):
        trader = _FakeTrader()
        trader.instrument_metadata = _FakeMetadata(min_contracts=Decimal("0.02"))
        inst = broker_instrument_from_trader(trader)
        assert inst.min_qty == Decimal("0.02")

    def test_min_notional_is_zero(self):
        trader = _FakeTrader()
        inst = broker_instrument_from_trader(trader)
        assert inst.min_notional == Decimal("0")

    def test_price_tick_from_tick_size(self):
        trader = _FakeTrader()
        trader.tick_size = Decimal("0.05")
        inst = broker_instrument_from_trader(trader)
        assert inst.price_tick == Decimal("0.05")

    def test_price_tick_default(self):
        trader = _FakeTrader()
        object.__setattr__(trader, "tick_size", None)
        inst = broker_instrument_from_trader(trader)
        assert inst.price_tick == Decimal("0.01")

    def test_price_tick_from_float_converts_to_decimal(self):
        trader = _FakeTrader()
        trader.tick_size = 0.5  # type: ignore[assignment]
        inst = broker_instrument_from_trader(trader)
        assert inst.price_tick == Decimal("0.5")


# ---------------------------------------------------------------------------
# broker_position_from_snapshot
# ---------------------------------------------------------------------------


class TestBrokerPositionFromSnapshot:
    @staticmethod
    def _snap(
        side: str | None,
        contracts: Decimal = Decimal("0"),
        avg_entry: float = 0.0,
        eth_qty: float = 0.0,
        raw_pos: Decimal | None = None,
    ) -> _FakePositionSnapshot:
        if raw_pos is None:
            raw_pos = contracts
        return _FakePositionSnapshot(
            side=side,
            contracts=contracts,
            avg_entry_price=avg_entry,
            eth_qty=eth_qty,
            raw_pos=raw_pos,
        )

    def test_long_snapshot(self):
        snap = self._snap("LONG", Decimal("1.5"), 3400.0, 0.15)
        pos = broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            snapshot=snap,
        )
        assert pos.side == BrokerPositionSide.LONG
        assert pos.contracts == Decimal("1.5")
        assert pos.base_qty == Decimal("0.15")
        assert pos.avg_entry_price == Decimal("3400.0")
        assert pos.has_position is True

    def test_short_snapshot(self):
        snap = self._snap("SHORT", Decimal("2.0"), 3500.0, 0.2)
        pos = broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            snapshot=snap,
        )
        assert pos.side == BrokerPositionSide.SHORT
        assert pos.contracts == Decimal("2.0")

    def test_flat_no_requested_side(self):
        snap = self._snap(None, Decimal("0"), 0.0, 0.0)
        pos = broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            snapshot=snap,
        )
        assert pos.side == BrokerPositionSide.NET
        assert pos.contracts == Decimal("0")
        assert pos.has_position is False

    def test_flat_with_requested_side_long(self):
        snap = self._snap(None, Decimal("0"), 0.0, 0.0)
        pos = broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            snapshot=snap,
            requested_side=BrokerPositionSide.LONG,
        )
        assert pos.side == BrokerPositionSide.LONG
        assert pos.contracts == Decimal("0")
        assert pos.has_position is False

    def test_requested_side_mismatch_snapshot_long_request_short(self):
        """When requesting SHORT but snapshot is LONG → return flat SHORT."""
        snap = self._snap("LONG", Decimal("1.0"), 3400.0, 0.1)
        pos = broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            snapshot=snap,
            requested_side=BrokerPositionSide.SHORT,
        )
        assert pos.side == BrokerPositionSide.SHORT
        assert pos.contracts == Decimal("0")
        assert pos.has_position is False

    def test_requested_side_mismatch_snapshot_short_request_long(self):
        snap = self._snap("SHORT", Decimal("1.0"), 3400.0, 0.1)
        pos = broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            snapshot=snap,
            requested_side=BrokerPositionSide.LONG,
        )
        assert pos.side == BrokerPositionSide.LONG
        assert pos.contracts == Decimal("0")
        assert pos.has_position is False


# ---------------------------------------------------------------------------
# broker_order_from_okx_pending_order
# ---------------------------------------------------------------------------


class TestBrokerOrderFromPendingOrder:
    def test_live_limit_sell_reduce_only(self):
        item = {
            "ordId": "123456",
            "clOrdId": "client-abc",
            "side": "sell",
            "posSide": "long",
            "ordType": "limit",
            "state": "live",
            "px": "3500.00",
            "sz": "1.5",
            "accFillSz": "0.5",
            "reduceOnly": "true",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_id == "123456"
        assert order.client_order_id == "client-abc"
        assert order.side == BrokerOrderSide.SELL
        assert order.position_side == BrokerPositionSide.LONG
        assert order.order_type == BrokerOrderType.LIMIT
        assert order.status == BrokerOrderStatus.NEW
        assert order.price == Decimal("3500.00")
        assert order.quantity == Decimal("1.5")
        assert order.filled_quantity == Decimal("0.5")
        assert order.reduce_only is True

    def test_partially_filled(self):
        item = {
            "ordId": "789",
            "side": "buy",
            "posSide": "short",
            "ordType": "limit",
            "state": "partially_filled",
            "px": "3400.00",
            "sz": "2.0",
            "accFillSz": "1.0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.PARTIALLY_FILLED

    def test_filled(self):
        item = {
            "ordId": "900",
            "side": "buy",
            "posSide": "long",
            "ordType": "market",
            "state": "filled",
            "px": "3450.00",
            "sz": "1.0",
            "accFillSz": "1.0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.FILLED
        assert order.order_type == BrokerOrderType.MARKET

    def test_canceled(self):
        item = {
            "ordId": "cancel-1",
            "side": "sell",
            "posSide": "long",
            "ordType": "limit",
            "state": "canceled",
            "px": "3600.00",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.CANCELED

    def test_cancelled_alt_spelling(self):
        item = {
            "ordId": "cancel-2",
            "side": "sell",
            "posSide": "long",
            "ordType": "limit",
            "state": "cancelled",
            "px": "3600.00",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.CANCELED

    def test_rejected(self):
        item = {
            "ordId": "rej-1",
            "side": "buy",
            "posSide": "long",
            "ordType": "limit",
            "state": "rejected",
            "px": "999999.00",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.REJECTED

    def test_conditional_stop_market(self):
        item = {
            "ordId": "cond-1",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
            "state": "live",
            "px": "",
            "sz": "2.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_type == BrokerOrderType.STOP_MARKET

    def test_missing_ord_id_does_not_crash(self):
        item = {
            "side": "buy",
            "ordType": "limit",
            "state": "live",
            "sz": "1.0",
            "accFillSz": "0",
        }
        # Should not raise
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_id == ""
        assert order.status == BrokerOrderStatus.UNKNOWN

    def test_none_values_in_numeric_fields(self):
        item = {
            "ordId": "none-test",
            "side": "buy",
            "posSide": "long",
            "ordType": "limit",
            "state": "live",
            "px": None,
            "sz": None,
            "accFillSz": None,
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.price is None
        assert order.quantity == Decimal("0")
        assert order.filled_quantity == Decimal("0")

    def test_unknown_state(self):
        item = {
            "ordId": "unknown-1",
            "side": "buy",
            "posSide": "long",
            "ordType": "limit",
            "state": "something_strange",
            "px": "100.00",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.UNKNOWN

    def test_unknown_ord_type_falls_back_to_limit(self):
        item = {
            "ordId": "type-1",
            "side": "buy",
            "posSide": "long",
            "ordType": "iceberg",
            "state": "live",
            "px": "100.00",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_type == BrokerOrderType.LIMIT

    def test_raw_is_preserved(self):
        item = {
            "ordId": "raw-1",
            "side": "sell",
            "ordType": "limit",
            "state": "live",
            "sz": "1.0",
            "custom_field": "extra",
        }
        order = broker_order_from_okx_pending_order(item, symbol="ETH-USDT-SWAP")
        assert order.raw.get("custom_field") == "extra"


# ---------------------------------------------------------------------------
# unsupported_okx_order_request_error
# ---------------------------------------------------------------------------


class TestUnsupportedOrderRequestError:
    def test_returns_exchange_error_with_correct_kind(self):
        req = BrokerOrderRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.STOP_MARKET,
            quantity=Decimal("1"),
            reduce_only=True,
        )
        err = unsupported_okx_order_request_error(req, "stop-market not yet supported")
        assert isinstance(err, ExchangeError)
        assert err.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
        assert "stop-market" in err.detail.message
        # The order_type is stored in raw as a separate key
        assert err.detail.raw.get("order_type") == "STOP_MARKET"

    def test_raw_contains_request_info(self):
        req = BrokerOrderRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.SHORT,
            order_type=BrokerOrderType.TAKE_PROFIT_MARKET,
            quantity=Decimal("2"),
            reduce_only=True,
        )
        err = unsupported_okx_order_request_error(req, "not supported")
        assert err.detail.raw["symbol"] == "ETH-USDT-SWAP"
        assert err.detail.raw["order_type"] == "TAKE_PROFIT_MARKET"
        assert err.detail.raw["position_side"] == "SHORT"
        assert err.detail.raw["reduce_only"] is True
