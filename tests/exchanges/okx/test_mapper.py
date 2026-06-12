"""Tests for src.exchanges.okx.mapper — pure mapping functions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerExecutionAction,
    BrokerExecutionResult,
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerTimeInForce,
    ExchangeName,
)
from src.exchanges.okx.mapper import (
    broker_execution_result_from_live_trade_result,
    broker_instrument_from_trader,
    broker_order_from_okx_pending_algo_order,
    broker_order_from_okx_pending_order,
    broker_order_side_from_okx_side,
    broker_order_status_from_okx_state,
    broker_order_type_from_okx_ord_type,
    broker_position_from_snapshot,
    broker_position_side_from_okx_pos_side,
    okx_reduce_only_flag,
    okx_trigger_price,
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
            "label": "tp-label-1",
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
        assert order.close_position is False
        # trigger_price falls back to px when no specific trigger fields exist
        assert order.trigger_price == Decimal("3500.00")
        assert order.label == "tp-label-1"

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
        """Ordinary pending orders with 'conditional' ordType fall back to LIMIT.

        Only algo orders map 'conditional' → STOP_MARKET.
        """
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
        # Conditional is NOT recognized for ordinary orders → conservative LIMIT
        assert order.order_type == BrokerOrderType.LIMIT

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
# broker_order_side_from_okx_side
# ---------------------------------------------------------------------------


class TestBrokerOrderSideFromOkxSide:
    def test_buy(self):
        assert broker_order_side_from_okx_side("buy") == BrokerOrderSide.BUY

    def test_sell(self):
        assert broker_order_side_from_okx_side("sell") == BrokerOrderSide.SELL

    def test_unknown_falls_back_to_buy(self):
        assert broker_order_side_from_okx_side("unknown") == BrokerOrderSide.BUY

    def test_none_falls_back_to_buy(self):
        assert broker_order_side_from_okx_side(None) == BrokerOrderSide.BUY

    def test_empty_string_falls_back_to_buy(self):
        assert broker_order_side_from_okx_side("") == BrokerOrderSide.BUY


# ---------------------------------------------------------------------------
# broker_position_side_from_okx_pos_side
# ---------------------------------------------------------------------------


class TestBrokerPositionSideFromOkxPosSide:
    def test_long(self):
        assert broker_position_side_from_okx_pos_side("long") == BrokerPositionSide.LONG

    def test_short(self):
        assert broker_position_side_from_okx_pos_side("short") == BrokerPositionSide.SHORT

    def test_net(self):
        assert broker_position_side_from_okx_pos_side("net") == BrokerPositionSide.NET

    def test_none_falls_back_to_net(self):
        assert broker_position_side_from_okx_pos_side(None) == BrokerPositionSide.NET

    def test_empty_falls_back_to_net(self):
        assert broker_position_side_from_okx_pos_side("") == BrokerPositionSide.NET

    def test_unknown_falls_back_to_net(self):
        assert broker_position_side_from_okx_pos_side("unknown") == BrokerPositionSide.NET


# ---------------------------------------------------------------------------
# broker_order_status_from_okx_state
# ---------------------------------------------------------------------------


class TestBrokerOrderStatusFromOkxState:
    def test_live(self):
        assert broker_order_status_from_okx_state("live") == BrokerOrderStatus.NEW

    def test_partially_filled(self):
        assert broker_order_status_from_okx_state("partially_filled") == BrokerOrderStatus.PARTIALLY_FILLED

    def test_partially_filled_alt_spelling(self):
        assert broker_order_status_from_okx_state("partially-filled") == BrokerOrderStatus.PARTIALLY_FILLED

    def test_filled(self):
        assert broker_order_status_from_okx_state("filled") == BrokerOrderStatus.FILLED

    def test_canceled(self):
        assert broker_order_status_from_okx_state("canceled") == BrokerOrderStatus.CANCELED

    def test_cancelled(self):
        assert broker_order_status_from_okx_state("cancelled") == BrokerOrderStatus.CANCELED

    def test_rejected(self):
        assert broker_order_status_from_okx_state("rejected") == BrokerOrderStatus.REJECTED

    def test_expired(self):
        assert broker_order_status_from_okx_state("expired") == BrokerOrderStatus.EXPIRED

    def test_unknown(self):
        assert broker_order_status_from_okx_state("something_else") == BrokerOrderStatus.UNKNOWN

    def test_none(self):
        assert broker_order_status_from_okx_state(None) == BrokerOrderStatus.UNKNOWN


# ---------------------------------------------------------------------------
# broker_order_type_from_okx_ord_type
# ---------------------------------------------------------------------------


class TestBrokerOrderTypeFromOkxOrdType:
    def test_market_ordinary(self):
        assert broker_order_type_from_okx_ord_type("market") == BrokerOrderType.MARKET

    def test_limit_ordinary(self):
        assert broker_order_type_from_okx_ord_type("limit") == BrokerOrderType.LIMIT

    def test_unknown_ordinary_falls_back_to_limit(self):
        assert broker_order_type_from_okx_ord_type("iceberg") == BrokerOrderType.LIMIT

    def test_conditional_algo(self):
        assert broker_order_type_from_okx_ord_type("conditional", is_algo=True) == BrokerOrderType.STOP_MARKET

    def test_oco_algo(self):
        assert broker_order_type_from_okx_ord_type("oco", is_algo=True) == BrokerOrderType.STOP_MARKET

    def test_trigger_algo(self):
        assert broker_order_type_from_okx_ord_type("trigger", is_algo=True) == BrokerOrderType.STOP_MARKET

    def test_move_order_stop_algo(self):
        assert broker_order_type_from_okx_ord_type("move_order_stop", is_algo=True) == BrokerOrderType.STOP_MARKET

    def test_unknown_algo_falls_back_to_stop_market(self):
        assert broker_order_type_from_okx_ord_type("unknown_algo", is_algo=True) == BrokerOrderType.STOP_MARKET


# ---------------------------------------------------------------------------
# okx_reduce_only_flag
# ---------------------------------------------------------------------------


class TestOkxReduceOnlyFlag:
    def test_boolean_true(self):
        assert okx_reduce_only_flag({"reduceOnly": True}) is True

    def test_string_true(self):
        assert okx_reduce_only_flag({"reduceOnly": "true"}) is True

    def test_string_1(self):
        assert okx_reduce_only_flag({"reduceOnly": "1"}) is True

    def test_string_yes(self):
        assert okx_reduce_only_flag({"reduceOnly": "yes"}) is True

    def test_string_y(self):
        assert okx_reduce_only_flag({"reduceOnly": "y"}) is True

    def test_false_bool(self):
        assert okx_reduce_only_flag({"reduceOnly": False}) is False

    def test_false_string(self):
        assert okx_reduce_only_flag({"reduceOnly": "false"}) is False

    def test_missing(self):
        assert okx_reduce_only_flag({}) is False

    def test_reduce_only_underscore_key(self):
        assert okx_reduce_only_flag({"reduce_only": "true"}) is True

    def test_reduce_only_hyphen_key(self):
        assert okx_reduce_only_flag({"reduce-only": "true"}) is True


# ---------------------------------------------------------------------------
# okx_trigger_price
# ---------------------------------------------------------------------------


class TestOkxTriggerPrice:
    def test_sl_trigger_px(self):
        assert okx_trigger_price({"slTriggerPx": "3100.00"}) == Decimal("3100.00")

    def test_tp_trigger_px(self):
        assert okx_trigger_price({"tpTriggerPx": "3500.00"}) == Decimal("3500.00")

    def test_trigger_px(self):
        assert okx_trigger_price({"triggerPx": "3300.00"}) == Decimal("3300.00")

    def test_ord_px(self):
        assert okx_trigger_price({"ordPx": "3400.00"}) == Decimal("3400.00")

    def test_px(self):
        assert okx_trigger_price({"px": "3600.00"}) == Decimal("3600.00")

    def test_priority_sl_over_tp(self):
        """slTriggerPx takes priority over tpTriggerPx."""
        assert okx_trigger_price({"slTriggerPx": "3100.00", "tpTriggerPx": "3500.00"}) == Decimal("3100.00")

    def test_missing(self):
        assert okx_trigger_price({}) is None

    def test_invalid(self):
        assert okx_trigger_price({"slTriggerPx": "not_a_number"}) is None

    def test_empty_string(self):
        assert okx_trigger_price({"slTriggerPx": ""}) is None


# ---------------------------------------------------------------------------
# broker_order_from_okx_pending_algo_order
# ---------------------------------------------------------------------------


class TestBrokerOrderFromPendingAlgoOrder:
    def test_conditional_protective_sl(self):
        item = {
            "algoId": "algo-001",
            "algoClOrdId": "algo-client-abc",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
            "state": "live",
            "slTriggerPx": "3100.00",
            "sz": "1.5",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_id == "algo-001"
        assert order.client_order_id == "algo-client-abc"
        assert order.side == BrokerOrderSide.SELL
        assert order.position_side == BrokerPositionSide.LONG
        assert order.order_type == BrokerOrderType.STOP_MARKET
        assert order.status == BrokerOrderStatus.NEW
        assert order.trigger_price == Decimal("3100.00")
        assert order.quantity == Decimal("1.5")
        assert order.reduce_only is False

    def test_algo_id_as_order_id(self):
        item = {
            "algoId": "algo-002",
            "side": "buy",
            "posSide": "short",
            "ordType": "conditional",
            "state": "live",
            "sz": "2.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_id == "algo-002"

    def test_algo_cl_ord_id_as_client_order_id(self):
        item = {
            "algoId": "algo-003",
            "algoClOrdId": "my-algo-client-id",
            "side": "sell",
            "posSide": "long",
            "ordType": "oco",
            "state": "live",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.client_order_id == "my-algo-client-id"

    def test_sl_trigger_px_as_trigger_price(self):
        item = {
            "algoId": "algo-004",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
            "state": "live",
            "slTriggerPx": "2999.50",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.trigger_price == Decimal("2999.50")

    def test_missing_algo_id_does_not_crash(self):
        item = {
            "side": "sell",
            "ordType": "conditional",
            "state": "live",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.order_id == ""
        assert order.status == BrokerOrderStatus.UNKNOWN

    def test_algo_state_fallback(self):
        """algoState is used when state is missing."""
        item = {
            "algoId": "algo-005",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
            "algoState": "live",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.status == BrokerOrderStatus.NEW

    def test_label_from_algo_cl_ord_id(self):
        item = {
            "algoId": "algo-006",
            "algoClOrdId": "label-from-clordid",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
            "state": "live",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.label == "label-from-clordid"

    def test_label_prefers_explicit_label(self):
        item = {
            "algoId": "algo-007",
            "algoClOrdId": "clordid-label",
            "label": "explicit-label",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
            "state": "live",
            "sz": "1.0",
            "accFillSz": "0",
        }
        order = broker_order_from_okx_pending_algo_order(item, symbol="ETH-USDT-SWAP")
        assert order.label == "explicit-label"


# ---------------------------------------------------------------------------
# broker_execution_result_from_live_trade_result
# ---------------------------------------------------------------------------


class TestBrokerExecutionResultFromLiveTradeResult:
    """Tests for LiveTradeResult → BrokerExecutionResult mapping.

    Uses lightweight fake objects — does NOT import the real LiveTradeResult.
    """

    @staticmethod
    def _fake_result(**overrides: object) -> object:
        """Build a minimal fake that exposes the same attributes as LiveTradeResult."""

        class _FakeLiveTradeResult:
            ok: bool = True
            action: str = "OPEN_LONG"
            order_id: str | None = "order-123"
            tp_order_id: str | None = "tp-456"
            tp_order_ids: tuple[str, ...] = ()
            protective_sl_order_id: str | None = None
            contracts: str = "1.5"
            tp_price: str = "3500.00"
            message: str = "entry ok"
            entry_filled: bool = True
            tp_ok: bool | None = True
            protective_sl_price: str = ""
            protective_sl_ok: bool | None = None

            def __init__(self, **kw: object) -> None:
                for k, v in kw.items():
                    object.__setattr__(self, k, v)

        return _FakeLiveTradeResult(**overrides)

    def test_open_long_success(self):
        result = self._fake_result(
            ok=True,
            action="OPEN_LONG",
            order_id="order-001",
            tp_order_id="tp-001",
            contracts="1.5",
            tp_price="3500.00",
            message="entry ok",
            entry_filled=True,
            tp_ok=True,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert isinstance(br, BrokerExecutionResult)
        assert br.exchange == ExchangeName.OKX
        assert br.symbol == "ETH-USDT-SWAP"
        assert br.action == BrokerExecutionAction.OPEN_LONG
        assert br.ok is True
        assert br.order_id == "order-001"
        assert br.tp_order_id == "tp-001"
        assert br.contracts == Decimal("1.5")
        assert br.tp_price == Decimal("3500.00")
        assert br.message == "entry ok"
        assert br.entry_filled is True
        assert br.tp_ok is True

    def test_update_tp_with_tp_order_ids_string(self):
        result = self._fake_result(
            action="UPDATE_TP",
            tp_order_ids="id1,id2,id3",
            tp_ok=True,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.UPDATE_TP
        assert br.tp_order_ids == ("id1", "id2", "id3")

    def test_update_tp_with_tp_order_ids_list(self):
        result = self._fake_result(
            action="UPDATE_TP",
            tp_order_ids=["id-a", "id-b"],
            tp_ok=True,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.tp_order_ids == ("id-a", "id-b")

    def test_update_tp_with_tp_order_ids_empty(self):
        result = self._fake_result(
            action="UPDATE_TP",
            tp_order_ids=(),
            tp_ok=True,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.tp_order_ids == ()

    def test_market_exit_runner_failed_with_protective_sl(self):
        result = self._fake_result(
            ok=False,
            action="MARKET_EXIT_RUNNER",
            protective_sl_order_id="sl-999",
            protective_sl_price="3000.00",
            protective_sl_ok=False,
            message="exit failed",
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.MARKET_EXIT_RUNNER
        assert br.ok is False
        assert br.protective_sl_order_id == "sl-999"
        assert br.protective_sl_price == Decimal("3000.00")
        assert br.protective_sl_ok is False
        assert br.message == "exit failed"

    def test_unknown_action(self):
        result = self._fake_result(action="SOME_NEW_ACTION")
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.UNKNOWN

    def test_empty_action(self):
        result = self._fake_result(action="")
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.UNKNOWN

    def test_none_numeric_fields(self):
        result = self._fake_result(
            contracts=None,
            tp_price=None,
            protective_sl_price=None,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.contracts is None
        assert br.tp_price is None
        assert br.protective_sl_price is None

    def test_raw_contains_lightweight_metadata(self):
        result = self._fake_result(action="OPEN_SHORT", message="short entry")
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.raw["legacy_type"] == "_FakeLiveTradeResult"
        assert br.raw["legacy_action"] == "OPEN_SHORT"
        assert br.raw["legacy_message"] == "short entry"

    def test_add_long_action(self):
        result = self._fake_result(
            action="ADD_LONG",
            contracts="0.5",
            tp_ok=None,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.ADD_LONG
        assert br.contracts == Decimal("0.5")

    def test_add_short_action(self):
        result = self._fake_result(action="ADD_SHORT")
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.ADD_SHORT

    def test_near_tp_reduce_action(self):
        result = self._fake_result(
            action="NEAR_TP_REDUCE",
            tp_ok=True,
        )
        br = broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            result=result,
        )
        assert br.action == BrokerExecutionAction.NEAR_TP_REDUCE


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
