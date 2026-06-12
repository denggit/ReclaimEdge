from decimal import Decimal

from src.exchanges.models import (
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    ExchangeName,
)


def test_broker_position_has_position_uses_contracts_or_base_quantity():
    flat = BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerPositionSide.LONG,
        contracts=Decimal("0"),
        base_qty=Decimal("0"),
        avg_entry_price=Decimal("0"),
    )
    contract_position = BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerPositionSide.LONG,
        contracts=Decimal("-1"),
        base_qty=Decimal("0"),
        avg_entry_price=Decimal("3000"),
    )
    base_position = BrokerPosition(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        contracts=Decimal("0"),
        base_qty=Decimal("-0.25"),
        avg_entry_price=Decimal("3000"),
    )

    assert flat.has_position is False
    assert contract_position.has_position is True
    assert base_position.has_position is True


def test_decimal_fields_are_preserved_as_decimal_values():
    instrument = BrokerInstrument(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        contract_type="PERPETUAL",
        price_tick=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        contract_size=Decimal("1"),
    )

    assert isinstance(instrument.price_tick, Decimal)
    assert isinstance(instrument.qty_step, Decimal)
    assert instrument.price_tick == Decimal("0.01")
    assert instrument.qty_step == Decimal("0.001")


def test_broker_order_reduce_only_trigger_price_and_label_are_available():
    order = BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        order_id="internal-order-1",
        client_order_id="client-1",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.STOP_MARKET,
        status=BrokerOrderStatus.NEW,
        price=None,
        quantity=Decimal("1"),
        reduce_only=True,
        trigger_price=Decimal("2800"),
        label="protective-sl",
    )

    assert order.reduce_only is True
    assert order.trigger_price == Decimal("2800")
    assert order.label == "protective-sl"


def test_raw_defaults_are_empty_and_not_shared():
    first = BrokerInstrument(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        base_asset="ETH",
        quote_asset="USDT",
        contract_type="SWAP",
        price_tick=Decimal("0.01"),
        qty_step=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        min_notional=Decimal("1"),
    )
    second = BrokerInstrument(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        contract_type="PERPETUAL",
        price_tick=Decimal("0.01"),
        qty_step=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
    )

    assert first.raw == {}
    assert second.raw == {}
    assert first.raw is not second.raw
