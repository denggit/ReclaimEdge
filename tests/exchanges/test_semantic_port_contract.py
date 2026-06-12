from decimal import Decimal
from pathlib import Path

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrderSide,
    BrokerOrderType,
    BrokerPositionSide,
    ExchangeName,
)
from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticRequest
from src.exchanges.semantics import (
    semantic_request_to_broker_order_request,
    validate_semantic_request,
)


def test_validate_semantic_request_rejects_empty_symbol():
    request = _semantic_request(symbol="")

    with pytest.raises(ExchangeError) as exc_info:
        validate_semantic_request(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL


def test_validate_semantic_request_rejects_unknown_action():
    request = _semantic_request(action=BrokerSemanticAction.UNKNOWN)

    with pytest.raises(ExchangeError) as exc_info:
        validate_semantic_request(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


def test_validate_semantic_request_rejects_non_positive_quantity():
    request = _semantic_request(quantity=Decimal("0"))

    with pytest.raises(ExchangeError) as exc_info:
        validate_semantic_request(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_QUANTITY


def test_validate_semantic_request_rejects_non_positive_price():
    request = _semantic_request(price=Decimal("0"))

    with pytest.raises(ExchangeError) as exc_info:
        validate_semantic_request(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_PRICE


def test_validate_semantic_request_rejects_non_positive_trigger_price():
    request = _semantic_request(trigger_price=Decimal("0"))

    with pytest.raises(ExchangeError) as exc_info:
        validate_semantic_request(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_TRIGGER_PRICE


def test_open_position_semantic_to_market_broker_order_request():
    request = _semantic_request(action=BrokerSemanticAction.OPEN_POSITION)

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.MARKET,
    )

    assert broker_request.exchange == request.exchange
    assert broker_request.symbol == request.symbol
    assert broker_request.side == request.side
    assert broker_request.position_side == request.position_side
    assert broker_request.quantity == request.quantity
    assert broker_request.order_type == BrokerOrderType.MARKET
    assert broker_request.reduce_only is False
    assert broker_request.close_position is False


def test_add_position_semantic_to_market_broker_order_request():
    request = _semantic_request(action=BrokerSemanticAction.ADD_POSITION)

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.MARKET,
    )

    assert broker_request.order_type == BrokerOrderType.MARKET
    assert broker_request.reduce_only is False
    assert broker_request.symbol == request.symbol
    assert broker_request.exchange == request.exchange
    assert broker_request.side == request.side
    assert broker_request.position_side == request.position_side
    assert broker_request.quantity == request.quantity


def test_reduce_only_tp_semantic_to_limit_broker_order_request():
    request = _semantic_request(
        action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        price=Decimal("3500"),
    )

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.LIMIT,
    )

    assert broker_request.order_type == BrokerOrderType.LIMIT
    assert broker_request.reduce_only is True
    assert broker_request.price == Decimal("3500")


def test_sidecar_tp_semantic_to_limit_broker_order_request():
    request = _semantic_request(
        action=BrokerSemanticAction.SIDECAR_TP,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        price=Decimal("3500"),
    )

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.LIMIT,
    )

    assert broker_request.order_type == BrokerOrderType.LIMIT
    assert broker_request.reduce_only is True
    assert broker_request.price == Decimal("3500")


def test_sidecar_entry_semantic_to_market_broker_order_request():
    request = _semantic_request(action=BrokerSemanticAction.SIDECAR_ENTRY)

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.MARKET,
    )

    assert broker_request.order_type == BrokerOrderType.MARKET
    assert broker_request.reduce_only is False
    assert broker_request.close_position is False


def test_protective_stop_semantic_to_stop_market_broker_order_request():
    request = _semantic_request(
        action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        trigger_price=Decimal("2800"),
    )

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.STOP_MARKET,
    )

    assert broker_request.order_type == BrokerOrderType.STOP_MARKET
    assert broker_request.reduce_only is True
    assert broker_request.trigger_price == Decimal("2800")


def test_market_exit_semantic_to_reduce_only_market_broker_order_request():
    request = _semantic_request(
        action=BrokerSemanticAction.MARKET_EXIT,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
    )

    broker_request = semantic_request_to_broker_order_request(
        request,
        BrokerOrderType.MARKET,
    )

    assert broker_request.order_type == BrokerOrderType.MARKET
    assert broker_request.reduce_only is True
    assert broker_request.close_position is True


def test_protective_stop_requires_trigger_price():
    request = _semantic_request(action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP)

    with pytest.raises(ExchangeError) as exc_info:
        semantic_request_to_broker_order_request(request, BrokerOrderType.STOP_MARKET)

    assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_TRIGGER_PRICE


def test_unsupported_semantic_action_to_order_request_raises():
    request = _semantic_request(action=BrokerSemanticAction.CANCEL_ORDER)

    with pytest.raises(ExchangeError) as exc_info:
        semantic_request_to_broker_order_request(request, BrokerOrderType.MARKET)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


def test_semantics_port_does_not_contain_adapter_or_raw_exchange_details():
    source = Path("src/exchanges/semantics.py").read_text()

    assert "src.execution.trader" not in source
    assert "OkxPrivateClient" not in source
    assert "/api/v5" not in source
    assert "ordId" not in source
    assert "algoId" not in source
    assert "sCode" not in source
    assert "sMsg" not in source


def _semantic_request(
    *,
    exchange: ExchangeName = ExchangeName.OKX,
    symbol: str = "ETH-USDT-SWAP",
    action: BrokerSemanticAction = BrokerSemanticAction.OPEN_POSITION,
    side: BrokerOrderSide = BrokerOrderSide.BUY,
    position_side: BrokerPositionSide = BrokerPositionSide.LONG,
    quantity: Decimal | None = Decimal("1"),
    price: Decimal | None = None,
    trigger_price: Decimal | None = None,
) -> BrokerSemanticRequest:
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=action,
        side=side,
        position_side=position_side,
        quantity=quantity,
        price=price,
        trigger_price=trigger_price,
    )
