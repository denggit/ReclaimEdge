from decimal import Decimal
from pathlib import Path

import pytest

from src.exchanges.capabilities import okx_capabilities
from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerBalance,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    ExchangeName,
)
from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderQuery,
    BrokerSemanticRequest,
)


class FakeBroker:
    def __init__(
        self,
        *,
        exchange: ExchangeName = ExchangeName.OKX,
        result_exchange: ExchangeName | None = None,
        result_symbol: str | None = None,
    ) -> None:
        self._exchange = exchange
        self._result_exchange = result_exchange
        self._result_symbol = result_symbol
        self.place_order_requests: list[BrokerOrderRequest] = []
        self.cancel_order_calls: list[tuple[str, str]] = []
        self.fetch_open_orders_calls: list[str] = []
        self.fetch_position_calls: list[tuple[str, BrokerPositionSide | None]] = []
        self.closed = False
        self.open_orders = [_broker_order()]
        self.position = _broker_position()

    @property
    def exchange(self) -> ExchangeName:
        return self._exchange

    @property
    def capabilities(self):
        return okx_capabilities()

    async def fetch_instrument(self, symbol: str) -> BrokerInstrument:
        raise NotImplementedError

    async def fetch_balance(self, asset: str = "USDT") -> BrokerBalance:
        raise NotImplementedError

    async def fetch_position(
        self, symbol: str, side: BrokerPositionSide | None = None
    ) -> BrokerPosition:
        self.fetch_position_calls.append((symbol, side))
        return self.position

    async def fetch_open_orders(self, symbol: str) -> list[BrokerOrder]:
        self.fetch_open_orders_calls.append(symbol)
        return list(self.open_orders)

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        self.place_order_requests.append(request)
        return BrokerOrderResult(
            exchange=self._result_exchange or request.exchange,
            symbol=self._result_symbol or request.symbol,
            order_id="order-1",
            client_order_id=request.client_order_id,
            status=BrokerOrderStatus.NEW,
            raw={"source": "fake"},
        )

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        self.cancel_order_calls.append((symbol, order_id))

    async def cancel_all_open_orders(self, symbol: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        self.closed = True


def test_constructor_rejects_non_okx_broker():
    broker = FakeBroker(exchange=ExchangeName.BINANCE)

    with pytest.raises(ExchangeError) as exc_info:
        OkxBrokerSemanticExecutor(broker)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


@pytest.mark.asyncio
async def test_execute_open_position_calls_broker_place_market_order():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.OPEN_POSITION)

    result = await executor.execute_semantic_order(request)

    assert len(broker.place_order_requests) == 1
    assert broker.place_order_requests[0].order_type == BrokerOrderType.MARKET
    assert broker.place_order_requests[0].reduce_only is False
    assert result.ok is True
    assert result.action == BrokerSemanticAction.OPEN_POSITION


@pytest.mark.asyncio
async def test_execute_rejects_request_exchange_mismatch():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(
        exchange=ExchangeName.BINANCE,
        action=BrokerSemanticAction.OPEN_POSITION,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute_semantic_order(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert broker.place_order_requests == []


@pytest.mark.asyncio
async def test_execute_reduce_only_tp_calls_broker_place_limit_order():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(
        action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
        side=BrokerOrderSide.SELL,
        price=Decimal("3500"),
    )

    result = await executor.execute_semantic_order(request)

    assert len(broker.place_order_requests) == 1
    assert broker.place_order_requests[0].order_type == BrokerOrderType.LIMIT
    assert broker.place_order_requests[0].reduce_only is True
    assert result.ok is True


@pytest.mark.asyncio
async def test_cancel_semantic_order_calls_broker_cancel_order():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.CANCEL_ORDER)

    result = await executor.cancel_semantic_order(request, "order-1")

    assert broker.cancel_order_calls == [("ETH-USDT-SWAP", "order-1")]
    assert result.ok is True
    assert result.order_id == "order-1"


@pytest.mark.asyncio
async def test_cancel_semantic_order_rejects_exchange_mismatch():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(
        exchange=ExchangeName.BINANCE,
        action=BrokerSemanticAction.CANCEL_ORDER,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await executor.cancel_semantic_order(request, "order-1")

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert broker.cancel_order_calls == []


@pytest.mark.asyncio
async def test_cancel_semantic_order_rejects_non_cancel_action():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.OPEN_POSITION)

    with pytest.raises(ExchangeError) as exc_info:
        await executor.cancel_semantic_order(request, "order-1")

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert broker.cancel_order_calls == []


@pytest.mark.asyncio
async def test_cancel_semantic_order_allows_sidecar_cancel():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.SIDECAR_CANCEL)

    result = await executor.cancel_semantic_order(request, "order-1")

    assert broker.cancel_order_calls == [("ETH-USDT-SWAP", "order-1")]
    assert result.ok is True
    assert result.action == BrokerSemanticAction.SIDECAR_CANCEL


@pytest.mark.asyncio
async def test_fetch_semantic_orders_calls_fetch_open_orders():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    query = BrokerSemanticOrderQuery(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        include_algo=False,
    )

    orders = await executor.fetch_semantic_orders(query)

    assert orders == tuple(broker.open_orders)
    assert broker.fetch_open_orders_calls == ["ETH-USDT-SWAP"]


@pytest.mark.asyncio
async def test_fetch_semantic_orders_rejects_query_exchange_mismatch():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    query = BrokerSemanticOrderQuery(
        exchange=ExchangeName.BINANCE,
        symbol="ETH-USDT-SWAP",
        include_algo=False,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await executor.fetch_semantic_orders(query)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert broker.fetch_open_orders_calls == []


@pytest.mark.asyncio
async def test_fetch_semantic_position_calls_fetch_position():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)

    position = await executor.fetch_semantic_position(
        "ETH-USDT-SWAP",
        BrokerPositionSide.LONG,
    )

    assert position == broker.position
    assert broker.fetch_position_calls == [("ETH-USDT-SWAP", BrokerPositionSide.LONG)]


@pytest.mark.asyncio
async def test_unsupported_semantic_action_raises_exchange_error():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP)

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute_semantic_order(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


@pytest.mark.asyncio
async def test_execute_rejects_broker_result_exchange_mismatch():
    broker = FakeBroker(result_exchange=ExchangeName.BINANCE)
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.OPEN_POSITION)

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute_semantic_order(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert len(broker.place_order_requests) == 1


@pytest.mark.asyncio
async def test_execute_rejects_broker_result_symbol_mismatch():
    broker = FakeBroker(result_symbol="BTC-USDT-SWAP")
    executor = OkxBrokerSemanticExecutor(broker)
    request = _semantic_request(action=BrokerSemanticAction.OPEN_POSITION)

    with pytest.raises(ExchangeError) as exc_info:
        await executor.execute_semantic_order(request)

    assert exc_info.value.detail.kind == ExchangeErrorKind.INVALID_SYMBOL
    assert len(broker.place_order_requests) == 1


@pytest.mark.asyncio
async def test_executor_does_not_require_trader():
    broker = FakeBroker()
    executor = OkxBrokerSemanticExecutor(broker)

    await executor.close()

    assert broker.closed is True


def test_okx_semantic_executor_shell_has_no_live_or_raw_endpoint_logic():
    source = Path("src/exchanges/okx/semantic_executor.py").read_text()

    assert "src.execution.trader" not in source
    assert "OkxPrivateClient" not in source
    assert "/api/v5" not in source


def _semantic_request(
    *,
    exchange: ExchangeName = ExchangeName.OKX,
    action: BrokerSemanticAction,
    side: BrokerOrderSide = BrokerOrderSide.BUY,
    position_side: BrokerPositionSide = BrokerPositionSide.LONG,
    quantity: Decimal = Decimal("1"),
    price: Decimal | None = None,
) -> BrokerSemanticRequest:
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol="ETH-USDT-SWAP",
        action=action,
        side=side,
        position_side=position_side,
        quantity=quantity,
        price=price,
    )


def _broker_order() -> BrokerOrder:
    return BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        order_id="order-1",
        client_order_id=None,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.LIMIT,
        status=BrokerOrderStatus.NEW,
        price=Decimal("3500"),
        quantity=Decimal("1"),
        reduce_only=True,
    )


def _broker_position() -> BrokerPosition:
    return BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerPositionSide.LONG,
        contracts=Decimal("1"),
        base_qty=Decimal("0.1"),
        avg_entry_price=Decimal("3000"),
    )
