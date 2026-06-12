from decimal import Decimal

import pytest

from src.exchanges.base import BrokerClient
from src.exchanges.capabilities import ExchangeCapabilities, okx_capabilities
from src.exchanges.models import (
    BrokerBalance,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerPositionSide,
    ExchangeName,
)


def test_broker_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BrokerClient()


class FakeBrokerClient(BrokerClient):
    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.OKX

    @property
    def capabilities(self) -> ExchangeCapabilities:
        return okx_capabilities()

    async def fetch_instrument(self, symbol: str) -> BrokerInstrument:
        return BrokerInstrument(
            exchange=self.exchange,
            symbol=symbol,
            base_asset="ETH",
            quote_asset="USDT",
            contract_type="SWAP",
            price_tick=Decimal("0.01"),
            qty_step=Decimal("0.01"),
            min_qty=Decimal("0.01"),
            min_notional=Decimal("1"),
        )

    async def fetch_balance(self, asset: str = "USDT") -> BrokerBalance:
        return BrokerBalance(
            exchange=self.exchange,
            asset=asset,
            total=Decimal("100"),
            available=Decimal("90"),
        )

    async def fetch_position(self, symbol: str, side: BrokerPositionSide | None = None) -> BrokerPosition:
        return BrokerPosition(
            exchange=self.exchange,
            symbol=symbol,
            side=side or BrokerPositionSide.NET,
            contracts=Decimal("0"),
            base_qty=Decimal("0"),
            avg_entry_price=Decimal("0"),
        )

    async def fetch_open_orders(self, symbol: str) -> list[BrokerOrder]:
        return []

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        return BrokerOrderResult(
            exchange=request.exchange,
            symbol=request.symbol,
            order_id="fake-order",
            client_order_id=request.client_order_id,
            status=BrokerOrderStatus.NEW,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        return None

    async def cancel_all_open_orders(self, symbol: str) -> None:
        return None

    async def close(self) -> None:
        return None


def test_fake_broker_client_can_be_instantiated():
    assert isinstance(FakeBrokerClient(), BrokerClient)


def test_fake_broker_client_returns_exchange_and_capabilities():
    client = FakeBrokerClient()

    assert client.exchange == ExchangeName.OKX
    assert client.capabilities.exchange == ExchangeName.OKX
