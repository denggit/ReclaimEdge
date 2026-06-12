from __future__ import annotations

from abc import ABC, abstractmethod

from src.exchanges.capabilities import ExchangeCapabilities
from src.exchanges.models import (
    BrokerBalance,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerPosition,
    BrokerPositionSide,
    ExchangeName,
)


class BrokerClient(ABC):
    @property
    @abstractmethod
    def exchange(self) -> ExchangeName:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ExchangeCapabilities:
        ...

    @abstractmethod
    async def fetch_instrument(self, symbol: str) -> BrokerInstrument:
        ...

    @abstractmethod
    async def fetch_balance(self, asset: str = "USDT") -> BrokerBalance:
        ...

    @abstractmethod
    async def fetch_position(self, symbol: str, side: BrokerPositionSide | None = None) -> BrokerPosition:
        ...

    @abstractmethod
    async def fetch_open_orders(self, symbol: str) -> list[BrokerOrder]:
        ...

    @abstractmethod
    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> None:
        ...

    @abstractmethod
    async def cancel_all_open_orders(self, symbol: str) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
