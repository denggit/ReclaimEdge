from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


class ExchangeName(str, Enum):
    OKX = "OKX"
    BINANCE = "BINANCE"


class BrokerPositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NET = "NET"


class BrokerOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class BrokerOrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class BrokerOrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class BrokerTimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "POST_ONLY"


@dataclass(frozen=True)
class BrokerInstrument:
    exchange: ExchangeName
    symbol: str
    base_asset: str
    quote_asset: str
    contract_type: str
    price_tick: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    contract_size: Decimal = Decimal("1")
    margin_asset: str = "USDT"
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerPosition:
    exchange: ExchangeName
    symbol: str
    side: BrokerPositionSide
    contracts: Decimal
    base_qty: Decimal
    avg_entry_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    leverage: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.contracts.copy_abs() > Decimal("0") or self.base_qty.copy_abs() > Decimal("0")


@dataclass(frozen=True)
class BrokerOrder:
    exchange: ExchangeName
    symbol: str
    order_id: str
    client_order_id: str | None
    side: BrokerOrderSide
    position_side: BrokerPositionSide
    order_type: BrokerOrderType
    status: BrokerOrderStatus
    price: Decimal | None
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    reduce_only: bool = False
    close_position: bool = False
    trigger_price: Decimal | None = None
    label: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerOrderRequest:
    exchange: ExchangeName
    symbol: str
    side: BrokerOrderSide
    position_side: BrokerPositionSide
    order_type: BrokerOrderType
    quantity: Decimal
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    reduce_only: bool = False
    close_position: bool = False
    time_in_force: BrokerTimeInForce | None = None
    client_order_id: str | None = None
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerOrderResult:
    exchange: ExchangeName
    symbol: str
    order_id: str | None
    client_order_id: str | None
    status: BrokerOrderStatus
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerBalance:
    exchange: ExchangeName
    asset: str
    total: Decimal
    available: Decimal
    equity: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)
