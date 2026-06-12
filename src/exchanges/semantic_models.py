from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from src.exchanges.models import (
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerPositionSide,
    ExchangeName,
)


class BrokerSemanticAction(str, Enum):
    OPEN_POSITION = "OPEN_POSITION"
    ADD_POSITION = "ADD_POSITION"
    PLACE_REDUCE_ONLY_TP = "PLACE_REDUCE_ONLY_TP"
    CANCEL_ORDER = "CANCEL_ORDER"
    CANCEL_ALL_ORDINARY_ORDERS = "CANCEL_ALL_ORDINARY_ORDERS"
    PLACE_PROTECTIVE_STOP = "PLACE_PROTECTIVE_STOP"
    CANCEL_PROTECTIVE_STOP = "CANCEL_PROTECTIVE_STOP"
    MARKET_EXIT = "MARKET_EXIT"
    CLOSE_POSITION = "CLOSE_POSITION"
    FETCH_OPEN_ORDERS = "FETCH_OPEN_ORDERS"
    FETCH_PROTECTIVE_ORDERS = "FETCH_PROTECTIVE_ORDERS"
    SYNC_POSITION = "SYNC_POSITION"
    SIDECAR_ENTRY = "SIDECAR_ENTRY"
    SIDECAR_TP = "SIDECAR_TP"
    SIDECAR_CANCEL = "SIDECAR_CANCEL"
    UNKNOWN = "UNKNOWN"


class BrokerSemanticOrderRole(str, Enum):
    ENTRY = "ENTRY"
    ADD = "ADD"
    CORE_TP = "CORE_TP"
    MIDDLE_TP = "MIDDLE_TP"
    RUNNER_TP = "RUNNER_TP"
    NEAR_TP = "NEAR_TP"
    PROTECTIVE_SL = "PROTECTIVE_SL"
    MIDDLE_RUNNER_SL = "MIDDLE_RUNNER_SL"
    THREE_STAGE_SL = "THREE_STAGE_SL"
    TREND_RUNNER_SL = "TREND_RUNNER_SL"
    SIDECAR_ENTRY = "SIDECAR_ENTRY"
    SIDECAR_TP = "SIDECAR_TP"
    MARKET_EXIT = "MARKET_EXIT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class BrokerSemanticRequest:
    exchange: ExchangeName
    symbol: str
    action: BrokerSemanticAction
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.UNKNOWN
    side: BrokerOrderSide | None = None
    position_side: BrokerPositionSide | None = None
    quantity: Decimal | None = None
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    reduce_only: bool = False
    close_position: bool = False
    client_order_id: str | None = None
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerSemanticResult:
    exchange: ExchangeName
    symbol: str
    action: BrokerSemanticAction
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.UNKNOWN
    ok: bool = False
    message: str = ""
    order_id: str | None = None
    related_order_ids: tuple[str, ...] = ()
    status: BrokerOrderStatus | None = None
    filled_quantity: Decimal | None = None
    avg_price: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerSemanticOrderQuery:
    exchange: ExchangeName
    symbol: str
    roles: tuple[BrokerSemanticOrderRole, ...] = ()
    include_ordinary: bool = True
    include_algo: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
