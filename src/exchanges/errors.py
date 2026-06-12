from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.exchanges.models import ExchangeName


class ExchangeErrorKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    NETWORK_ERROR = "NETWORK_ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_TRIGGER_PRICE = "INVALID_TRIGGER_PRICE"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    ORDER_REJECTED = "ORDER_REJECTED"
    REDUCE_ONLY_REJECTED = "REDUCE_ONLY_REJECTED"
    POSITION_MODE_MISMATCH = "POSITION_MODE_MISMATCH"
    POSITION_NOT_FOUND = "POSITION_NOT_FOUND"
    EXCHANGE_MAINTENANCE = "EXCHANGE_MAINTENANCE"


@dataclass(frozen=True)
class ExchangeErrorDetail:
    exchange: ExchangeName
    kind: ExchangeErrorKind
    message: str
    code: str | None = None
    retryable: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


class ExchangeError(RuntimeError):
    def __init__(self, detail: ExchangeErrorDetail):
        self.detail = detail
        super().__init__(f"{detail.exchange.value} {detail.kind.value}: {detail.message}")
