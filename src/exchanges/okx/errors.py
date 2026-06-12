from __future__ import annotations

import asyncio
import ast
from typing import Any, Mapping

from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import ExchangeName


_AUTH_CODES = frozenset(str(code) for code in range(50100, 50120))
_RATE_LIMIT_CODES = frozenset({"50011", "50040"})
_SERVER_ERROR_CODES = frozenset({"50000", "50001", "50002", "50004", "50005", "50013"})
_BAD_REQUEST_CODES = frozenset(
    {"51000", "51001", "51002", "51003", "51004", "51005", "51006", "51007", "51009"}
)
_INSUFFICIENT_MARGIN_CODES = frozenset({"51008", "51031"})
_ORDER_NOT_FOUND_CODES = frozenset({"51603", "51604"})

_AUTH_KEYWORDS = (
    "api key",
    "api-key",
    "passphrase",
    "signature",
    "sign",
    "permission",
    "auth",
    "unauthorized",
    "invalid key",
)
_RATE_LIMIT_KEYWORDS = (
    "rate limit",
    "too many requests",
    "frequency",
    "requests too frequent",
)
_TIMEOUT_KEYWORDS = ("timeout", "timed out")
_NETWORK_KEYWORDS = (
    "connection",
    "network",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
)
_MAINTENANCE_KEYWORDS = (
    "maintenance",
    "under maintenance",
    "service unavailable",
    "system maintenance",
    "exchange maintenance",
)
_SERVER_MAINTENANCE_KEYWORDS = _MAINTENANCE_KEYWORDS + ("unavailable",)
_TRIGGER_PRICE_KEYWORDS = (
    "trigger",
    "stop",
    "sltriggerpx",
    "tptriggerpx",
    "stop price",
    "trigger price",
)
_ORDER_NOT_FOUND_KEYWORDS = (
    "order not exist",
    "order does not exist",
    "order not found",
    "not found",
    "cannot find order",
    "order not exists",
)
_POSITION_NOT_FOUND_KEYWORDS = (
    "position not found",
    "no position",
    "position does not exist",
)
_DUPLICATE_ORDER_KEYWORDS = (
    "duplicate",
    "duplicated",
    "client order id exists",
    "clordid exists",
    "already exists",
)
_ORDER_ALREADY_DONE_KEYWORDS = (
    "already filled",
    "already canceled",
    "already cancelled",
    "order already completed",
    "order completed",
    "order finished",
)
_REDUCE_ONLY_KEYWORDS = ("reduce-only", "reduce only", "reduceonly")
_POSITION_MODE_KEYWORDS = (
    "posside",
    "pos side",
    "position side",
    "long_short",
    "long short",
    "net mode",
    "position mode",
    "hedge mode",
)
_INVALID_SYMBOL_KEYWORDS = (
    "instid",
    "inst id",
    "instrument",
    "symbol",
    "not exist",
    "unavailable",
    "does not exist",
    "invalid inst",
)
_INVALID_QUANTITY_KEYWORDS = (
    "size",
    "sz",
    "quantity",
    "amount",
    "lot",
    "min size",
    "minimum size",
    "order size",
    "contract size",
)
_INVALID_PRICE_KEYWORDS = ("price", "px", "tick size", "price precision")
_INSUFFICIENT_MARGIN_KEYWORDS = (
    "insufficient",
    "margin",
    "balance",
    "available",
    "not enough",
    "account balance",
    "insufficient balance",
    "insufficient margin",
)


def okx_error_kind_from_code(
    code: str | int | None,
    message: str | None = None,
) -> ExchangeErrorKind:
    """Map OKX raw ``code`` / ``sCode`` and message text to a unified kind."""
    normalized_code = _normalize_code(code)
    normalized_message = _normalize_message(message)

    if normalized_code == "0":
        return ExchangeErrorKind.UNKNOWN

    kind = _specific_message_kind(normalized_message)
    if normalized_code in _AUTH_CODES:
        return ExchangeErrorKind.AUTH_ERROR
    if normalized_code in _RATE_LIMIT_CODES:
        return ExchangeErrorKind.RATE_LIMITED
    if normalized_code in _INSUFFICIENT_MARGIN_CODES:
        return ExchangeErrorKind.INSUFFICIENT_MARGIN
    if normalized_code in _ORDER_NOT_FOUND_CODES:
        return ExchangeErrorKind.ORDER_NOT_FOUND
    if normalized_code in _SERVER_ERROR_CODES:
        if _contains_any(normalized_message, _SERVER_MAINTENANCE_KEYWORDS):
            return ExchangeErrorKind.EXCHANGE_MAINTENANCE
        return _kind_if_enum_exists("SERVER_ERROR", ExchangeErrorKind.EXCHANGE_MAINTENANCE)
    if normalized_code in _BAD_REQUEST_CODES:
        return kind or _kind_if_enum_exists("BAD_REQUEST", ExchangeErrorKind.ORDER_REJECTED)
    if normalized_code and (
        normalized_code.startswith("510")
        or normalized_code.startswith("515")
        or normalized_code.startswith("516")
    ):
        return kind or ExchangeErrorKind.ORDER_REJECTED

    return kind or ExchangeErrorKind.UNKNOWN


def okx_retryable_from_kind(
    kind: ExchangeErrorKind,
    code: str | int | None = None,
    message: str | None = None,
) -> bool:
    """Return whether an OKX error is worth retrying shortly."""
    del code
    if kind in {
        ExchangeErrorKind.RATE_LIMITED,
        ExchangeErrorKind.NETWORK_ERROR,
        ExchangeErrorKind.REQUEST_TIMEOUT,
        ExchangeErrorKind.EXCHANGE_MAINTENANCE,
        ExchangeErrorKind.SERVER_ERROR,
    }:
        return True
    if kind == ExchangeErrorKind.UNKNOWN:
        return _contains_any(
            _normalize_message(message),
            _TIMEOUT_KEYWORDS + _NETWORK_KEYWORDS,
        )
    return False


def okx_error_detail_from_response(
    response: Mapping[str, Any],
    *,
    message: str | None = None,
) -> ExchangeErrorDetail | None:
    """Parse OKX response-level or item-level errors into an error detail."""
    if not isinstance(response, Mapping):
        resolved_message = message or f"Malformed OKX API response type={type(response).__name__}"
        kind = ExchangeErrorKind.UNKNOWN
        return ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=kind,
            message=resolved_message,
            retryable=okx_retryable_from_kind(kind, message=resolved_message),
            raw={"response_type": type(response).__name__, "response": response},
        )

    raw = dict(response)
    top_code = _normalize_code(response.get("code"))
    top_msg = _string_or_none(response.get("msg"))

    if top_code and top_code != "0":
        resolved_message = _join_message(message, top_msg) or f"OKX API error code={top_code}"
        kind = okx_error_kind_from_code(top_code, resolved_message)
        return ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=kind,
            message=resolved_message,
            code=top_code,
            retryable=okx_retryable_from_kind(kind, code=top_code, message=resolved_message),
            raw=raw,
        )

    item_error = _first_item_error(response.get("data"))
    if item_error is None:
        return None

    s_code, s_msg = item_error
    resolved_message = _join_message(message, s_msg) or f"OKX API item error sCode={s_code}"
    kind = okx_error_kind_from_code(s_code, resolved_message)
    return ExchangeErrorDetail(
        exchange=ExchangeName.OKX,
        kind=kind,
        message=resolved_message,
        code=s_code,
        retryable=okx_retryable_from_kind(kind, code=s_code, message=resolved_message),
        raw=raw,
    )


def raise_okx_exchange_error_from_response(
    response: Mapping[str, Any],
    *,
    message: str | None = None,
) -> None:
    """Raise ``ExchangeError`` when an OKX response contains an error."""
    detail = okx_error_detail_from_response(response, message=message)
    if detail is not None:
        raise ExchangeError(detail)


def okx_exception_to_exchange_error(
    exc: Exception,
    *,
    message: str | None = None,
) -> ExchangeError:
    """Wrap exceptions raised near the OKX adapter boundary as ``ExchangeError``."""
    if isinstance(exc, ExchangeError):
        return exc

    exc_message = str(exc)
    resolved_message = _join_message(message, exc_message) or exc_message

    response = _extract_okx_response_from_exception_message(exc_message)
    if response is not None:
        detail = okx_error_detail_from_response(response, message=message)
        if detail is not None:
            raw = dict(detail.raw)
            raw["response"] = response
            raw["exception_type"] = type(exc).__name__
            raw["exception_message"] = exc_message
            return ExchangeError(
                ExchangeErrorDetail(
                    exchange=detail.exchange,
                    kind=detail.kind,
                    message=detail.message,
                    code=detail.code,
                    retryable=detail.retryable,
                    raw=raw,
                )
            )

    kind = _exception_kind(exc, resolved_message)
    return ExchangeError(
        ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=kind,
            message=resolved_message,
            retryable=okx_retryable_from_kind(kind, message=resolved_message),
            raw={
                "exception_type": type(exc).__name__,
                "exception_message": exc_message,
            },
        )
    )


def _exception_kind(exc: Exception, message: str) -> ExchangeErrorKind:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return _kind_if_enum_exists("REQUEST_TIMEOUT", ExchangeErrorKind.NETWORK_ERROR)
    if _is_aiohttp_client_error(exc):
        return ExchangeErrorKind.NETWORK_ERROR
    if isinstance(exc, (ConnectionError, OSError)) and _contains_any(
        _normalize_message(message),
        _NETWORK_KEYWORDS + _TIMEOUT_KEYWORDS,
    ):
        return ExchangeErrorKind.NETWORK_ERROR
    return okx_error_kind_from_code(None, message)


def _specific_message_kind(message: str) -> ExchangeErrorKind | None:
    if not message:
        return None
    if _contains_any(message, _AUTH_KEYWORDS):
        return ExchangeErrorKind.AUTH_ERROR
    if _contains_any(message, _RATE_LIMIT_KEYWORDS):
        return ExchangeErrorKind.RATE_LIMITED
    if _contains_any(message, _TIMEOUT_KEYWORDS):
        return _kind_if_enum_exists("REQUEST_TIMEOUT", ExchangeErrorKind.NETWORK_ERROR)
    if _contains_any(message, _MAINTENANCE_KEYWORDS):
        return ExchangeErrorKind.EXCHANGE_MAINTENANCE
    if _contains_any(message, _NETWORK_KEYWORDS):
        return ExchangeErrorKind.NETWORK_ERROR
    if _contains_any(message, _ORDER_NOT_FOUND_KEYWORDS):
        return ExchangeErrorKind.ORDER_NOT_FOUND
    if _contains_any(message, _POSITION_NOT_FOUND_KEYWORDS):
        return ExchangeErrorKind.POSITION_NOT_FOUND
    if _contains_any(message, _DUPLICATE_ORDER_KEYWORDS):
        return _kind_if_enum_exists("DUPLICATE_ORDER", ExchangeErrorKind.ORDER_REJECTED)
    if _contains_any(message, _ORDER_ALREADY_DONE_KEYWORDS):
        return _kind_if_enum_exists("ORDER_ALREADY_DONE", ExchangeErrorKind.ORDER_REJECTED)
    if _contains_any(message, _REDUCE_ONLY_KEYWORDS):
        return ExchangeErrorKind.REDUCE_ONLY_REJECTED
    if _contains_any(message, _POSITION_MODE_KEYWORDS):
        return ExchangeErrorKind.POSITION_MODE_MISMATCH
    if _contains_any(message, _TRIGGER_PRICE_KEYWORDS):
        return ExchangeErrorKind.INVALID_TRIGGER_PRICE
    if _contains_any(message, _INVALID_SYMBOL_KEYWORDS):
        return ExchangeErrorKind.INVALID_SYMBOL
    if _contains_any(message, _INSUFFICIENT_MARGIN_KEYWORDS):
        return ExchangeErrorKind.INSUFFICIENT_MARGIN
    if _contains_any(message, _INVALID_QUANTITY_KEYWORDS):
        return ExchangeErrorKind.INVALID_QUANTITY
    if _contains_any(message, _INVALID_PRICE_KEYWORDS):
        return ExchangeErrorKind.INVALID_PRICE
    return None


def _extract_okx_response_from_exception_message(text: str) -> Mapping[str, Any] | None:
    marker = "response="
    if marker not in text:
        return None
    response_text = text.split(marker, 1)[1].strip()
    try:
        parsed = ast.literal_eval(response_text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _first_item_error(data: Any) -> tuple[str, str | None] | None:
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, Mapping):
            continue
        s_code = _normalize_code(item.get("sCode"))
        if s_code and s_code != "0":
            return s_code, _string_or_none(item.get("sMsg"))
    return None


def _normalize_code(code: str | int | None) -> str:
    if code is None:
        return ""
    return str(code).strip()


def _normalize_message(message: str | None) -> str:
    return str(message or "").lower()


def _contains_any(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in message for keyword in keywords)


def _kind_if_enum_exists(name: str, fallback: ExchangeErrorKind) -> ExchangeErrorKind:
    return ExchangeErrorKind.__members__.get(name, fallback)


def _join_message(prefix: str | None, detail: str | None) -> str:
    prefix = str(prefix).strip() if prefix else ""
    detail = str(detail).strip() if detail else ""
    if prefix and detail:
        return f"{prefix}: {detail}"
    return prefix or detail


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _is_aiohttp_client_error(exc: Exception) -> bool:
    try:
        import aiohttp
    except ImportError:
        return False
    return isinstance(exc, aiohttp.ClientError)
