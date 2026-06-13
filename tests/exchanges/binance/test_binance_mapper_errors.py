#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_mapper_errors.py
@Description: Unit tests for Binance error response mapping.
"""

from __future__ import annotations

from src.exchanges.binance.mapper import map_binance_error
from src.exchanges.errors import ExchangeErrorKind
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# AUTH_ERROR
# ---------------------------------------------------------------------------

def test_status_401_is_auth_error() -> None:
    err = map_binance_error(status_code=401, payload={"msg": "Invalid API key"})
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.AUTH_ERROR
    assert "Invalid API key" in err.message


def test_status_403_is_auth_error() -> None:
    err = map_binance_error(status_code=403, payload={"msg": "Forbidden"})
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.AUTH_ERROR


def test_code_minus_2015_is_auth_error() -> None:
    """Binance error code -2015: Invalid API-key, IP, or permissions."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -2015, "msg": "Invalid API-key, IP, or permissions for this action."},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.AUTH_ERROR


def test_code_minus_1022_is_auth_error() -> None:
    """Binance error code -1022: Signature for this request is not valid."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -1022, "msg": "Signature for this request is not valid."},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.AUTH_ERROR


# ---------------------------------------------------------------------------
# RATE_LIMITED
# ---------------------------------------------------------------------------

def test_status_418_is_rate_limited() -> None:
    """HTTP 418: IP banned or rate limited."""
    err = map_binance_error(status_code=418, payload={"msg": "IP banned"})
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.RATE_LIMITED


def test_status_429_is_rate_limited() -> None:
    """HTTP 429: Too many requests."""
    err = map_binance_error(status_code=429, payload={"msg": "Too many requests"})
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.RATE_LIMITED


def test_code_minus_1003_is_rate_limited() -> None:
    """Binance error code -1003: Too many requests / rate limit."""
    err = map_binance_error(
        status_code=418,
        payload={"code": -1003, "msg": "Too much request weight used; please use the websocket for live updates."},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.RATE_LIMITED


# ---------------------------------------------------------------------------
# INSUFFICIENT_BALANCE
# ---------------------------------------------------------------------------

def test_code_minus_2019_is_insufficient_balance() -> None:
    """Binance error code -2019: Margin is insufficient."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -2019, "msg": "Margin is insufficient."},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.INSUFFICIENT_BALANCE


# ---------------------------------------------------------------------------
# ORDER_NOT_FOUND
# ---------------------------------------------------------------------------

def test_code_minus_2011_is_order_not_found() -> None:
    """Binance error code -2011: Unknown order sent."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -2011, "msg": "Unknown order sent."},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.ORDER_NOT_FOUND


# ---------------------------------------------------------------------------
# EXCHANGE_REJECTED (default for known-but-uncategorised codes)
# ---------------------------------------------------------------------------

def test_unknown_error_code_defaults_to_exchange_rejected() -> None:
    """Any Binance error code not explicitly mapped -> EXCHANGE_REJECTED."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -99999, "msg": "Some obscure error"},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.EXCHANGE_REJECTED


def test_no_code_field_defaults_to_exchange_rejected() -> None:
    """Payload with msg but no code field -> EXCHANGE_REJECTED."""
    err = map_binance_error(
        status_code=400,
        payload={"msg": "Bad request"},
    )
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.EXCHANGE_REJECTED


# ---------------------------------------------------------------------------
# UNKNOWN (empty / no info)
# ---------------------------------------------------------------------------

def test_empty_payload_and_no_status_defaults_to_unknown() -> None:
    """No status_code and empty payload -> UNKNOWN."""
    err = map_binance_error()
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.UNKNOWN


def test_empty_payload_none_status_defaults_to_unknown() -> None:
    """Explicit None status_code and empty payload -> UNKNOWN."""
    err = map_binance_error(status_code=None, payload={})
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.UNKNOWN


# ---------------------------------------------------------------------------
# Message precedence
# ---------------------------------------------------------------------------

def test_message_param_takes_precedence_over_payload_msg() -> None:
    """Explicit message parameter wins over payload msg."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -2011, "msg": "Original msg"},
        message="Custom override",
    )
    assert err.message == "Custom override"


def test_falls_back_to_payload_message_field() -> None:
    """When 'msg' is absent, fall back to 'message' field."""
    err = map_binance_error(
        status_code=400,
        payload={"code": -99999, "message": "From message field"},
    )
    assert err.message == "From message field"


# ---------------------------------------------------------------------------
# Raw preservation
# ---------------------------------------------------------------------------

def test_raw_preserves_status_code_and_payload() -> None:
    payload = {"code": -2011, "msg": "Unknown order sent."}
    err = map_binance_error(status_code=400, payload=payload)

    assert err.raw["status_code"] == 400
    assert err.raw["payload"] == payload
