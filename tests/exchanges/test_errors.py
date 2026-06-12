#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_errors.py
@Description: Tests for src.exchanges.errors – ExchangeError.
"""

from __future__ import annotations

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import ExchangeName


class TestExchangeError:
    def test_stores_all_fields(self) -> None:
        err = ExchangeError(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.RATE_LIMITED,
            message="Too many requests",
            raw={"retry_after": "2"},
        )
        assert err.exchange == ExchangeName.OKX
        assert err.kind == ExchangeErrorKind.RATE_LIMITED
        assert err.message == "Too many requests"
        assert err.raw == {"retry_after": "2"}

    def test_str_contains_exchange_kind_and_message(self) -> None:
        err = ExchangeError(
            exchange=ExchangeName.BINANCE,
            kind=ExchangeErrorKind.AUTH_ERROR,
            message="Invalid API key",
        )
        s = str(err)
        assert ExchangeName.BINANCE.value in s
        assert ExchangeErrorKind.AUTH_ERROR.value in s
        assert "Invalid API key" in s

    def test_raw_defaults_to_empty_mapping(self) -> None:
        err = ExchangeError(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.NETWORK_ERROR,
            message="Connection timeout",
        )
        assert isinstance(err.raw, dict)
        assert len(err.raw) == 0

    def test_all_error_kinds_are_distinct(self) -> None:
        """Ensure every ExchangeErrorKind value is unique."""
        values = [e.value for e in ExchangeErrorKind]
        assert len(values) == len(set(values))

    def test_is_exception(self) -> None:
        err = ExchangeError(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.UNKNOWN,
            message="???",
        )
        assert isinstance(err, Exception)
