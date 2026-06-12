#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : client.py
@Description: OKX broker client – skeleton placeholder.

This module exists only so that the adapter can be imported and tested.
It does NOT implement any real API calls, does NOT import Trader, and
does NOT affect the live trading path.
"""

from __future__ import annotations

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import ExchangeName


class OkxBrokerClientNotWired:
    """Explicit placeholder – raises on any attempt to use it."""

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.OKX

    def _not_wired(self) -> None:
        raise ExchangeError(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
            message="OKX broker client is not wired in this skeleton step.",
        )


__all__ = ["OkxBrokerClientNotWired"]
