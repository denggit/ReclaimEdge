#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : factory.py
@Description: Placeholder factory for exchange adapters.

The factory is intentionally empty in this skeleton step.  It will be
wired once real adapters (OkxBrokerClient, BinanceBrokerClient, …) are
implemented in later tasks.
"""

from __future__ import annotations

from src.exchanges.models import ExchangeName


def unsupported_exchange_message(exchange: ExchangeName) -> str:
    """Return a human-readable message for an unsupported exchange."""
    return f"Exchange adapter is not wired yet: {exchange.value}"
