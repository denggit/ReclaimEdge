#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_transport_boundaries.py
@Description: Source-level guards ensuring transport.py has no real HTTP
              dependency and client.py has no live / factory / strategy wiring.
"""

from __future__ import annotations

from pathlib import Path


def test_binance_transport_has_no_real_http_dependency() -> None:
    text = Path("src/exchanges/binance/transport.py").read_text(encoding="utf-8")

    forbidden = [
        "requests",
        "aiohttp",
        "httpx",
        "websockets",
        "urlopen",
        "HTTPConnection",
        "AsyncClient",
        "ClientSession",
        "os.environ",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "src.live",
        "src.execution",
        "config.",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in transport.py"


def test_binance_client_has_no_live_or_factory_dependency() -> None:
    text = Path("src/exchanges/binance/client.py").read_text(encoding="utf-8")

    forbidden = [
        "src.live",
        "src.execution",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "config.",
        "OKX_CONFIG",
        "os.environ",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance client"
