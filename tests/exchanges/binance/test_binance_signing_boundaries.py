#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_signing_boundaries.py
@Description: Boundary / source-level tests for the Binance signing helper.
"""

from __future__ import annotations

from pathlib import Path


def test_binance_signing_has_no_live_execution_or_network_dependency() -> None:
    text = Path("src/exchanges/binance/signing.py").read_text(encoding="utf-8")

    forbidden = [
        "src.execution",
        "src.live",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "config.",
        "OKX_CONFIG",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "os.environ",
        "import requests",
        "from requests",
        "import aiohttp",
        "from aiohttp",
        "import httpx",
        "from httpx",
        "import websockets",
        "from websockets",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance signing helper"


def test_binance_signing_does_not_send_http_requests() -> None:
    text = Path("src/exchanges/binance/signing.py").read_text(encoding="utf-8")

    forbidden = [
        ".request(",
        ".get(",
        ".post(",
        ".delete(",
        "urlopen",
        "HTTPConnection",
        "AsyncClient",
        "ClientSession",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in signing helper"
