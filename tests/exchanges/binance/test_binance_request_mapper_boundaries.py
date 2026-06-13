#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_request_mapper_boundaries.py
@Description: Source-level boundary tests for the Binance request mapper.

Ensures the request mapper has no live execution, network, strategy, risk,
reporting, config, API credential, or Trader/factory dependencies.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# No live / network / strategy / config / credential dependency
# ---------------------------------------------------------------------------


def test_binance_request_mapper_has_no_live_execution_or_network_dependency() -> None:
    text = Path("src/exchanges/binance/request_mapper.py").read_text(encoding="utf-8")

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
        "requests",
        "aiohttp",
        "httpx",
        "websockets",
        "/fapi/",
        "/dapi/",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance request mapper"
