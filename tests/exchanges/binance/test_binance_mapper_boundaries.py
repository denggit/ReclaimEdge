#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_mapper_boundaries.py
@Description: Source-level boundary tests for the Binance response mapper.

Ensures the mapper has no live execution, network, strategy, risk, reporting,
config, API credential, or request-mapping dependencies.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# No live / network / strategy / config / credential dependency
# ---------------------------------------------------------------------------

def test_binance_mapper_has_no_live_execution_or_network_dependency() -> None:
    text = Path("src/exchanges/binance/mapper.py").read_text(encoding="utf-8")

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
        assert token not in text, f"{token} should not appear in Binance mapper"


# ---------------------------------------------------------------------------
# No request mapper yet
# ---------------------------------------------------------------------------

def test_binance_mapper_does_not_implement_request_mapping_yet() -> None:
    text = Path("src/exchanges/binance/mapper.py").read_text(encoding="utf-8")

    forbidden = [
        "broker_order_request_to_binance",
        "map_broker_order_request",
        "to_binance_order_params",
        "place_order_params",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not be implemented in 15A"
