#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_broker_runtime_factory_boundaries.py
@Description: Boundary / safety checks for the exchange broker runtime factory.

These tests verify that ``src/exchanges/factory.py`` does **not** depend on
live infrastructure, environment variables, real transports, or the live
entrypoint.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Source-level forbidden-token checks
# ---------------------------------------------------------------------------


def test_broker_runtime_factory_has_no_live_env_or_real_transport_dependency() -> None:
    """Verify factory.py does not import or reference live/real-transport tokens."""
    text = Path("src/exchanges/factory.py").read_text(encoding="utf-8")

    forbidden = [
        "src.live",
        "src.execution",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "scripts.run_boll_cvd_live",
        "OKX_CONFIG",
        "os.environ",
        "dotenv",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "EXCHANGE_API_PASSPHRASE",
        "AiohttpBinanceTransport",
        "websockets",
        "requests",
        "httpx",
        "aiohttp",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in exchange factory"


def test_runtime_factory_is_not_wired_into_live_entrypoint_yet() -> None:
    """Verify the live entrypoint does not import factory symbols yet."""
    text = Path("scripts/run_boll_cvd_live.py").read_text(encoding="utf-8")

    forbidden = [
        "build_broker_client",
        "build_broker_semantic_executor",
        "normalize_exchange_name",
        "BinanceBrokerClient",
        "BinanceBrokerSemanticExecutor",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not be wired into live entrypoint yet"
