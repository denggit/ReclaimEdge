#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_aiohttp_transport_boundaries.py
@Description: Source-level guards ensuring aiohttp_transport.py has no live,
              factory, env, or runtime wiring dependency.
"""

from __future__ import annotations

from pathlib import Path


def test_aiohttp_transport_has_no_live_factory_or_env_dependency() -> None:
    text = Path("src/exchanges/binance/aiohttp_transport.py").read_text(encoding="utf-8")

    forbidden = [
        "src.live",
        "src.execution",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "config.",
        "OKX_CONFIG",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "os.environ",
        "dotenv",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in aiohttp_transport.py"


def test_aiohttp_transport_is_not_wired_into_factory_or_live() -> None:
    allowed = {
        "src/exchanges/binance/__init__.py",
        "src/exchanges/binance/aiohttp_transport.py",
        "src/exchanges/binance/algo_orders.py",
        "tests/exchanges/binance/test_binance_aiohttp_transport.py",
        "tests/exchanges/binance/test_binance_aiohttp_transport_boundaries.py",
        "tests/exchanges/binance/test_binance_algo_orders.py",
        # Broker runtime factory boundary test uses the token in its
        # forbidden-import assertion list — not a real wiring.
        "tests/exchanges/test_broker_runtime_factory_boundaries.py",
        # Smoke test script and its tests legitimately reference
        # AiohttpBinanceTransport for real (but opt-in) REST calls.
        "scripts/binance_live_smoke_test.py",
        "tests/scripts/test_binance_live_smoke_test.py",
        "tests/scripts/test_binance_live_smoke_test_boundaries.py",
        # Unified runtime config test references AiohttpBinanceTransport
        # in a forbidden-import assertion check — not a real wiring.
        "tests/exchanges/test_unified_eth_runtime_config.py",
        # Read-only smoke test script and its tests legitimately reference
        # AiohttpBinanceTransport for read-only signed REST calls.
        "scripts/binance_read_only_smoke_test.py",
        "tests/scripts/test_binance_read_only_smoke_test.py",
        "tests/scripts/test_binance_read_only_smoke_test_boundaries.py",
        # BinanceLiveTrader legitimately uses AiohttpBinanceTransport.
        "src/execution/binance_live_trader.py",
        "tests/execution/test_binance_live_trader.py",
    }

    for path in Path(".").rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue

        file_name = path.as_posix()
        text = path.read_text(encoding="utf-8")

        if "AiohttpBinanceTransport" in text:
            assert file_name in allowed, (
                f"AiohttpBinanceTransport must not be wired into runtime yet; found in {file_name}"
            )
