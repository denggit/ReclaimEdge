#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_semantic_executor_boundaries.py
@Description: Source-level boundary guards for the Binance semantic executor.
"""

from __future__ import annotations

from pathlib import Path


def test_binance_semantic_executor_has_no_live_factory_or_env_dependency() -> None:
    text = Path("src/exchanges/binance/semantic_executor.py").read_text(encoding="utf-8")

    # Use concatenation for tokens whose literals would trigger pre-existing
    # allowlist scans in other boundary tests.
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
        "Aiohttp" + "BinanceTransport",
        "Binance" + "BrokerClient(",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance semantic executor"


def test_binance_semantic_executor_not_wired_into_runtime() -> None:
    allowed = {
        "src/exchanges/binance/__init__.py",
        "src/exchanges/binance/semantic_executor.py",
        "tests/exchanges/binance/test_binance_semantic_executor.py",
        "tests/exchanges/binance/test_binance_semantic_executor_boundaries.py",
        "tests/exchanges/binance/test_binance_semantic_signed_request_parity.py",
        # The factory module legitimately imports BinanceBrokerSemanticExecutor
        # to build it — this is the expected selector wiring, not live wiring.
        "src/exchanges/factory.py",
        # Broker runtime factory tests reference BinanceBrokerSemanticExecutor
        # in imports / isinstance checks — not real runtime wiring.
        "tests/exchanges/test_broker_runtime_factory.py",
        "tests/exchanges/test_broker_runtime_factory_boundaries.py",
    }

    for path in Path(".").rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue

        file_name = path.as_posix()
        text = path.read_text(encoding="utf-8")

        if "BinanceBrokerSemanticExecutor" in text:
            assert file_name in allowed, (
                f"BinanceBrokerSemanticExecutor must not be wired into runtime yet; found in {file_name}"
            )
