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
        "tests/exchanges/binance/test_binance_aiohttp_transport.py",
        "tests/exchanges/binance/test_binance_aiohttp_transport_boundaries.py",
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
