#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_no_live_wiring.py
@Description: Source-level guard: the Binance shell must not import or reference
              live execution, strategy, config, or real API credential modules.
"""

from __future__ import annotations

from pathlib import Path


def _binance_source_text() -> str:
    root = Path("src/exchanges/binance")
    # signing.py legitimately references fapi / dapi URLs in endpoint constants;
    # it is separately guarded by test_binance_signing_boundaries.py.
    # client.py legitimately references "/fapi/" in endpoint path constants
    # imported from signing.py for building signed requests.
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
        if path.name not in {"signing.py", "client.py", "aiohttp_transport.py"}
    )


def test_binance_adapter_shell_does_not_import_live_execution_or_config_modules() -> None:
    text = _binance_source_text()

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
        "EXCHANGE_API_PASSPHRASE",
        "os.environ",
        "import requests",
        "from requests",
        "import aiohttp",
        "from aiohttp",
        "import httpx",
        "from httpx",
        "import websockets",
        "from websockets",
        "fapi",
        "dapi",
        "/fapi/",
        "/dapi/",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance shell"


def test_binance_adapter_shell_not_wired_into_runtime() -> None:
    allowed = {
        "src/exchanges/binance/__init__.py",
        "src/exchanges/binance/client.py",
        "src/exchanges/binance/errors.py",
        "src/exchanges/binance/transport.py",
        "tests/exchanges/binance/test_binance_broker_client_shell.py",
        "tests/exchanges/binance/test_binance_broker_client_transport.py",
        "tests/exchanges/binance/test_binance_broker_client_position_fetch.py",
        "tests/exchanges/binance/test_binance_no_live_wiring.py",
        "tests/exchanges/binance/test_binance_transport_boundaries.py",
        # Pre-existing mention in docstring; not part of this change.
        "src/exchanges/factory.py",
    }

    for path in Path(".").rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue

        file_name = path.as_posix()
        text = path.read_text(encoding="utf-8")

        if "BinanceBrokerClient" in text:
            assert file_name in allowed, (
                f"BinanceBrokerClient should not be wired into runtime yet; found in {file_name}"
            )
