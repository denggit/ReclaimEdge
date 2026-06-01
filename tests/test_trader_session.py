from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

import src.execution.trader as trader_module  # noqa: E402
from src.execution.trader import Trader  # noqa: E402


class FakeResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        pass

    async def json(self) -> dict[str, object]:
        return {"code": "0", "data": []}


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self) -> None:
        self.closed = False
        self.get_calls: list[dict[str, object]] = []
        self.post_calls: list[dict[str, object]] = []
        FakeSession.instances.append(self)

    def get(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        self.get_calls.append({"url": url, **kwargs})
        return FakeResponse()

    def post(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        self.post_calls.append({"url": url, **kwargs})
        return FakeResponse()

    async def close(self) -> None:
        self.closed = True


class TraderSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_request_reuses_session(self) -> None:
        FakeSession.instances = []
        trader = Trader.__new__(Trader)
        trader.base_url = "https://www.okx.test"
        trader.api_key = "key"
        trader.secret_key = "secret"
        trader.passphrase = "pass"
        trader._session = None
        trader._timeout_seconds = 7.0

        with patch.object(trader_module.aiohttp, "ClientSession", FakeSession, create=True):
            await trader.request("GET", "/api/v5/account/balance?ccy=USDT")
            await trader.request("POST", "/api/v5/trade/order", {"instId": "ETH-USDT-SWAP"})
            await trader.close()

        self.assertEqual(len(FakeSession.instances), 1)
        session = FakeSession.instances[0]
        self.assertEqual(session.get_calls[0]["timeout"], 7.0)
        self.assertEqual(session.post_calls[0]["timeout"], 7.0)
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
