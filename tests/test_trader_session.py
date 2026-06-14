from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

import src.execution.okx_private_client as client_module  # noqa: E402
from src.execution.okx_private_client import OkxPrivateClient, OkxPrivateClientConfig  # noqa: E402


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
    """Test OkxPrivateClient reuses aiohttp sessions.

    The Trader no longer owns a private REST client or exposes a request()
    tunnel.  Session reuse is tested directly on OkxPrivateClient.
    """

    async def test_request_reuses_session(self) -> None:
        FakeSession.instances = []
        client = OkxPrivateClient(
            OkxPrivateClientConfig(
                base_url="https://www.okx.test",
                api_key="key",
                secret_key="secret",
                passphrase="pass",
                timeout_seconds=7.0,
            )
        )

        with patch.object(client_module.aiohttp, "ClientSession", FakeSession, create=True):
            await client.request("GET", "/api/v5/account/balance?ccy=USDT")
            await client.request("POST", "/api/v5/trade/order", {"instId": "ETH-USDT-SWAP"})
            await client.close()

        self.assertEqual(len(FakeSession.instances), 1)
        session = FakeSession.instances[0]
        self.assertEqual(session.get_calls[0]["timeout"], 7.0)
        self.assertEqual(session.post_calls[0]["timeout"], 7.0)
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
