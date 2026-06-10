from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

import src.execution.okx_private_client as client_module  # noqa: E402
from src.execution.okx_private_client import OkxPrivateClient, OkxPrivateClientConfig  # noqa: E402
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


from src.execution.trader import parse_allowed_live_symbols  # noqa: E402


class ParseAllowedLiveSymbolsTest(unittest.TestCase):
    """Unit tests for the pure helper."""

    def test_none_returns_default(self) -> None:
        self.assertEqual(parse_allowed_live_symbols(None), ("ETH-USDT-SWAP",))

    def test_empty_string_returns_default(self) -> None:
        self.assertEqual(parse_allowed_live_symbols(""), ("ETH-USDT-SWAP",))

    def test_single_eth(self) -> None:
        self.assertEqual(parse_allowed_live_symbols("ETH-USDT-SWAP"), ("ETH-USDT-SWAP",))

    def test_multiple_comma_separated(self) -> None:
        self.assertEqual(
            parse_allowed_live_symbols("ETH-USDT-SWAP, BTC-USDT-SWAP"),
            ("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
        )

    def test_dedup_keeps_first(self) -> None:
        self.assertEqual(
            parse_allowed_live_symbols("ETH-USDT-SWAP,ETH-USDT-SWAP"),
            ("ETH-USDT-SWAP",),
        )

    def test_wildcard_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_allowed_live_symbols("*")

    def test_internal_whitespace_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_allowed_live_symbols("ETH USDT SWAP")

    def test_tab_whitespace_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_allowed_live_symbols("ETH\tUSDT-SWAP")

    def test_newline_whitespace_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_allowed_live_symbols("ETH\nUSDT-SWAP")

    def test_blank_with_spaces_returns_default(self) -> None:
        self.assertEqual(parse_allowed_live_symbols("   "), ("ETH-USDT-SWAP",))

    def test_trailing_comma_handled(self) -> None:
        self.assertEqual(
            parse_allowed_live_symbols("ETH-USDT-SWAP,"),
            ("ETH-USDT-SWAP",),
        )


class TraderAllowlistIntegrationTest(unittest.TestCase):
    """Integration tests for the Trader live-symbol gate (no OKX requests)."""

    def setUp(self) -> None:
        # Save original environ so we can restore it.
        self._original_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)

    @staticmethod
    def _fake_okx_config() -> dict[str, str]:
        return {"api_key": "fake-key", "secret_key": "fake-secret", "passphrase": "fake-pass"}

    @staticmethod
    def _trader_module():
        return sys.modules["src.execution.trader"]

    def test_default_env_allows_eth_only(self) -> None:
        """OKX_INST_ID unset, RECLAIM_ALLOWED_LIVE_SYMBOLS unset → ETH only."""
        env = {
            "LIVE_TRADING": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self._trader_module(), "OKX_CONFIG", self._fake_okx_config()):
                t = self._trader_module().Trader()
                self.assertEqual(t.symbol, "ETH-USDT-SWAP")
                self.assertEqual(t.allowed_live_symbols, ("ETH-USDT-SWAP",))

    def test_non_allowlisted_symbol_rejected(self) -> None:
        """BTC requested but not in allowlist → RuntimeError with correct message."""
        env = {
            "OKX_INST_ID": "BTC-USDT-SWAP",
            # RECLAIM_ALLOWED_LIVE_SYMBOLS intentionally unset
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                self._trader_module().Trader()
            msg = str(ctx.exception)
            self.assertIn("BTC-USDT-SWAP", msg)
            self.assertIn("RECLAIM_ALLOWED_LIVE_SYMBOLS", msg)

    def test_explicit_allowlist_passes_symbol_gate_for_btc(self) -> None:
        """BTC explicitly allowlisted → Trader passes symbol gate (no OKX calls)."""
        env = {
            "OKX_INST_ID": "BTC-USDT-SWAP",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self._trader_module(), "OKX_CONFIG", self._fake_okx_config()):
                t = self._trader_module().Trader()
                self.assertEqual(t.symbol, "BTC-USDT-SWAP")
                self.assertEqual(
                    t.allowed_live_symbols,
                    ("ETH-USDT-SWAP", "BTC-USDT-SWAP"),
                )
                # Prove we didn't initialize or call OKX
                self.assertFalse(hasattr(t, "account_equity_usdt") and t.account_equity_usdt != 0.0)


class TraderSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_request_reuses_session(self) -> None:
        FakeSession.instances = []
        trader = Trader.__new__(Trader)
        trader.base_url = "https://www.okx.test"
        trader.api_key = "key"
        trader.secret_key = "secret"
        trader.passphrase = "pass"
        trader._timeout_seconds = 7.0
        trader._client = OkxPrivateClient(
            OkxPrivateClientConfig(
                base_url=trader.base_url,
                api_key=trader.api_key,
                secret_key=trader.secret_key,
                passphrase=trader.passphrase,
                timeout_seconds=trader._timeout_seconds,
            )
        )

        with patch.object(client_module.aiohttp, "ClientSession", FakeSession, create=True):
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
