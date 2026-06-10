from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from decimal import Decimal
from unittest.mock import patch

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

import src.execution.okx_private_client as client_module  # noqa: E402
from src.execution.okx_private_client import OkxPrivateClient, OkxPrivateClientConfig  # noqa: E402
from src.execution.trader import Trader  # noqa: E402
from src.execution.trader import (  # noqa: E402
    DEFAULT_ETH_INSTRUMENT_METADATA,
    TraderInstrumentMetadata,
    default_instrument_metadata_for,
)


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


class TraderInstrumentMetadataTest(unittest.TestCase):
    """Unit tests for TraderInstrumentMetadata and related helpers."""

    # ── DEFAULT_ETH_INSTRUMENT_METADATA fields ──
    def test_default_eth_inst_id(self) -> None:
        self.assertEqual(DEFAULT_ETH_INSTRUMENT_METADATA.inst_id, "ETH-USDT-SWAP")

    def test_default_eth_contract_multiplier(self) -> None:
        self.assertEqual(DEFAULT_ETH_INSTRUMENT_METADATA.contract_multiplier, Decimal("0.1"))

    def test_default_eth_contract_precision(self) -> None:
        self.assertEqual(DEFAULT_ETH_INSTRUMENT_METADATA.contract_precision, Decimal("0.01"))

    def test_default_eth_min_contracts(self) -> None:
        self.assertEqual(DEFAULT_ETH_INSTRUMENT_METADATA.min_contracts, Decimal("0.01"))

    # ── accepts Decimal / str ──
    def test_accepts_decimal_values(self) -> None:
        m = TraderInstrumentMetadata(
            inst_id="ETH-USDT-SWAP",
            contract_multiplier=Decimal("0.1"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        self.assertIsInstance(m.contract_multiplier, Decimal)
        self.assertEqual(m.contract_multiplier, Decimal("0.1"))

    def test_accepts_string_values(self) -> None:
        m = TraderInstrumentMetadata(
            inst_id="ETH-USDT-SWAP",
            contract_multiplier="0.01",
            contract_precision="1",
            min_contracts="1",
        )
        self.assertEqual(m.contract_multiplier, Decimal("0.01"))
        self.assertEqual(m.contract_precision, Decimal("1"))
        self.assertEqual(m.min_contracts, Decimal("1"))

    def test_accepts_int_values(self) -> None:
        m = TraderInstrumentMetadata(
            inst_id="ETH-USDT-SWAP",
            contract_multiplier=1,
            contract_precision=1,
            min_contracts=1,
        )
        self.assertEqual(m.contract_multiplier, Decimal("1"))
        self.assertEqual(m.contract_precision, Decimal("1"))
        self.assertEqual(m.min_contracts, Decimal("1"))

    def test_accepts_float_values(self) -> None:
        m = TraderInstrumentMetadata(
            inst_id="ETH-USDT-SWAP",
            contract_multiplier=0.1,
            contract_precision=0.01,
            min_contracts=0.01,
        )
        self.assertEqual(m.contract_multiplier, Decimal("0.1"))
        self.assertEqual(m.contract_precision, Decimal("0.01"))
        self.assertEqual(m.min_contracts, Decimal("0.01"))

    # ── illegal values ──
    def test_empty_inst_id_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="",
                contract_multiplier="0.1",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_whitespace_inst_id_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="   ",
                contract_multiplier="0.1",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_contract_multiplier_zero_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier=0,
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_contract_precision_zero_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier="0.1",
                contract_precision=0,
                min_contracts="0.01",
            )

    def test_min_contracts_zero_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier="0.1",
                contract_precision="0.01",
                min_contracts=0,
            )

    def test_contract_multiplier_bool_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier=True,
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_contract_multiplier_none_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TraderInstrumentMetadata(
                inst_id="ETH-USDT-SWAP",
                contract_multiplier=None,
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_inst_id_strips_whitespace(self) -> None:
        m = TraderInstrumentMetadata(
            inst_id="  ETH-USDT-SWAP  ",
            contract_multiplier="0.1",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        self.assertEqual(m.inst_id, "ETH-USDT-SWAP")

    # ── default_instrument_metadata_for ──
    def test_default_metadata_for_eth(self) -> None:
        m = default_instrument_metadata_for("ETH-USDT-SWAP")
        self.assertIs(m, DEFAULT_ETH_INSTRUMENT_METADATA)

    def test_default_metadata_for_eth_with_whitespace(self) -> None:
        m = default_instrument_metadata_for("  ETH-USDT-SWAP  ")
        self.assertIs(m, DEFAULT_ETH_INSTRUMENT_METADATA)

    def test_default_metadata_for_btc_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            default_instrument_metadata_for("BTC-USDT-SWAP")
        msg = str(ctx.exception)
        self.assertIn("No default instrument metadata", msg)
        self.assertIn("BTC-USDT-SWAP", msg)

    def test_default_metadata_for_unknown_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            default_instrument_metadata_for("UNKNOWN-SWAP")

    # ── frozen / immutability ──
    def test_metadata_is_frozen(self) -> None:
        m = DEFAULT_ETH_INSTRUMENT_METADATA
        with self.assertRaises(Exception):
            m.contract_multiplier = Decimal("0.2")  # type: ignore[misc]


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

    def test_btc_allowlisted_but_no_default_metadata_rejected(self) -> None:
        """BTC allowlisted but no default metadata → ValueError, not RuntimeError."""
        env = {
            "OKX_INST_ID": "BTC-USDT-SWAP",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self._trader_module(), "OKX_CONFIG", self._fake_okx_config()):
                with self.assertRaises(ValueError) as ctx:
                    self._trader_module().Trader()
                msg = str(ctx.exception)
                self.assertIn("No default instrument metadata", msg)
                self.assertIn("BTC-USDT-SWAP", msg)

    # ── Trader default ETH metadata ──
    def test_default_env_eth_has_correct_metadata(self) -> None:
        """Trader() default ETH → instrument_metadata == DEFAULT_ETH_INSTRUMENT_METADATA."""
        env = {
            "LIVE_TRADING": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self._trader_module(), "OKX_CONFIG", self._fake_okx_config()):
                t = self._trader_module().Trader()
                self.assertIs(t.instrument_metadata, DEFAULT_ETH_INSTRUMENT_METADATA)
                self.assertEqual(t.contract_multiplier, Decimal("0.1"))
                self.assertEqual(t.contract_precision, Decimal("0.01"))
                self.assertEqual(t.min_contracts, Decimal("0.01"))

    # ── explicit BTC metadata injection ──
    def test_explicit_btc_metadata_passes_metadata_gate(self) -> None:
        """Explicit BTC metadata passes metadata gate, no OKX requests."""
        metadata = TraderInstrumentMetadata(
            inst_id="BTC-USDT-SWAP",
            contract_multiplier=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        env = {
            "OKX_INST_ID": "BTC-USDT-SWAP",
            "RECLAIM_ALLOWED_LIVE_SYMBOLS": "ETH-USDT-SWAP,BTC-USDT-SWAP",
            "LIVE_TRADING": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self._trader_module(), "OKX_CONFIG", self._fake_okx_config()):
                t = self._trader_module().Trader(instrument_metadata=metadata)
                self.assertEqual(t.symbol, "BTC-USDT-SWAP")
                self.assertIs(t.instrument_metadata, metadata)
                self.assertEqual(t.contract_multiplier, Decimal("0.01"))
                # Prove we didn't initialize or call OKX
                self.assertFalse(hasattr(t, "account_equity_usdt") and t.account_equity_usdt != 0.0)

    # ── metadata inst_id mismatch ──
    def test_metadata_inst_id_mismatch_rejected(self) -> None:
        """metadata.inst_id != Trader symbol → ValueError."""
        metadata = TraderInstrumentMetadata(
            inst_id="BTC-USDT-SWAP",
            contract_multiplier=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        env = {
            "OKX_INST_ID": "ETH-USDT-SWAP",
            "LIVE_TRADING": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(self._trader_module(), "OKX_CONFIG", self._fake_okx_config()):
                with self.assertRaises(ValueError) as ctx:
                    self._trader_module().Trader(instrument_metadata=metadata)
                msg = str(ctx.exception)
                self.assertIn("does not match", msg.lower())


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
