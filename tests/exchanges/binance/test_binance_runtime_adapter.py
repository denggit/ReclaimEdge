#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_runtime_adapter.py
@Description: Tests for Binance runtime adapter wiring behind preflight gate.

Covers:
  - Default env → RuntimeError (blocked)
  - SIGNAL_ONLY=true → RuntimeError (blocked)
  - All gates satisfied → creates adapters (no real network)
  - Credentials missing → ValueError / RuntimeError (fail fast)
  - Runtime factory generic path dispatches to Binance adapters
  - OKX runtime unaffected by Binance wiring
  - Boundary scan: forbidden paths must NOT import Binance concrete classes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_feed.binance.market_data_client import BinanceMarketDataClient
from src.exchanges.binance.runtime_adapter import create_binance_runtime_adapters
from src.exchanges.binance.trading_client import BinanceTradingClient
from src.exchanges.runtime_config import load_unified_runtime_config
from src.exchanges.models import ExchangeName

# ======================================================================
# Helpers
# ======================================================================

LIVE_PREFLIGHT_PASS_ENV: dict[str, str] = {
    "EXCHANGE": "binance",
    "EXCHANGE_API_KEY": "test-key",
    "EXCHANGE_API_SECRET": "test-secret",
    "LIVE_ENABLED": "true",
    "LIVE_ALLOW_ORDERS": "true",
    "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
    "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
    "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
    "LIVE_LEVERAGE": "20",
}


# ======================================================================
# 1. Default env still blocks
# ======================================================================


class TestDefaultEnvBlocked:
    """With only EXCHANGE=binance, creation MUST raise RuntimeError."""

    def test_default_env_raises_runtime_error(self) -> None:
        env = {"EXCHANGE": "binance"}
        config = load_unified_runtime_config(env)
        with pytest.raises(RuntimeError) as exc_info:
            create_binance_runtime_adapters(config, env)
        msg = str(exc_info.value)
        assert "blocking_reasons=" in msg

    def test_empty_env_raises_runtime_error(self) -> None:
        """Empty env → exchange_is_not_binance in preflight."""
        config = load_unified_runtime_config({})
        with pytest.raises(RuntimeError) as exc_info:
            create_binance_runtime_adapters(config, {})
        msg = str(exc_info.value)
        assert "blocking_reasons=" in msg


# ======================================================================
# 2. SIGNAL_ONLY=true still blocks
# ======================================================================


class TestSignalOnlyBlocked:
    """SIGNAL_ONLY=true MUST block creation even with all other gates set."""

    def test_signal_only_true_blocked(self) -> None:
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "true",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        with pytest.raises(RuntimeError) as exc_info:
            create_binance_runtime_adapters(config, env)
        msg = str(exc_info.value)
        assert "binance_signal_only_enabled" in msg

    def test_signal_only_alias_blocked(self) -> None:
        env = {
            "EXCHANGE": "binance",
            "BINANCE_SIGNAL_ONLY": "true",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        with pytest.raises(RuntimeError) as exc_info:
            create_binance_runtime_adapters(config, env)
        msg = str(exc_info.value)
        assert "binance_signal_only_enabled" in msg


# ======================================================================
# 3. All live gates satisfied → returns adapters (no real network)
# ======================================================================


class TestLiveGatesAllSatisfied:
    """When every preflight gate passes, adapters are created correctly."""

    def test_creates_all_three_adapters(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        # market_data_client
        assert adapters.market_data_client is not None
        assert isinstance(adapters.market_data_client, BinanceMarketDataClient)

        # trading_client
        assert adapters.trading_client is not None
        assert isinstance(adapters.trading_client, BinanceTradingClient)

        # trader
        assert adapters.trader is not None

    def test_market_data_client_has_correct_symbol(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        client = adapters.market_data_client
        assert isinstance(client, BinanceMarketDataClient)
        assert client._symbol == "ETHUSDT"

    def test_trading_client_has_correct_symbol(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        client = adapters.trading_client
        assert isinstance(client, BinanceTradingClient)
        assert client._symbol == "ETHUSDT"

    def test_trader_is_bound_to_trading_client(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        trader = adapters.trader
        assert trader.trading_client is adapters.trading_client

    def test_trader_has_live_trading_true(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        trader = adapters.trader
        assert trader.live_trading is True

    def test_trader_symbol_is_binance(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        trader = adapters.trader
        assert trader.symbol == "ETHUSDT"

    def test_does_not_trigger_network(self) -> None:
        """Creating adapters MUST NOT touch the network.

        We verify this by checking that no aiohttp session is created —
        BinancePrivateClient creates its transport lazily on start().
        """
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        # Verify the private client has no transport yet (lazy init)
        client = adapters.trading_client
        assert isinstance(client, BinanceTradingClient)
        private = client._client
        assert private._transport is None

    def test_using_live_env_var_names(self) -> None:
        """LIVE_* primary names (not BINANCE_* aliases) satisfy all gates."""
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_KEY": "primary-key",
            "EXCHANGE_API_SECRET": "primary-secret",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        assert isinstance(adapters.market_data_client, BinanceMarketDataClient)
        assert isinstance(adapters.trading_client, BinanceTradingClient)
        assert adapters.trader is not None

    def test_using_binance_alias_env_var_names(self) -> None:
        """BINANCE_* alias names still work (backward compat)."""
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_KEY": "alias-key",
            "EXCHANGE_API_SECRET": "alias-secret",
            "BINANCE_LIVE_ENABLED": "true",
            "BINANCE_LIVE_ALLOW_ORDERS": "true",
            "BINANCE_LIVE_CONFIRMATION": "I_UNDERSTAND_BINANCE_LIVE_TRADING",
            "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "BINANCE_LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        assert isinstance(adapters.market_data_client, BinanceMarketDataClient)
        assert isinstance(adapters.trading_client, BinanceTradingClient)
        assert adapters.trader is not None


# ======================================================================
# 4. Credentials missing → fail fast
# ======================================================================


class TestCredentialsMissing:
    """When preflight passes but credentials are missing, fail fast."""

    def test_no_api_key_raises(self) -> None:
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_SECRET": "test-secret",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        with pytest.raises(ValueError) as exc_info:
            create_binance_runtime_adapters(config, env)
        msg = str(exc_info.value)
        assert "API key" in msg
        # Must NOT leak secret values
        assert "test-secret" not in msg

    def test_no_api_secret_raises(self) -> None:
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_KEY": "test-key",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        with pytest.raises(ValueError) as exc_info:
            create_binance_runtime_adapters(config, env)
        msg = str(exc_info.value)
        assert "API secret" in msg
        # Must NOT leak secret values
        assert "test-key" not in msg

    def test_empty_api_key_in_config_raises(self) -> None:
        """When config.api_key is empty but env vars pass preflight."""
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        # Override to remove credentials
        env.pop("EXCHANGE_API_KEY", None)
        env.pop("EXCHANGE_API_SECRET", None)
        config = load_unified_runtime_config(env)
        with pytest.raises(ValueError):
            create_binance_runtime_adapters(config, env)

    def test_credential_error_message_does_not_leak_secret(self) -> None:
        """The error message must never contain actual key/secret values."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        config = load_unified_runtime_config(env)
        with pytest.raises(ValueError) as exc_info:
            create_binance_runtime_adapters(config, env)
        msg = str(exc_info.value)
        # The resolve_binance_credentials function returns an error that
        # mentions env var names, not actual secret values.
        assert "EXCHANGE_API_KEY" in msg or "BINANCE_API_KEY" in msg
        # Should not contain confirmation phrase as a secret leak
        assert "I_UNDERSTAND_EXCHANGE_LIVE_TRADING" not in msg


# ======================================================================
# 5. Runtime factory generic path dispatches to Binance
# ======================================================================


class TestRuntimeFactoryDispatchesBinance:
    """The generic runtime factory correctly dispatches to Binance adapters."""

    def test_create_runtime_bundle_binance(self) -> None:
        from src.live.runtime_factory import create_runtime_bundle

        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        bundle = create_runtime_bundle(env)

        assert isinstance(bundle.market_data_client, BinanceMarketDataClient)
        assert isinstance(bundle.trading_client, BinanceTradingClient)
        assert bundle.trader is not None

    def test_create_runtime_bundle_returns_live_runtime_bundle(self) -> None:
        from src.live.runtime_bundle import LiveRuntimeBundle
        from src.live.runtime_factory import create_runtime_bundle

        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        bundle = create_runtime_bundle(env)

        assert isinstance(bundle, LiveRuntimeBundle)
        assert bundle.runtime_config is not None
        assert bundle.runtime_config.exchange == ExchangeName.BINANCE


# ======================================================================
# 6. OKX runtime unaffected
# ======================================================================


class TestOkxRuntimeUnaffected:
    """OKX runtime path is not affected by Binance wiring."""

    def test_okx_runtime_adapter_still_works(self) -> None:
        from src.exchanges.okx.runtime_adapter import create_okx_runtime_adapters

        env = {
            "EXCHANGE": "okx",
            "EXCHANGE_API_KEY": "okx-test-key",
            "EXCHANGE_API_SECRET": "okx-test-secret",
            "EXCHANGE_API_PASSPHRASE": "okx-test-passphrase",
        }
        config = load_unified_runtime_config(env)
        adapters = create_okx_runtime_adapters(config, env)

        from src.data_feed.okx_market_data_client import OkxMarketDataClient
        from src.execution.okx_trading_client import OkxTradingClient

        assert isinstance(adapters.market_data_client, OkxMarketDataClient)
        assert isinstance(adapters.trading_client, OkxTradingClient)
        assert adapters.trader is not None

    def test_okx_path_does_not_import_binance(self) -> None:
        """Verify OKX adapter module does not import Binance concrete classes."""
        okx_adapter_path = Path("src/exchanges/okx/runtime_adapter.py")
        text = okx_adapter_path.read_text(encoding="utf-8")

        forbidden = [
            "BinanceMarketDataClient",
            "BinanceTradingClient",
            "BinancePrivateClient",
            "from src.data_feed.binance",
            "from src.exchanges.binance",
        ]
        for token in forbidden:
            assert token not in text, (
                f"OKX runtime adapter must not import '{token}'"
            )


# ======================================================================
# 7. Boundary scan — forbidden paths must NOT import Binance concrete classes
# ======================================================================


class TestBoundaryScanNoBinanceConcreteImports:
    """Forbidden paths must NOT import Binance concrete client classes."""

    FORBIDDEN_CONCRETE_CLASSES: tuple[str, ...] = (
        "BinanceMarketDataClient",
        "BinanceTradingClient",
        "BinancePrivateClient",
    )

    FORBIDDEN_IMPORT_PATHS: tuple[str, ...] = (
        "src.data_feed.binance.market_data_client",
        "src.exchanges.binance.trading_client",
        "src.exchanges.binance.private_client",
    )

    ALLOWED_DIRS: tuple[str, ...] = (
        "src/exchanges/binance/",
        "tests/exchanges/binance/",
        "tests/data_feed/binance/",
    )

    FORBIDDEN_DIRS: tuple[str, ...] = (
        "src/live/",
        "src/strategies/",
        "src/monitors/",
    )

    FORBIDDEN_FILES: tuple[str, ...] = (
        "src/execution/trader.py",
        "scripts/run_boll_cvd_live.py",
    )

    def _forbidden_source_texts(self):
        """Yield (rel_path, text) for all forbidden files."""
        for dir_prefix in self.FORBIDDEN_DIRS:
            dir_path = Path(dir_prefix)
            if dir_path.is_dir():
                for py_file in sorted(dir_path.rglob("*.py")):
                    yield py_file.as_posix(), py_file.read_text(encoding="utf-8")

        for file_path_str in self.FORBIDDEN_FILES:
            file_path = Path(file_path_str)
            if file_path.is_file():
                yield file_path.as_posix(), file_path.read_text(encoding="utf-8")

    def test_no_binance_concrete_classes_in_forbidden_dirs(self) -> None:
        violations = []
        for rel_path, text in self._forbidden_source_texts():
            for cls_name in self.FORBIDDEN_CONCRETE_CLASSES:
                if cls_name in text:
                    for i, line in enumerate(text.split("\n"), 1):
                        if cls_name in line and not line.strip().startswith("#"):
                            violations.append(f"{rel_path}:{i}: {line.strip()}")
        assert not violations, (
            "Binance concrete classes must not appear in forbidden dirs/files:\n"
            + "\n".join(violations)
        )

    def test_no_binance_import_paths_in_forbidden_dirs(self) -> None:
        violations = []
        for rel_path, text in self._forbidden_source_texts():
            for import_path in self.FORBIDDEN_IMPORT_PATHS:
                if import_path in text:
                    for i, line in enumerate(text.split("\n"), 1):
                        if import_path in line and not line.strip().startswith("#"):
                            violations.append(f"{rel_path}:{i}: {line.strip()}")
        assert not violations, (
            "Binance import paths must not appear in forbidden dirs/files:\n"
            + "\n".join(violations)
        )

    def test_allowed_dirs_may_import_binance(self) -> None:
        """Sanity check: allowed dirs DO contain Binance concrete classes."""
        found = False
        for dir_prefix in self.ALLOWED_DIRS:
            dir_path = Path(dir_prefix)
            if dir_path.is_dir():
                for py_file in sorted(dir_path.rglob("*.py")):
                    text = py_file.read_text(encoding="utf-8")
                    if any(cls in text for cls in self.FORBIDDEN_CONCRETE_CLASSES):
                        found = True
                        break
            if found:
                break
        assert found, (
            "Expected at least one file in allowed dirs to reference Binance "
            "concrete classes — this verifies the boundary scan is not vacuous."
        )

    def test_runtime_factory_py_no_binance_concrete_classes(self) -> None:
        """src/live/runtime_factory.py must NOT reference Binance concrete classes."""
        text = Path("src/live/runtime_factory.py").read_text(encoding="utf-8")
        for cls_name in self.FORBIDDEN_CONCRETE_CLASSES:
            assert cls_name not in text, (
                f"runtime_factory.py must not reference {cls_name}"
            )

    def test_trader_py_no_binance_concrete_imports(self) -> None:
        """src/execution/trader.py must NOT import Binance concrete classes."""
        text = Path("src/execution/trader.py").read_text(encoding="utf-8")
        for import_path in self.FORBIDDEN_IMPORT_PATHS:
            assert import_path not in text, (
                f"trader.py must not import {import_path}"
            )
        # Check for direct class name imports
        for cls_name in self.FORBIDDEN_CONCRETE_CLASSES:
            # Only flag if it appears as an import
            if f"import {cls_name}" in text or f"from src" in text and cls_name in text:
                for i, line in enumerate(text.split("\n"), 1):
                    if cls_name in line and ("import" in line or "from src" in line):
                        if not line.strip().startswith("#"):
                            assert False, (
                                f"trader.py:{i}: must not import Binance: {line.strip()}"
                            )

    def test_run_boll_cvd_live_no_binance_concrete_classes(self) -> None:
        """scripts/run_boll_cvd_live.py must NOT import Binance concrete classes."""
        script_path = Path("scripts/run_boll_cvd_live.py")
        if not script_path.is_file():
            pytest.skip("scripts/run_boll_cvd_live.py not found")
        text = script_path.read_text(encoding="utf-8")
        for cls_name in self.FORBIDDEN_CONCRETE_CLASSES:
            assert cls_name not in text, (
                f"run_boll_cvd_live.py must not reference {cls_name}"
            )


# ======================================================================
# 8. No broker semantic executor created
# ======================================================================


class TestNoBrokerSemanticExecutor:
    """Binance runtime adapter does NOT create or bind a broker semantic executor."""

    def test_trader_broker_semantic_executor_is_none(self) -> None:
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        trader = adapters.trader
        assert trader._broker_semantic_executor is None

    def test_trader_broker_semantic_reads_disabled_by_default(self) -> None:
        """Without BROKER_SEMANTIC_READS_ENABLED, broker path is disabled."""
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        trader = adapters.trader
        assert trader._broker_semantic_reads_enabled() is False

    def test_broker_exchange_name_is_still_okx(self) -> None:
        """Known residual: broker_exchange_name still returns 'okx'.

        This is a known limitation — the Trader's broker_exchange_name is not
        yet parameterised by exchange.  Binance does not use broker paths,
        so this is harmless for the current wiring scope.
        """
        env = dict(LIVE_PREFLIGHT_PASS_ENV)
        config = load_unified_runtime_config(env)
        adapters = create_binance_runtime_adapters(config, env)

        trader = adapters.trader
        # Known: returns "okx" even when bound to Binance
        assert trader.broker_exchange_name == "okx"
