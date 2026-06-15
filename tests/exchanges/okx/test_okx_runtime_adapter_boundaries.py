#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_okx_runtime_adapter_boundaries.py
@Description: Boundary tests for the OKX runtime adapter freeze.

These tests ensure that:
1. runtime_factory.py does NOT import or instantiate any Okx* class.
2. runtime_factory.py does NOT import OKX_CONFIG from config.env_loader.
3. The OKX runtime_adapter is the ONLY place that creates OKX concrete instances.
4. Credential validation lives in the OKX adapter, not in business layers.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

RUNTIME_FACTORY_PATH = ROOT / "src" / "live" / "runtime_factory.py"
OKX_ADAPTER_PATH = ROOT / "src" / "exchanges" / "okx" / "runtime_adapter.py"

RUNTIME_FACTORY_SOURCE = RUNTIME_FACTORY_PATH.read_text(encoding="utf-8")
OKX_ADAPTER_SOURCE = OKX_ADAPTER_PATH.read_text(encoding="utf-8")

# Okx* concrete classes that must NOT appear in runtime_factory.py
FORBIDDEN_OKX_IMPORTS = [
    "OkxPrivateClient",
    "OkxPrivateClientConfig",
    "OkxTradingClient",
    "OkxMarketDataClient",
    "OkxMarketDataClientConfig",
    "OkxBrokerClient",
    "OkxBrokerSemanticExecutor",
    "PrivateWriteRateLimiter",
]

# Exchange-specific import paths that must NOT appear in runtime_factory.py
FORBIDDEN_EXCHANGE_PATHS = [
    "src.exchanges.okx",
    "src.exchanges.binance",
    "src.execution.okx_",
    "src.data_feed.okx_",
    "src.live.binance_live_preflight",
]

# Exchange concrete classes from any exchange that must NOT appear
FORBIDDEN_EXCHANGE_CLASSES = [
    "BinanceBrokerClient",
    "BinanceTradingClient",
    "BinanceMarketDataClient",
    "AiohttpBinanceTransport",
    "OKX_CONFIG",
]


class TestRuntimeFactoryDoesNotImportOkxClasses:
    """runtime_factory.py is a generic dispatch layer — no exchange concrete classes."""

    def test_no_okx_class_imports(self) -> None:
        violations = []
        for symbol in FORBIDDEN_OKX_IMPORTS:
            if symbol in RUNTIME_FACTORY_SOURCE:
                for i, line in enumerate(RUNTIME_FACTORY_SOURCE.split("\n"), 1):
                    if symbol in line and not line.strip().startswith("#"):
                        violations.append(f"src/live/runtime_factory.py:{i}: {line.strip()}")
        assert not violations, (
            f"runtime_factory.py must NOT import Okx* classes:\n" + "\n".join(violations)
        )

    def test_no_exchange_adapter_paths(self) -> None:
        """runtime_factory.py must NOT import any exchange-specific adapter path."""
        violations = []
        for path_prefix in FORBIDDEN_EXCHANGE_PATHS:
            if path_prefix in RUNTIME_FACTORY_SOURCE:
                for i, line in enumerate(RUNTIME_FACTORY_SOURCE.split("\n"), 1):
                    if path_prefix in line and "import" in line and not line.strip().startswith("#"):
                        violations.append(f"src/live/runtime_factory.py:{i}: {line.strip()}")
        assert not violations, (
            f"runtime_factory.py must NOT import exchange adapter paths:\n" + "\n".join(violations)
        )

    def test_no_exchange_concrete_classes(self) -> None:
        violations = []
        for symbol in FORBIDDEN_EXCHANGE_CLASSES:
            if symbol in RUNTIME_FACTORY_SOURCE:
                for i, line in enumerate(RUNTIME_FACTORY_SOURCE.split("\n"), 1):
                    if symbol in line and not line.strip().startswith("#"):
                        violations.append(f"src/live/runtime_factory.py:{i}: {line.strip()}")
        assert not violations, (
            f"runtime_factory.py must NOT reference exchange concrete classes:\n" + "\n".join(violations)
        )

    def test_no_okx_config_import(self) -> None:
        assert "from config.env_loader import OKX_CONFIG" not in RUNTIME_FACTORY_SOURCE, (
            "runtime_factory.py must NOT import OKX_CONFIG from config.env_loader"
        )

    def test_runtime_factory_uses_generic_factory(self) -> None:
        """runtime_factory.py must delegate to the generic adapter factory, not OKX directly."""
        assert "from src.exchanges.runtime_adapter_factory import" in RUNTIME_FACTORY_SOURCE, (
            "runtime_factory.py must delegate to src.exchanges.runtime_adapter_factory"
        )


class TestOkxRuntimeAdapterIsCompositionRoot:
    """The OKX runtime_adapter.py is the sole composition root for OKX instances."""

    def test_adapter_file_exists(self) -> None:
        assert OKX_ADAPTER_PATH.exists(), "OKX runtime_adapter.py must exist"

    def test_adapter_creates_okx_private_client(self) -> None:
        assert "OkxPrivateClient(" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must create OkxPrivateClient"
        )

    def test_adapter_creates_okx_trading_client(self) -> None:
        assert "OkxTradingClient(" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must create OkxTradingClient"
        )

    def test_adapter_creates_okx_market_data_client(self) -> None:
        assert "OkxMarketDataClient(" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must create OkxMarketDataClient"
        )

    def test_adapter_creates_okx_broker_client(self) -> None:
        assert "OkxBrokerClient(" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must create OkxBrokerClient"
        )

    def test_adapter_creates_okx_broker_semantic_executor(self) -> None:
        assert "OkxBrokerSemanticExecutor(" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must create OkxBrokerSemanticExecutor"
        )

    def test_credential_validation_in_adapter(self) -> None:
        """Credential validation ValueError must live in the OKX adapter."""
        assert "OKX API config is incomplete" in OKX_ADAPTER_SOURCE, (
            "Credential validation error message must be in OKX runtime_adapter"
        )

    def test_credential_from_config_not_okx_config_global(self) -> None:
        """Adapter must read credentials from ExchangeRuntimeConfig, not OKX_CONFIG global."""
        assert "OKX_CONFIG" not in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must NOT import OKX_CONFIG global"
        )

    def test_uses_resolve_okx_credentials(self) -> None:
        """Adapter resolves credentials via resolve_okx_credentials, not OKX_CONFIG global."""
        assert "resolve_okx_credentials" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must resolve credentials via resolve_okx_credentials"
        )

    def test_returns_exchange_runtime_adapters_not_live_bundle(self) -> None:
        """Adapter must return ExchangeRuntimeAdapters, not LiveRuntimeBundle."""
        assert "ExchangeRuntimeAdapters" in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must return ExchangeRuntimeAdapters"
        )
        assert "LiveRuntimeBundle" not in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must NOT import or return LiveRuntimeBundle"
        )

    def test_uses_config_leverage_not_env_leverage(self) -> None:
        """Adapter must use config.leverage, not env.get('LEVERAGE', ...)."""
        assert 'env.get("LEVERAGE"' not in OKX_ADAPTER_SOURCE, (
            "OKX runtime_adapter must NOT read LEVERAGE from env directly"
        )


class TestOkxAdapterHasCorrectImports:
    """The OKX runtime_adapter properly imports all necessary concrete classes."""

    def test_imports_okx_private_client(self) -> None:
        assert "from src.execution.okx_private_client import" in OKX_ADAPTER_SOURCE

    def test_imports_okx_trading_client(self) -> None:
        assert "from src.execution.okx_trading_client import" in OKX_ADAPTER_SOURCE

    def test_imports_okx_market_data_client(self) -> None:
        assert "from src.data_feed.okx_market_data_client import" in OKX_ADAPTER_SOURCE

    def test_imports_okx_broker_client(self) -> None:
        assert "from src.exchanges.okx.client import" in OKX_ADAPTER_SOURCE

    def test_imports_okx_broker_semantic_executor(self) -> None:
        assert "from src.exchanges.okx.semantic_executor import" in OKX_ADAPTER_SOURCE
