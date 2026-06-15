#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_runtime_factory_exchange_boundaries.py
@Description: Boundary tests — runtime_factory.py must NOT know about any
              specific exchange adapter.

The live runtime factory is a pure bundle assembly layer.  It must only depend
on the generic exchange adapter factory and the runtime config.  Any exchange-
specific adapter path, class name, or env variable is a boundary violation.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

RUNTIME_FACTORY_PATH = ROOT / "src" / "live" / "runtime_factory.py"
RUNTIME_CONFIG_PATH = ROOT / "src" / "exchanges" / "runtime_config.py"
GENERIC_FACTORY_PATH = ROOT / "src" / "exchanges" / "runtime_adapter_factory.py"

RUNTIME_FACTORY_SOURCE = RUNTIME_FACTORY_PATH.read_text(encoding="utf-8")
RUNTIME_CONFIG_SOURCE = RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
GENERIC_FACTORY_SOURCE = GENERIC_FACTORY_PATH.read_text(encoding="utf-8")

# ── Forbidden in runtime_factory.py ──────────────────────────────────────

FORBIDDEN_EXCHANGE_ADAPTER_PATHS = [
    "src.exchanges.okx",
    "src.exchanges.binance",
    "src.execution.okx_",
    "src.data_feed.okx_",
]

FORBIDDEN_EXCHANGE_CLASSES = [
    "OkxPrivateClient",
    "OkxTradingClient",
    "OkxMarketDataClient",
    "OkxBrokerClient",
    "OkxBrokerSemanticExecutor",
    "BinanceBrokerClient",
    "BinanceTradingClient",
    "BinanceMarketDataClient",
    "AiohttpBinanceTransport",
    "OKX_CONFIG",
]

FORBIDDEN_IMPORT_PATHS = [
    "src.live.binance_live_preflight",
    "from src.exchanges.okx",
    "from src.exchanges.binance",
]

ALLOWED_IMPORTS = [
    "from src.exchanges.runtime_adapter_factory import",
    "from src.exchanges.runtime_config import",
    "from src.live.runtime_bundle import",
]

# ── Forbidden in runtime_config.py ───────────────────────────────────────

FORBIDDEN_CREDENTIAL_VARS = [
    "OKX_API_KEY",
    "OKX_SECRET_KEY",
    "OKX_API_SECRET",
    "OKX_PASSPHASE",
    "OKX_PASSPHRASE",
]


class TestRuntimeFactoryNoExchangeAdapterPaths:
    """runtime_factory.py must not import any exchange-specific adapter path."""

    def test_no_exchange_adapter_paths(self) -> None:
        violations = []
        for path_prefix in FORBIDDEN_EXCHANGE_ADAPTER_PATHS:
            if path_prefix in RUNTIME_FACTORY_SOURCE:
                for i, line in enumerate(RUNTIME_FACTORY_SOURCE.split("\n"), 1):
                    if path_prefix in line and not line.strip().startswith("#"):
                        violations.append(
                            f"runtime_factory.py:{i}: {line.strip()}"
                        )
        assert not violations, (
            "runtime_factory.py must NOT import exchange adapter paths:\n"
            + "\n".join(violations)
        )

    def test_no_forbidden_import_paths(self) -> None:
        violations = []
        for path_prefix in FORBIDDEN_IMPORT_PATHS:
            if path_prefix in RUNTIME_FACTORY_SOURCE:
                for i, line in enumerate(RUNTIME_FACTORY_SOURCE.split("\n"), 1):
                    if path_prefix in line and "import" in line and not line.strip().startswith("#"):
                        violations.append(
                            f"runtime_factory.py:{i}: {line.strip()}"
                        )
        assert not violations, (
            "runtime_factory.py must NOT import from forbidden paths:\n"
            + "\n".join(violations)
        )

    def test_no_exchange_concrete_classes(self) -> None:
        violations = []
        for symbol in FORBIDDEN_EXCHANGE_CLASSES:
            if symbol in RUNTIME_FACTORY_SOURCE:
                for i, line in enumerate(RUNTIME_FACTORY_SOURCE.split("\n"), 1):
                    if symbol in line and not line.strip().startswith("#"):
                        violations.append(
                            f"runtime_factory.py:{i}: {line.strip()}"
                        )
        assert not violations, (
            "runtime_factory.py must NOT reference exchange concrete classes:\n"
            + "\n".join(violations)
        )

    def test_uses_allowed_imports(self) -> None:
        missing = []
        for allowed in ALLOWED_IMPORTS:
            if allowed not in RUNTIME_FACTORY_SOURCE:
                missing.append(allowed)
        assert not missing, (
            "runtime_factory.py must import these generic modules:\n"
            + "\n".join(missing)
        )


class TestRuntimeConfigNoLegacyCredentials:
    """runtime_config.py must not read OKX legacy credential env vars."""

    def test_no_okx_legacy_credential_reading(self) -> None:
        violations = []
        for var in FORBIDDEN_CREDENTIAL_VARS:
            # Only flag if it appears in a values.get() call (not in a comment)
            if f'values.get("{var}"' in RUNTIME_CONFIG_SOURCE or \
               f"values.get('{var}'" in RUNTIME_CONFIG_SOURCE:
                for i, line in enumerate(RUNTIME_CONFIG_SOURCE.split("\n"), 1):
                    if f".get(\"{var}\"" in line or f".get('{var}'" in line:
                        violations.append(f"runtime_config.py:{i}: {line.strip()}")
        assert not violations, (
            "runtime_config.py must NOT read OKX legacy credential env vars:\n"
            + "\n".join(violations)
        )

    def test_docstring_does_not_say_not_wired(self) -> None:
        """The docstring must no longer claim this module is not wired into live paths."""
        assert "NOT wired into live trading paths" not in RUNTIME_CONFIG_SOURCE, (
            "runtime_config.py docstring must no longer claim it is NOT wired into live trading paths"
        )


class TestGenericAdapterFactoryBoundaries:
    """The generic adapter factory dispatches to exchange adapter modules only."""

    def test_factory_dispatches_to_okx_adapter(self) -> None:
        assert "src.exchanges.okx.runtime_adapter" in GENERIC_FACTORY_SOURCE, (
            "generic factory must dispatch to OKX adapter"
        )

    def test_factory_dispatches_to_binance_adapter(self) -> None:
        assert "src.exchanges.binance.runtime_adapter" in GENERIC_FACTORY_SOURCE, (
            "generic factory must dispatch to Binance adapter"
        )

    def test_factory_does_not_import_okx_trading_client(self) -> None:
        assert "from src.execution.okx_trading_client" not in GENERIC_FACTORY_SOURCE, (
            "generic factory must NOT import OkxTradingClient"
        )

    def test_factory_does_not_import_okx_private_client(self) -> None:
        assert "from src.execution.okx_private_client" not in GENERIC_FACTORY_SOURCE, (
            "generic factory must NOT import OkxPrivateClient"
        )

    def test_factory_does_not_import_okx_market_data_client(self) -> None:
        assert "from src.data_feed.okx_market_data_client" not in GENERIC_FACTORY_SOURCE, (
            "generic factory must NOT import OkxMarketDataClient"
        )
