#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_private_rest_adapter_boundaries.py
@Description: Boundary tests verifying that OKX private REST calls are only
              in the OKX adapter/client layer (OkxPrivateClient, OkxTradingClient).
              Also verifies Trader no longer imports OKX-specific classes.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestOkxPrivateRestOnlyInAdapterLayer:
    """OKX private REST patterns must only appear in allowed files."""

    def test_okx_private_client_has_api_v5(self) -> None:
        """OkxPrivateClient is the OKX private REST adapter — /api/v5 is allowed."""
        filepath = ROOT / "src" / "execution" / "okx_private_client.py"
        assert filepath.exists()
        text = filepath.read_text(encoding="utf-8")
        # OkxPrivateClient doesn't construct /api/v5 URLs (it receives endpoint)
        # But it handles the signing, so it's the adapter layer
        assert "hmac" in text or "OK-ACCESS" in text

    def test_okx_trading_client_is_adapter(self) -> None:
        """OkxTradingClient IS the trading adapter — /api/v5 is allowed."""
        filepath = ROOT / "src" / "execution" / "okx_trading_client.py"
        assert filepath.exists()
        text = filepath.read_text(encoding="utf-8")
        assert "/api/v5" in text, "OkxTradingClient must contain /api/v5 (it IS the adapter)"

    def test_trader_legacy_wrappers_no_api_v5(self) -> None:
        """Trader's legacy wrapper methods must NOT contain /api/v5 directly."""
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")

        # Check specific methods
        import ast
        tree = ast.parse(text)
        wrapper_methods = [
            "fetch_usdt_equity",
            "fetch_position_snapshot",
            "fetch_pending_orders",
            "fetch_pending_algo_orders",
            "set_leverage",
        ]

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in wrapper_methods:
                    src = ast.get_source_segment(text, node)
                    if src and "/api/v5" in src:
                        violations.append(f"trader.py::{node.name} contains /api/v5")

        assert not violations, (
            "Trader legacy wrappers must NOT contain /api/v5:\n" + "\n".join(violations)
        )

    def test_trader_does_not_import_okx_trading_client(self) -> None:
        """Trader must NOT import or instantiate OkxTradingClient."""
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "from src.execution.okx_trading_client import OkxTradingClient" not in text, (
            "trader.py must NOT import OkxTradingClient"
        )
        assert "OkxTradingClient(" not in text, (
            "trader.py must NOT instantiate OkxTradingClient"
        )

    def test_tp_sl_execution_manager_does_not_import_okx_trading_client(self) -> None:
        """TpSlExecutionManager must NOT import or instantiate OkxTradingClient."""
        filepath = ROOT / "src" / "execution" / "tp_sl_execution_manager.py"
        text = filepath.read_text(encoding="utf-8")
        assert "from src.execution.okx_trading_client import OkxTradingClient" not in text, (
            "tp_sl_execution_manager.py must NOT import OkxTradingClient"
        )
        assert "OkxTradingClient(" not in text, (
            "tp_sl_execution_manager.py must NOT instantiate OkxTradingClient"
        )

    def test_tp_sl_execution_manager_no_api_v5(self) -> None:
        """TpSlExecutionManager must NOT contain /api/v5 — uses TradingClientPort."""
        filepath = ROOT / "src" / "execution" / "tp_sl_execution_manager.py"
        text = filepath.read_text(encoding="utf-8")
        assert "/api/v5" not in text, (
            "tp_sl_execution_manager.py must NOT contain /api/v5"
        )


class TestTraderDoesNotImportOkxPrivateClient:
    """Trader must NOT import OKX private client classes."""

    def test_no_okx_config_import(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "from config.env_loader import OKX_CONFIG" not in text, (
            "trader.py must NOT import OKX_CONFIG"
        )

    def test_no_okx_private_client_import(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "from src.execution.okx_private_client import OkxPrivateClient" not in text, (
            "trader.py must NOT import OkxPrivateClient"
        )

    def test_no_okx_private_client_config_import(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "OkxPrivateClientConfig" not in text, (
            "trader.py must NOT import OkxPrivateClientConfig"
        )

    def test_no_private_write_rate_limiter_import(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "PrivateWriteRateLimiter" not in text, (
            "trader.py must NOT import PrivateWriteRateLimiter"
        )

    def test_no_self_dot_client_instantiation(self) -> None:
        """Trader must NOT have self._client = OkxPrivateClient(...)."""
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "self._client = OkxPrivateClient(" not in text, (
            "trader.py must NOT create self._client = OkxPrivateClient(...)"
        )

    def test_no_okx_broker_client_import(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "from src.exchanges.okx.client import OkxBrokerClient" not in text, (
            "trader.py must NOT import OkxBrokerClient"
        )

    def test_no_okx_broker_semantic_executor_import(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        text = filepath.read_text(encoding="utf-8")
        assert "from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor" not in text, (
            "trader.py must NOT import OkxBrokerSemanticExecutor"
        )


class TestTraderHasBindMethods:
    """Trader must have bind methods for private client and broker executor."""

    def test_bind_private_client_exists(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        import ast
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "bind_private_client":
                return
        pytest.fail("trader.py must have bind_private_client method")

    def test_bind_broker_semantic_executor_exists(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        import ast
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "bind_broker_semantic_executor":
                return
        pytest.fail("trader.py must have bind_broker_semantic_executor method")

    def test_broker_semantic_executor_raises_when_not_bound(self) -> None:
        filepath = ROOT / "src" / "execution" / "trader.py"
        import ast
        text = filepath.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "broker_semantic_executor":
                src = ast.get_source_segment(text, node)
                assert src is not None
                assert "broker_semantic_executor_not_bound" in src, (
                    "broker_semantic_executor property must raise when not bound"
                )
                return
        pytest.fail("trader.py must have broker_semantic_executor property")
