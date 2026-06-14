"""Source boundary tests for the modified TP/SL manager files.

Ref: 20C-CLEAN-PORTS-05 / 20C-CLEAN-PORTS-07

These tests scan ONLY the modified files:
- src/execution/tp_sl_execution_manager.py
- src/execution/tp_sl_core_tp_manager.py
- src/execution/tp_sl_protective_stop_manager.py
- src/execution/tp_sl_market_exit_manager.py
- src/execution/tp_sl_near_tp_manager.py

They verify that no forbidden patterns leak into the modified code.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Files under test
# ---------------------------------------------------------------------------

_MODIFIED_FILES = [
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
]


def _read_modified_files() -> str:
    """Return the concatenated text of all three modified files."""
    parts: list[str] = []
    for path in _MODIFIED_FILES:
        parts.append(Path(path).read_text(encoding="utf-8"))
    return "\n".join(parts)


# ===================================================================
# 1. No direct REST endpoint strings in replaced paths
# ===================================================================


class TestNoDirectRestEndpoints:
    """The replaced code paths must not contain direct REST endpoint URLs."""

    FORBIDDEN_IN_REPLACED = [
        "/api/v5/trade/order",
        "/api/v5/trade/order-algo",
        "/api/v5/trade/cancel-order",
        "/api/v5/trade/cancel-algos",
    ]

    def test_tp_sl_core_tp_no_direct_order(self) -> None:
        """CoreTakeProfitManager._place_reduce_only_take_profit_orders
        non-semantic branch must not call /api/v5/trade/order."""
        text = Path(
            "src/execution/tp_sl_core_tp_manager.py"
        ).read_text(encoding="utf-8")

        # Scan only the _place_reduce_only_take_profit_orders method
        lines = text.splitlines()
        in_method = False
        for line in lines:
            if "def _place_reduce_only_take_profit_orders" in line:
                in_method = True
            elif in_method and line.startswith("    def "):
                in_method = False
            if in_method and '"/api/v5/trade/order"' in line:
                pytest.fail(
                    "_place_reduce_only_take_profit_orders contains "
                    "direct /api/v5/trade/order"
                )

    def test_protective_stop_placement_methods_no_direct_order_algo(self) -> None:
        """All ProtectiveStopManager placement methods must not call
        /api/v5/trade/order-algo or extract_algo_id."""
        text = Path(
            "src/execution/tp_sl_protective_stop_manager.py"
        ).read_text(encoding="utf-8")

        # Placement method names (not cancel methods)
        placement_methods = [
            "place_near_tp_protective_stop_with_retries",
            "place_middle_runner_protective_stop_with_retries",
            "place_middle_bucket_fast_protective_stop_with_retries",
            "place_trend_runner_protective_stop_with_retries",
            "place_three_stage_post_tp1_protective_stop_with_retries",
            "_place_primary_protective_stop_semantic",
        ]

        lines = text.splitlines()
        for method_name in placement_methods:
            in_method = False
            for line in lines:
                if f"def {method_name}" in line:
                    in_method = True
                    continue
                if in_method and line.startswith("    def "):
                    in_method = False
                    continue
                if in_method and '"/api/v5/trade/order-algo"' in line:
                    pytest.fail(
                        f"{method_name} must not contain direct "
                        "/api/v5/trade/order-algo"
                    )
                if in_method and "extract_algo_id(" in line:
                    pytest.fail(
                        f"{method_name} must not call extract_algo_id"
                    )

    def test_execution_manager_cancel_no_direct_cancel_order(self) -> None:
        """cancel_existing_reduce_only_orders non-semantic branch must not
        call /api/v5/trade/cancel-order."""
        text = Path(
            "src/execution/tp_sl_execution_manager.py"
        ).read_text(encoding="utf-8")

        lines = text.splitlines()
        in_method = False
        for line in lines:
            if "def cancel_existing_reduce_only_orders" in line:
                in_method = True
            elif in_method and line.startswith("    def "):
                in_method = False
            if in_method and '"/api/v5/trade/cancel-order"' in line:
                pytest.fail(
                    "cancel_existing_reduce_only_orders contains "
                    "direct /api/v5/trade/cancel-order"
                )


# ===================================================================
# 2. No Binance or forbidden abstractions
# ===================================================================


class TestNoForbiddenImportsOrReferences:
    """The modified files must not reference Binance or forbidden abstractions."""

    FORBIDDEN_TOKENS = [
        "Binance",
        "binance",
        "ExchangeRuntimeBundle",
        "ThreeStageAdapter",
        "MiddleRunnerAdapter",
        "SidecarAdapter",
        "NearTpAdapter",
        "BrokerSemanticExecutor",
        "BinanceTradingClient",
        "BinanceMarketDataClient",
    ]

    @pytest.mark.parametrize("file_path", _MODIFIED_FILES)
    def test_no_forbidden_tokens(self, file_path: str) -> None:
        text = Path(file_path).read_text(encoding="utf-8")
        for token in self.FORBIDDEN_TOKENS:
            # Allow "BrokerSemanticExecutor" only when used as an
            # attribute access on the trader (t.broker_semantic_executor)
            # which existed before this change.
            if token == "BrokerSemanticExecutor" and "t.broker_semantic_executor" in text:
                # Only check for class-level import, not attribute access
                if f"import {token}" in text or f"from {token}" in text:
                    pytest.fail(f"{file_path} imports {token}")
                continue
            assert token not in text, (
                f"{file_path} must not reference {token}"
            )


# ===================================================================
# 3. No Trader() / OkxPrivateClient() instantiation
# ===================================================================


class TestNoNewClientInstantiation:
    """The modified files must not create new Trader or OkxPrivateClient."""

    def test_no_trader_instantiation(self) -> None:
        text = _read_modified_files()
        # "Trader()" should not appear (constructor call)
        assert "Trader()" not in text, (
            "must not instantiate Trader() in modified files"
        )

    def test_no_okx_private_client_instantiation(self) -> None:
        text = _read_modified_files()
        assert "OkxPrivateClient()" not in text, (
            "must not instantiate OkxPrivateClient() in modified files"
        )

    def test_no_new_trader_construction(self) -> None:
        """Trader.__new__ is fine in tests, not in production source."""
        for file_path in _MODIFIED_FILES:
            text = Path(file_path).read_text(encoding="utf-8")
            # Only flag actual Trader() constructor calls
            lines = text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "= Trader(" in stripped:
                    pytest.fail(
                        f"{file_path}:{i} creates new Trader()"
                    )


# ===================================================================
# 4. No env reads in the new code paths
# ===================================================================


class TestNoNewEnvReads:
    """The modified files must not read env vars for the new port wiring."""

    def test_no_new_os_getenv_for_port_wiring(self) -> None:
        """The trading_client creation in __init__ must not read env."""
        for file_path in _MODIFIED_FILES:
            text = Path(file_path).read_text(encoding="utf-8")
            lines = text.splitlines()
            in_init = False
            for line in lines:
                if "def __init__" in line and "trading_client" in line:
                    in_init = True
                    continue
                if in_init and line.strip() and not line.startswith("        "):
                    in_init = False
                if in_init and "os.getenv" in line:
                    pytest.fail(
                        f"{file_path} __init__ must not read env vars "
                        "for trading_client creation"
                    )

    def test_no_load_dotenv_in_modified_files(self) -> None:
        text = _read_modified_files()
        assert "load_dotenv" not in text, (
            "must not call load_dotenv in modified files"
        )


# ===================================================================
# 5. trading_client attribute exists on managers
# ===================================================================


class TestTradingClientAttribute:
    """Verify the trading_client attribute is properly wired."""

    def test_execution_manager_has_trading_client(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("0")
        trader.account_equity_usdt = 0.0
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True

        from unittest import mock

        with mock.patch.object(trader, "request", return_value={"data": []}):
            manager = TpSlExecutionManager(trader)

        from src.execution.trading_client_port import TradingClientPort
        from src.execution.okx_trading_client import OkxTradingClient

        assert hasattr(manager, "trading_client")
        assert isinstance(manager.trading_client, OkxTradingClient)

        # Sub-managers share the same trading_client instance
        assert manager.core_tp.trading_client is manager.trading_client
        assert manager.protective_stops.trading_client is manager.trading_client
        assert manager.market_exit.trading_client is manager.trading_client
        assert manager.near_tp.trading_client is manager.trading_client

    def test_core_tp_manager_has_trading_client(self) -> None:
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager
        from src.execution.trader import Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")

        from src.execution.okx_trading_client import OkxTradingClient
        tc = OkxTradingClient(trader)

        manager = CoreTakeProfitManager(trader, protective_stops=None,
                                        trading_client=tc)
        assert manager.trading_client is tc

    def test_protective_stop_manager_has_trading_client(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager
        from src.execution.trader import Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")

        from src.execution.okx_trading_client import OkxTradingClient
        tc = OkxTradingClient(trader)

        manager = ProtectiveStopManager(trader, trading_client=tc)
        assert manager.trading_client is tc


# ===================================================================
# 6. Method-level boundary checks for protective stop manager
# ===================================================================


class TestProtectiveStopManagerMethodBoundaries:
    """Granular per-method boundary verification for
    ProtectiveStopManager."""

    PLACEMENT_METHODS = [
        "place_near_tp_protective_stop_with_retries",
        "place_middle_runner_protective_stop_with_retries",
        "place_middle_bucket_fast_protective_stop_with_retries",
        "place_trend_runner_protective_stop_with_retries",
        "place_three_stage_post_tp1_protective_stop_with_retries",
        "_place_primary_protective_stop_semantic",
    ]

    CANCEL_METHODS = [
        "_cancel_unverified_near_tp_algo",
    ]

    FORBIDDEN_IN_PLACEMENT = [
        '"/api/v5/trade/order-algo"',
        "extract_algo_id(",
        '"/api/v5/trade/cancel-algos"',
    ]

    @pytest.mark.parametrize("method_name", PLACEMENT_METHODS)
    def test_placement_methods_no_forbidden_patterns(self, method_name: str) -> None:
        text = Path(
            "src/execution/tp_sl_protective_stop_manager.py"
        ).read_text(encoding="utf-8")

        lines = text.splitlines()
        in_method = False
        for i, line in enumerate(lines, 1):
            if f"def {method_name}" in line:
                in_method = True
                continue
            if in_method and line.startswith("    def "):
                in_method = False
                continue
            if in_method:
                for forbidden in self.FORBIDDEN_IN_PLACEMENT:
                    if forbidden in line:
                        # Skip comments
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        pytest.fail(
                            f"{method_name}:{i} must not contain {forbidden}"
                        )

    def test_cancel_method_allows_cancel_algos(self) -> None:
        """_cancel_unverified_near_tp_algo delegates to trader, which is
        allowed to use cancel-algos. This test verifies the method itself
        does not contain placement endpoints."""
        text = Path(
            "src/execution/tp_sl_protective_stop_manager.py"
        ).read_text(encoding="utf-8")

        lines = text.splitlines()
        in_method = False
        for i, line in enumerate(lines, 1):
            if "def _cancel_unverified_near_tp_algo" in line:
                in_method = True
                continue
            if in_method and line.startswith("    def "):
                in_method = False
                continue
            if in_method:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if '"/api/v5/trade/order-algo"' in line:
                    pytest.fail(
                        f"_cancel_unverified_near_tp_algo:{i} must not "
                        "contain placement endpoint order-algo"
                    )
                if "extract_algo_id(" in line:
                    pytest.fail(
                        f"_cancel_unverified_near_tp_algo:{i} must not "
                        "call extract_algo_id"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
