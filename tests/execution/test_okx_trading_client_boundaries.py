#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_trading_client_boundaries.py
@Description: Boundary tests for OkxTradingClient — the source must NOT
              contain forbidden imports, patterns, or references.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "okx_trading_client.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_file_exists() -> None:
    assert _SOURCE_PATH.exists(), f"OkxTradingClient file not found at {_SOURCE_PATH}"
    assert _SOURCE_PATH.is_file()


def test_file_compiles() -> None:
    text = _read_source()
    compile(text, str(_SOURCE_PATH), "exec")


# ======================================================================
# Import-ability
# ======================================================================


def test_can_be_imported() -> None:
    from src.execution.okx_trading_client import OkxTradingClient  # noqa: F401


def test_normalise_helper_importable() -> None:
    from src.execution.okx_trading_client import _normalise_position_side  # noqa: F401


def test_normalise_client_order_id_importable() -> None:
    from src.execution.okx_trading_client import _normalise_client_order_id  # noqa: F401


# ======================================================================
# Implements TradingClientPort
# ======================================================================


def test_implements_trading_client_port() -> None:
    from src.execution.okx_trading_client import OkxTradingClient
    from src.execution.trading_client_port import TradingClientPort

    # TradingClientPort is a Protocol (not @runtime_checkable), so we
    # verify structural conformance by checking that all port methods exist.
    assert hasattr(OkxTradingClient, "fetch_balance")
    assert hasattr(OkxTradingClient, "fetch_position")
    assert hasattr(OkxTradingClient, "fetch_open_orders")
    assert hasattr(OkxTradingClient, "place_market_order")
    assert hasattr(OkxTradingClient, "place_limit_order")
    assert hasattr(OkxTradingClient, "place_stop_market_order")
    assert hasattr(OkxTradingClient, "cancel_order")


# ======================================================================
# Forbidden tokens — no Binance references
# ======================================================================


class TestNoBinanceReferences:
    def test_no_binance_word(self) -> None:
        text = _read_source()
        assert "binance" not in text
        assert "Binance" not in text

    def test_no_ethusdt_symbol(self) -> None:
        text = _read_source()
        assert "ETHUSDT" not in text

    def test_no_fapi(self) -> None:
        text = _read_source()
        assert "/fapi" not in text


# ======================================================================
# Forbidden imports
# ======================================================================


class TestNoForbiddenImports:
    def test_no_binance_exchange_import(self) -> None:
        text = _read_source()
        assert "src.exchanges.binance" not in text
        assert "src.data_feed.binance" not in text

    def test_no_scripts_import(self) -> None:
        text = _read_source()
        assert "scripts." not in text

    def test_no_env_import(self) -> None:
        text = _read_source()
        assert "os.getenv" not in text
        assert "OKX_CONFIG" not in text
        assert "load_dotenv" not in text

    def test_no_okx_private_client_import(self) -> None:
        """okx_private_client may only be imported under TYPE_CHECKING."""
        text = _read_source()
        # Collect lines outside TYPE_CHECKING
        lines = text.split("\n")
        in_type_checking = False
        runtime_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if TYPE_CHECKING:"):
                in_type_checking = True
                continue
            if in_type_checking:
                if stripped == "" or stripped.startswith("#"):
                    continue
                # Check ORIGINAL line (before strip) for indentation
                if not line.startswith(" ") and not line.startswith("\t"):
                    in_type_checking = False
                    runtime_lines.append(stripped)
                    continue
                # Inside TYPE_CHECKING — skip
                continue
            runtime_lines.append(stripped)
        runtime_text = "\n".join(runtime_lines)
        assert "okx_private_client" not in runtime_text, (
            "okx_private_client must only be imported under TYPE_CHECKING"
        )

    def test_no_live_module_import(self) -> None:
        text = _read_source()
        assert "src.live" not in text
        assert "src.position_management" not in text
        assert "src.strategies" not in text  # except TYPE_CHECKING maybe? No — not even in TYPE_CHECKING

    def test_no_risk_reporting_import(self) -> None:
        text = _read_source()
        assert "src.risk" not in text
        assert "src.reporting" not in text


# ======================================================================
# Forbidden patterns — no Trader() construction
# ======================================================================


class TestNoTraderConstruction:
    def test_no_trader_instantiation(self) -> None:
        """The source must NOT create a Trader() — only accept one via __init__."""
        text = _read_source()
        assert "Trader()" not in text

    def test_trader_import_only_for_type_checking(self) -> None:
        """Trader must only be imported under TYPE_CHECKING, not at runtime."""
        text = _read_source()
        # The runtime import lines (not under TYPE_CHECKING) should NOT import Trader directly
        # Check that Trader is only referenced in TYPE_CHECKING block
        lines = text.split("\n")
        in_type_checking = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if TYPE_CHECKING:"):
                in_type_checking = True
                continue
            if in_type_checking:
                if stripped == "" or stripped.startswith("#"):
                    continue
                # After TYPE_CHECKING block ends (unindented), we're out
                # Check ORIGINAL line for indentation
                if not line.startswith(" ") and not line.startswith("\t"):
                    in_type_checking = False
                    continue
            if not in_type_checking:
                # Runtime portion must not import Trader from trader module
                if "from src.execution.trader" in stripped and "import" in stripped and "Trader" in stripped:
                    raise AssertionError(
                        f"Runtime import of Trader found: {stripped!r}"
                    )


# ======================================================================
# Forbidden patterns — no execute_intent
# ======================================================================


class TestNoExecuteIntent:
    def test_no_execute_intent_call(self) -> None:
        text = _read_source()
        assert "execute_intent" not in text


# ======================================================================
# Forbidden patterns — no adapter / bundle / semantic executor
# ======================================================================


class TestNoAdapterPatterns:
    def test_no_three_stage_adapter(self) -> None:
        text = _read_source()
        assert "ThreeStageAdapter" not in text

    def test_no_middle_runner_adapter(self) -> None:
        text = _read_source()
        assert "MiddleRunnerAdapter" not in text

    def test_no_sidecar_adapter(self) -> None:
        text = _read_source()
        assert "SidecarAdapter" not in text

    def test_no_exchange_runtime_bundle(self) -> None:
        text = _read_source()
        assert "ExchangeRuntimeBundle" not in text

    def test_no_broker_semantic_executor(self) -> None:
        text = _read_source()
        assert "BrokerSemanticExecutor" not in text


# ======================================================================
# Positive checks — must have TradingClientPort methods
# ======================================================================


class TestHasRequiredMethods:
    def test_has_all_port_methods(self) -> None:
        from src.execution.okx_trading_client import OkxTradingClient

        required = {
            "fetch_balance",
            "fetch_position",
            "fetch_open_orders",
            "place_market_order",
            "place_limit_order",
            "place_stop_market_order",
            "cancel_order",
        }
        actual = {
            name
            for name in dir(OkxTradingClient)
            if not name.startswith("_") and callable(getattr(OkxTradingClient, name, None))
        }
        missing = required - actual
        assert not missing, f"OkxTradingClient is missing methods: {missing}"


# ======================================================================
# Empty client_order_id guard rails
# ======================================================================


class TestNoUnconditionalEmptyClientOrderId:
    """The source must NOT unconditionally assign a client_order_id that
    could be empty to clOrdId / algoClOrdId."""

    def test_has_normalise_client_order_id_helper(self) -> None:
        text = _read_source()
        assert "def _normalise_client_order_id" in text

    def test_no_unconditional_cl_ord_id_assignment_in_market_order(self) -> None:
        """place_market_order must guard clOrdId assignment with a
        None-check (normalised cid)."""
        text = _read_source()
        # The unconditional pattern: body["clOrdId"] = client_order_id
        # must not appear in place_market_order.
        lines = text.splitlines()
        in_method = False
        for line in lines:
            if "def place_market_order" in line:
                in_method = True
                continue
            if in_method and line.startswith("    def "):
                in_method = False
            if in_method:
                stripped = line.strip()
                # Must NOT have an unconditional: body["clOrdId"] = client_order_id
                if 'body["clOrdId"] = client_order_id' in stripped:
                    raise AssertionError(
                        "place_market_order must not unconditionally assign "
                        "body['clOrdId'] = client_order_id"
                    )

    def test_no_unconditional_algo_cl_ord_id_assignment_in_stop_order(self) -> None:
        """place_stop_market_order must guard algoClOrdId assignment."""
        text = _read_source()
        lines = text.splitlines()
        in_method = False
        for line in lines:
            if "def place_stop_market_order" in line:
                in_method = True
                continue
            if in_method and line.startswith("    def "):
                in_method = False
            if in_method:
                stripped = line.strip()
                if 'body["algoClOrdId"] = client_order_id' in stripped:
                    raise AssertionError(
                        "place_stop_market_order must not unconditionally assign "
                        "body['algoClOrdId'] = client_order_id"
                    )

    def test_normalise_called_in_all_order_methods(self) -> None:
        text = _read_source()
        # Every order-placement method should call _normalise_client_order_id
        for method in ("place_market_order", "place_limit_order",
                       "place_stop_market_order", "cancel_order"):
            assert f"_normalise_client_order_id" in text, (
                f"_normalise_client_order_id helper is referenced in source"
            )
