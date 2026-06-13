#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_no_live_wiring.py
@Description: Boundary guard – prove the exchange abstraction is NOT wired
              into live trading paths beyond Trader lazy sidecar access and
              the optional core TP semantic placement switch, targeted
              TP/SL semantic cancel switches, the optional market-exit
              semantic placement switch, and the optional sidecar TP semantic
              placement switch.

Trader may lazily expose the OKX broker semantic executor.  Core TP may own
the optional semantic TP placement switch; the TP/SL execution manager may own
targeted semantic cancel switches; the market-exit manager may own the optional
semantic market-exit placement switch; the sidecar manager may own the optional
semantic sidecar TP placement switch.  Adapters must not be routed through
other TP/SL managers, live workers, strategies, or the live entry script.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Files that must NOT reference the broker semantic sidecar / OKX adapter
# ---------------------------------------------------------------------------

LIVE_FILES_THAT_MUST_NOT_REFERENCE_EXCHANGE_ADAPTERS: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/live/workers/strategy_tick_worker.py",
    "src/live/workers/execution_worker.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/workers/account_position_sync_worker.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

# Symbols that must not appear in any live-file source text.
FORBIDDEN_SYMBOLS: list[str] = [
    "BROKER_SEMANTIC_READS_ENABLED",
    "fetch_broker_open_orders",
    "fetch_broker_algo_orders",
    "recover_broker_open_orders",
    "broker_semantic_executor",
    "src.exchanges.okx",
    "OkxBrokerClient",
    "OkxBrokerSemanticExecutor",
]


def test_exchange_adapter_not_wired_into_live_path_yet() -> None:
    """Prove non-Trader live paths do not reference the broker semantic sidecar."""
    for file_name in LIVE_FILES_THAT_MUST_NOT_REFERENCE_EXCHANGE_ADAPTERS:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text()
        for symbol in FORBIDDEN_SYMBOLS:
            assert symbol not in text, (
                f"{file_name} MUST NOT reference '{symbol}' before routing is intentionally switched. "
                f"Found in {file_name}"
            )


def test_semantic_tp_placement_switch_only_lives_in_core_tp_manager() -> None:
    guarded_files = [
        "scripts/run_boll_cvd_live.py",
        "src/execution/trader.py",
        "src/execution/tp_sl_execution_manager.py",
        "src/execution/tp_sl_protective_stop_manager.py",
        "src/execution/tp_sl_market_exit_manager.py",
        "src/execution/tp_sl_near_tp_manager.py",
        "src/execution/tp_sl_sidecar_manager.py",
        "src/live/workers/execution_command_processor.py",
    ]
    forbidden_tokens = [
        "BROKER_SEMANTIC_TP_PLACEMENT_ENABLED",
        "_place_reduce_only_take_profit_order_semantic",
    ]

    for file_name in guarded_files:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"{token} unexpectedly found in {file_name}"


# ---------------------------------------------------------------------------
# Additional guard – the generic models themselves must not import live code
# ---------------------------------------------------------------------------

GENERIC_MODEL_FILES: list[str] = [
    "src/exchanges/models.py",
    "src/exchanges/errors.py",
    "src/exchanges/capabilities.py",
    "src/exchanges/base.py",
    "src/exchanges/semantic_models.py",
    "src/exchanges/semantics.py",
]

FORBIDDEN_MODEL_IMPORTS: list[str] = [
    "import src.execution.trader",
    "import src.execution.order_specs",
    "from src.execution.trader",
    "from src.execution.order_specs",
    "from src.exchanges.okx",
    "import src.live",
    "from src.live",
    "import ccxt",
    "import okx",
    "import binance",
]


def test_generic_models_do_not_depend_on_live_or_exchange_adapters() -> None:
    """Generic port modules must not import live / trader / concrete-adapter
    code.  They are ports, not adapters."""
    for file_name in GENERIC_MODEL_FILES:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text()
        for forbidden in FORBIDDEN_MODEL_IMPORTS:
            assert forbidden not in text, (
                f"{file_name} MUST NOT '{forbidden}'"
            )


# ---------------------------------------------------------------------------
# Additional guard – semantic reduce-only cancel switch boundary
# ---------------------------------------------------------------------------

FILES_FORBIDDEN_SEMANTIC_REDUCE_ONLY_CANCEL: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

FORBIDDEN_SEMANTIC_REDUCE_ONLY_CANCEL_TOKENS: list[str] = [
    "BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED",
    "_cancel_reduce_only_order_semantic",
]


def test_semantic_reduce_only_cancel_switch_boundary() -> None:
    """The reduce-only cancel semantic switch must only live in
    tp_sl_execution_manager.py and its tests."""
    for file_name in FILES_FORBIDDEN_SEMANTIC_REDUCE_ONLY_CANCEL:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SEMANTIC_REDUCE_ONLY_CANCEL_TOKENS:
            assert token not in text, f"{token} unexpectedly found in {file_name}"


# ---------------------------------------------------------------------------
# Additional guard – semantic protective SL cancel switch boundary
# ---------------------------------------------------------------------------

FILES_FORBIDDEN_SEMANTIC_PROTECTIVE_SL_CANCEL: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

FORBIDDEN_SEMANTIC_PROTECTIVE_SL_CANCEL_TOKENS: list[str] = [
    "BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED",
    "_cancel_protective_stop_semantic",
]


def test_semantic_protective_sl_cancel_switch_boundary() -> None:
    """The protective SL cancel semantic switch must only live in
    tp_sl_execution_manager.py and its tests."""
    for file_name in FILES_FORBIDDEN_SEMANTIC_PROTECTIVE_SL_CANCEL:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SEMANTIC_PROTECTIVE_SL_CANCEL_TOKENS:
            assert token not in text, f"{token} unexpectedly found in {file_name}"


# ---------------------------------------------------------------------------
# Additional guard – semantic protective SL placement switch boundary
# ---------------------------------------------------------------------------

FILES_FORBIDDEN_SEMANTIC_PROTECTIVE_SL_PLACEMENT: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

FORBIDDEN_SEMANTIC_PROTECTIVE_SL_PLACEMENT_TOKENS: list[str] = [
    "BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED",
    "_place_primary_protective_stop_semantic",
]


def test_semantic_protective_sl_placement_switch_boundary() -> None:
    """The protective SL placement semantic switch must only live in
    tp_sl_protective_stop_manager.py and its tests."""
    for file_name in FILES_FORBIDDEN_SEMANTIC_PROTECTIVE_SL_PLACEMENT:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SEMANTIC_PROTECTIVE_SL_PLACEMENT_TOKENS:
            assert token not in text, f"{token} unexpectedly found in {file_name}"


# ---------------------------------------------------------------------------
# Additional guard – semantic market exit placement switch boundary
# ---------------------------------------------------------------------------

FILES_FORBIDDEN_SEMANTIC_MARKET_EXIT_PLACEMENT: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

SEMANTIC_MARKET_EXIT_PLACEMENT_TOKENS: list[str] = [
    "BROKER_SEMANTIC_MARKET_EXIT_ENABLED",
    "_place_market_exit_order_semantic",
]

ALLOWED_SEMANTIC_MARKET_EXIT_PLACEMENT_FILES: set[str] = {
    "src/execution/tp_sl_market_exit_manager.py",
    "tests/test_tp_sl_market_exit_manager_semantic_exit.py",
    "tests/exchanges/test_no_live_wiring.py",
    "tests/exchanges/okx/test_okx_semantic_order_body_parity.py",
}


def test_semantic_market_exit_placement_switch_boundary() -> None:
    """The market-exit semantic switch must only live in the market-exit
    manager and its tests."""
    for file_name in FILES_FORBIDDEN_SEMANTIC_MARKET_EXIT_PLACEMENT:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in SEMANTIC_MARKET_EXIT_PLACEMENT_TOKENS:
            assert token not in text, f"{token} unexpectedly found in {file_name}"

    root = Path(".")
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        file_name = path.as_posix()
        text = path.read_text(encoding="utf-8")
        for token in SEMANTIC_MARKET_EXIT_PLACEMENT_TOKENS:
            if token in text:
                assert file_name in ALLOWED_SEMANTIC_MARKET_EXIT_PLACEMENT_FILES, (
                    f"{token} must only appear in market-exit semantic placement files; "
                    f"found in {file_name}"
                )


# ---------------------------------------------------------------------------
# Additional guard – semantic sidecar TP placement switch boundary
# ---------------------------------------------------------------------------

FILES_FORBIDDEN_SEMANTIC_SIDECAR_TP_PLACEMENT: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

SEMANTIC_SIDECAR_TP_PLACEMENT_TOKENS: list[str] = [
    "BROKER_SEMANTIC_SIDECAR_TP_PLACEMENT_ENABLED",
    "_place_sidecar_take_profit_semantic",
]

ALLOWED_SEMANTIC_SIDECAR_TP_PLACEMENT_FILES: set[str] = {
    "src/execution/tp_sl_sidecar_manager.py",
    "tests/test_tp_sl_sidecar_manager_semantic_tp_placement.py",
    "tests/exchanges/test_no_live_wiring.py",
    "tests/exchanges/okx/test_okx_semantic_order_body_parity.py",
}


def test_semantic_sidecar_tp_placement_switch_boundary() -> None:
    """The sidecar TP semantic switch must only live in the sidecar manager
    and its tests."""
    for file_name in FILES_FORBIDDEN_SEMANTIC_SIDECAR_TP_PLACEMENT:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in SEMANTIC_SIDECAR_TP_PLACEMENT_TOKENS:
            assert token not in text, f"{token} unexpectedly found in {file_name}"

    root = Path(".")
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        file_name = path.as_posix()
        text = path.read_text(encoding="utf-8")
        for token in SEMANTIC_SIDECAR_TP_PLACEMENT_TOKENS:
            if token in text:
                assert file_name in ALLOWED_SEMANTIC_SIDECAR_TP_PLACEMENT_FILES, (
                    f"{token} must only appear in sidecar TP semantic placement files; "
                    f"found in {file_name}"
                )


# ---------------------------------------------------------------------------
# Additional guard – semantic sidecar TP cancel switch boundary
# ---------------------------------------------------------------------------

FILES_FORBIDDEN_SEMANTIC_SIDECAR_TP_CANCEL: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/live/workers/execution_command_processor.py",
    "src/live/account_sync/protective_orders_phase.py",
    "src/live/startup_recovery/order_recovery.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

SEMANTIC_SIDECAR_TP_CANCEL_TOKENS: list[str] = [
    "BROKER_SEMANTIC_SIDECAR_TP_CANCEL_ENABLED",
    "_cancel_sidecar_take_profit_semantic",
]

ALLOWED_SEMANTIC_SIDECAR_TP_CANCEL_FILES: set[str] = {
    "src/execution/tp_sl_sidecar_manager.py",
    "tests/test_tp_sl_sidecar_manager_semantic_tp_cancel.py",
    "tests/exchanges/test_no_live_wiring.py",
    "tests/exchanges/okx/test_okx_semantic_order_body_parity.py",
}


def test_semantic_sidecar_tp_cancel_switch_boundary() -> None:
    """The sidecar TP cancel semantic switch must only live in the sidecar
    manager and its tests."""
    for file_name in FILES_FORBIDDEN_SEMANTIC_SIDECAR_TP_CANCEL:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in SEMANTIC_SIDECAR_TP_CANCEL_TOKENS:
            assert token not in text, f"{token} unexpectedly found in {file_name}"

    root = Path(".")
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        file_name = path.as_posix()
        text = path.read_text(encoding="utf-8")
        for token in SEMANTIC_SIDECAR_TP_CANCEL_TOKENS:
            if token in text:
                assert file_name in ALLOWED_SEMANTIC_SIDECAR_TP_CANCEL_FILES, (
                    f"{token} must only appear in sidecar TP semantic cancel files; "
                    f"found in {file_name}"
                )
