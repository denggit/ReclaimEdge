#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_no_live_wiring.py
@Description: Boundary guard – prove the exchange abstraction is NOT wired
              into live trading paths beyond Trader lazy sidecar access.

Trader may lazily expose the OKX broker semantic executor, but adapters must
not be routed through TP/SL managers, live workers, strategies, or the live
entry script.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Files that must NOT reference the broker semantic sidecar / OKX adapter
# ---------------------------------------------------------------------------

LIVE_FILES_THAT_MUST_NOT_REFERENCE_EXCHANGE_ADAPTERS: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
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
