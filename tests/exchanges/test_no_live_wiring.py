#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_no_live_wiring.py
@Description: Boundary guard – prove the exchange abstraction is NOT wired
              into live trading paths yet.

This test is intentionally strict.  It will be relaxed or removed in a
later task once adapters are intentionally connected to Trader / workers.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Files that must NOT import anything from src.exchanges
# ---------------------------------------------------------------------------

LIVE_FILES_THAT_MUST_NOT_IMPORT_EXCHANGES: list[str] = [
    "scripts/run_boll_cvd_live.py",
    "src/execution/trader.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
    "src/live/workers/strategy_tick_worker.py",
    "src/live/workers/execution_worker.py",
    "src/live/workers/execution_command_processor.py",
]

# Symbols that must not appear in any live-file source text.
FORBIDDEN_SYMBOLS: list[str] = [
    "src.exchanges",
    "BrokerSemanticExecutor",
    "OkxBrokerClient",
]


def test_exchange_abstraction_not_wired_into_live_path_yet() -> None:
    """Prove that none of the live execution files import or reference the
    new exchange abstraction (src.exchanges, BrokerSemanticExecutor, or
    OkxBrokerClient)."""
    for file_name in LIVE_FILES_THAT_MUST_NOT_IMPORT_EXCHANGES:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text()
        for symbol in FORBIDDEN_SYMBOLS:
            assert symbol not in text, (
                f"{file_name} MUST NOT reference '{symbol}' in this skeleton step. "
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
