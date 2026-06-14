#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_trader_broker_semantic_injection.py
@Description: Tests for Trader broker semantic executor lazy injection.
"""

from __future__ import annotations

from pathlib import Path

from src.execution.trader import Trader
from src.exchanges.models import ExchangeName


def test_trader_broker_exchange_name_defaults_to_okx() -> None:
    trader = object.__new__(Trader)

    assert trader.broker_exchange_name == "okx"


def test_trader_broker_semantic_executor_raises_when_not_bound() -> None:
    """After the adapter freeze, broker_semantic_executor is explicitly bound
    by the runtime factory.  Accessing it before binding raises RuntimeError."""
    trader = object.__new__(Trader)
    trader._broker_client = None
    trader._broker_semantic_executor = None
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "long_short"

    import pytest

    with pytest.raises(RuntimeError, match="broker_semantic_executor_not_bound"):
        _ = trader.broker_semantic_executor


def test_trader_broker_semantic_executor_works_when_bound() -> None:
    """When properly bound, broker_semantic_executor returns the injected executor."""
    from unittest import mock

    trader = object.__new__(Trader)
    trader._broker_client = None
    trader._broker_semantic_executor = None
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "long_short"

    fake_executor = mock.MagicMock()
    fake_executor.exchange = ExchangeName.OKX
    trader.bind_broker_semantic_executor(fake_executor)

    executor = trader.broker_semantic_executor

    assert executor is fake_executor
    assert trader._broker_semantic_executor is fake_executor
    assert executor.exchange == ExchangeName.OKX


def test_trader_broker_semantic_executor_imports_are_lazy() -> None:
    text = Path("src/execution/trader.py").read_text()
    top_import_section = text.split("class Trader", 1)[0]

    assert "src.exchanges.okx.client" not in top_import_section
    assert "src.exchanges.okx.semantic_executor" not in top_import_section


def test_execute_intent_does_not_route_to_broker_semantic_executor() -> None:
    text = Path("src/execution/trader.py").read_text()
    execute_intent_block = text.split("async def execute_intent", 1)[1].split(
        "\n    async def ",
        1,
    )[0]

    assert "broker_semantic_executor" not in execute_intent_block
    assert "OkxBrokerClient" not in execute_intent_block
    assert "OkxBrokerSemanticExecutor" not in execute_intent_block
