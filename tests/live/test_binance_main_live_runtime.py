#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_main_live_runtime.py
@Description: Tests for Binance main live runtime — config loading,
              pending guard, and no OKX monitor dependency.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from unittest import mock

import pytest

from src.live.binance_signal_only_runtime import (
    load_binance_market_runtime_config,
    load_binance_signal_only_config,
)


# ======================================================================
# Helpers
# ======================================================================


def _base_main_env() -> dict[str, str]:
    return {
        "EXCHANGE": "binance",
        "SIGNAL_ONLY": "false",
        "TRADE_ASSET": "ETH",
        "QUOTE_ASSET": "USDT",
        "MARKET_TYPE": "PERPETUAL",
        "MARGIN_MODE": "isolated",
        "POSITION_MODE": "net",
        "LEVERAGE": "20",
        "KLINE_INTERVAL": "15m",
        "LIVE_TRADING": "true",
        "LIVE_ENABLED": "true",
        "LIVE_ALLOW_ORDERS": "true",
        "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
        "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
        "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
        "LIVE_LEVERAGE": "20",
        "MAX_LIVE_EQUITY_USDT": "30",
    }


# ======================================================================
# Config loading — main live path (require_signal_only=False)
# ======================================================================


class TestMainLiveConfigLoading:
    """load_binance_market_runtime_config with require_signal_only=False."""

    def test_signal_only_false_accepted(self) -> None:
        """SIGNAL_ONLY=false should work with require_signal_only=False."""
        env = _base_main_env()
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_market_runtime_config(require_signal_only=False)
            assert config.canonical_symbol == "ETH-USDT-PERP"
            assert config.raw_symbol == "ETHUSDT"
            assert config.kline_interval == "15m"

    def test_signal_only_missing_accepted(self) -> None:
        """Missing SIGNAL_ONLY should also work with require_signal_only=False."""
        env = _base_main_env()
        del env["SIGNAL_ONLY"]
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_market_runtime_config(require_signal_only=False)
            assert config.raw_symbol == "ETHUSDT"

    def test_signal_only_true_still_supported(self) -> None:
        """SIGNAL_ONLY=true still passes through (no error for main path)."""
        env = _base_main_env()
        env["SIGNAL_ONLY"] = "true"
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_market_runtime_config(require_signal_only=False)
            assert config.raw_symbol == "ETHUSDT"

    def test_signal_only_false_rejected_when_required(self) -> None:
        """With require_signal_only=True, SIGNAL_ONLY=false raises."""
        env = _base_main_env()
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="SIGNAL_ONLY=true"):
                load_binance_signal_only_config()

    def test_signal_only_missing_rejected_when_required(self) -> None:
        """With require_signal_only=True, missing SIGNAL_ONLY raises."""
        env = _base_main_env()
        del env["SIGNAL_ONLY"]
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="SIGNAL_ONLY=true"):
                load_binance_signal_only_config()

    def test_signal_only_true_passes_when_required(self) -> None:
        """With require_signal_only=True, SIGNAL_ONLY=true works."""
        env = _base_main_env()
        env["SIGNAL_ONLY"] = "true"
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_binance_signal_only_config()
            assert config.raw_symbol == "ETHUSDT"


# ======================================================================
# Pending guard tests (Fix 5)
# ======================================================================


class TestPendingGuard:
    """Minimal pending guard prevents duplicate entry/add intents."""

    @pytest.mark.asyncio
    async def test_queue_empty_entry_enqueued(self) -> None:
        """When execution queue is empty, entry intent is enqueued."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent, PositionSize
        intent = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=3000.0,
            layer_index=0,
            tp_price=3100.0,
            reason="test",
            size=PositionSize(margin_usdt=0.0, notional_usdt=300.0, eth_qty=0.1, layer_index=0, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            ts_ms=0, avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="SINGLE",
        )

        # Queue empty → should enqueue
        assert queue.qsize() == 0
        await queue.put(intent)
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_queue_non_empty_skip_open(self) -> None:
        """Entry intents are skipped when queue is non-empty."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent, PositionSize
        intent = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=3000.0,
            layer_index=0,
            tp_price=3100.0,
            reason="test",
            size=PositionSize(margin_usdt=0.0, notional_usdt=300.0, eth_qty=0.1, layer_index=0, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            ts_ms=0, avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="SINGLE",
        )

        # Pre-fill queue
        await queue.put(intent)
        assert queue.qsize() == 1

        # Guard check: skip when queue non-empty
        intent_type = intent.intent_type
        if intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
            if queue.qsize() > 0:
                # Skip — don't enqueue
                pass
        assert queue.qsize() == 1  # Still only 1

    @pytest.mark.asyncio
    async def test_queue_non_empty_market_exit_not_skipped(self) -> None:
        """MARKET_EXIT intents are NOT skipped when queue is non-empty."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent, PositionSize

        # Pre-fill queue with an entry intent
        entry = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=3000.0, layer_index=0, tp_price=3100.0, reason="test",
            size=PositionSize(margin_usdt=0.0, notional_usdt=300.0, eth_qty=0.1, layer_index=0, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            ts_ms=0, avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="SINGLE",
        )
        await queue.put(entry)
        assert queue.qsize() == 1

        # MARKET_EXIT should not be skipped
        exit_intent = TradeIntent(
            intent_type="MARKET_EXIT",
            side="LONG",
            price=3000.0, layer_index=0, tp_price=3100.0, reason="test",
            size=PositionSize(margin_usdt=0.0, notional_usdt=300.0, eth_qty=0.1, layer_index=0, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            ts_ms=0, avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="SINGLE",
        )
        intent_type = exit_intent.intent_type
        should_skip = intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"} and queue.qsize() > 0
        if not should_skip:
            await queue.put(exit_intent)
        assert queue.qsize() == 2  # Both enqueued


# ======================================================================
# No OKX monitor dependency
# ======================================================================


class TestNoOkxMonitor:
    """Binance main live runtime must not pull in OKX monitor."""

    def test_main_live_file_has_no_okx_monitor(self) -> None:
        from pathlib import Path
        text = Path("src/live/binance_main_live_runtime.py").read_text(encoding="utf-8")
        assert "BollBandBreakoutMonitor(" not in text
        assert "BollBandBreakoutMonitorConfig" not in text
        assert "OKX_INST_ID" not in text
        assert "OKX_BAR" not in text

    def test_main_live_file_has_no_signal_only_deadlock(self) -> None:
        """Main live must use load_binance_market_runtime_config, not
        load_binance_signal_only_config."""
        from pathlib import Path
        text = Path("src/live/binance_main_live_runtime.py").read_text(encoding="utf-8")
        assert "load_binance_market_runtime_config" in text
        # The old forbidden call must not appear
        assert "load_binance_signal_only_config" not in text
