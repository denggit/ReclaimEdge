# -*- coding: utf-8 -*-
"""Tests for G05 portfolio allocator shadow mode."""

from __future__ import annotations

import asyncio
import os
import types
import unittest
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution.trader import PositionSnapshot
from src.portfolio.capital_allocator import AllocationCheckRequest, AllocationDecision
from src.portfolio.capital_ledger import (
    CapitalLedgerSnapshot,
    SymbolCapitalState,
    default_snapshot,
)
from src.portfolio.leader_follower import LeaderFollowerError, SymbolPermission
from src.portfolio.position_plan import PositionPlan
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy


# ── lightweight fakes ────────────────────────────────────────────────────────


class FakeTrader:
    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.account_equity_usdt = 1000.0
        self.contract_multiplier = Decimal("0.1")
        self.contract_precision = Decimal("0.01")
        self.min_contracts = Decimal("0.01")
        self.executed: list[int] = []


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def append(self, event_name: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event_name, dict(payload), position_id))

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"


class FakeTradeCommand:
    def __init__(
        self,
        intent_type: str = "OPEN_LONG",
        side: str = "LONG",
        layer_index: int = 1,
        margin_usdt: float = 30.0,
        eth_qty: float = 1.0,
        ts_ms: int = 1000,
    ) -> None:
        self.intent = FakeIntent(
            intent_type=intent_type,
            side=side,
            layer_index=layer_index,
            margin_usdt=margin_usdt,
            eth_qty=eth_qty,
            ts_ms=ts_ms,
        )
        self.strategy_state_snapshot = StrategyPositionState(side=side)
        self.tick_ts_ms = ts_ms
        self.created_monotonic = 0.0
        self.account_snapshot_updated_ts_ms = 0
        self.reason = "test"


class FakeIntent:
    def __init__(
        self,
        intent_type: str = "OPEN_LONG",
        side: str = "LONG",
        layer_index: int = 1,
        margin_usdt: float = 30.0,
        eth_qty: float = 1.0,
        ts_ms: int = 1000,
    ) -> None:
        self.intent_type = intent_type
        self.side = side
        self.layer_index = layer_index
        self.margin_usdt = margin_usdt
        self.eth_qty = eth_qty
        self.ts_ms = ts_ms
        # Create a minimal PositionSize mimic
        self.size = types.SimpleNamespace(margin_usdt=margin_usdt, eth_qty=eth_qty)
        # Other fields that TradeIntent normally has
        self.price = 3000.0
        self.tp_price = 3100.0
        self.reason = "test"
        self.fast_cvd = 1.0
        self.previous_fast_cvd = 0.0
        self.buy_ratio = 1.0
        self.sell_ratio = 0.0
        self.boll_upper = 3100.0
        self.boll_middle = 3000.0
        self.boll_lower = 2900.0
        self.avg_entry_price = 3000.0
        self.breakeven_price = 3000.0
        self.tp_mode = "MIDDLE"
        self.tp_plan = "SINGLE"


def make_strategy(max_layers: int = 8, layer_multiplier_step: float = 0.15) -> BollCvdShockReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(max_layers=max_layers)
    sizer_config = SimplePositionSizerConfig(layer_multiplier_step=layer_multiplier_step)
    return BollCvdShockReclaimStrategy(config, SimplePositionSizer(sizer_config))


def fake_ledger_snapshot(**overrides) -> CapitalLedgerSnapshot:
    """Build a snapshot with optional ETH state override and top-level overrides."""
    base = default_snapshot(updated_ms=1000)
    eth_state = overrides.pop("eth_state", None)
    global_no_new_entry = overrides.pop("global_no_new_entry", base.global_no_new_entry)
    leader_symbol = overrides.pop("leader_symbol", base.leader_symbol)

    symbols = dict(base.symbols)
    if eth_state is not None:
        symbols["ETH-USDT-SWAP"] = eth_state

    return CapitalLedgerSnapshot(
        version=base.version,
        updated_ms=base.updated_ms,
        leader_symbol=leader_symbol,
        global_no_new_entry=global_no_new_entry,
        symbols=symbols,
        **overrides,
    )


# ── tests ────────────────────────────────────────────────────────────────────


class TestPortfolioAllocatorShadowConfig:
    """Tests for PortfolioAllocatorShadowConfig.from_env()."""

    def test_default_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.enabled is False

    def test_env_true_enables(self) -> None:
        with patch.dict(os.environ, {"PORTFOLIO_ALLOCATOR_SHADOW_ENABLED": "true"}, clear=True):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.enabled is True

    def test_env_1_enables(self) -> None:
        with patch.dict(os.environ, {"PORTFOLIO_ALLOCATOR_SHADOW_ENABLED": "1"}, clear=True):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.enabled is True

    def test_default_runtime_dir_paths(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env(
                runtime_dir="runtime",
            )
            assert cfg.ledger_path == Path("runtime/portfolio/capital_ledger.json")
            assert cfg.lock_path == Path("runtime/portfolio/capital_ledger.lock")

    def test_custom_ledger_path_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PORTFOLIO_LEDGER_PATH": "/tmp/test_ledger.json"},
            clear=True,
        ):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.ledger_path == Path("/tmp/test_ledger.json")

    def test_custom_lock_path_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PORTFOLIO_LEDGER_LOCK_PATH": "/tmp/test_ledger.lock"},
            clear=True,
        ):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.lock_path == Path("/tmp/test_ledger.lock")

    def test_lock_timeout_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PORTFOLIO_ALLOCATOR_SHADOW_LOCK_TIMEOUT_SECONDS": "0.5"},
            clear=True,
        ):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.lock_timeout_seconds == 0.5

    def test_default_global_main_cap_pct(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.global_main_cap_pct == "0.70"

    def test_leader_follower_config_defaults_to_fixed_eth(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.leader_follower_config.leader_mode == "fixed"
            assert cfg.leader_follower_config.fixed_leader_symbol == "ETH-USDT-SWAP"

    def test_leader_follower_config_fixed_eth_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PORTFOLIO_LEADER_MODE": "fixed",
                "PORTFOLIO_FIXED_LEADER_SYMBOL": "ETH-USDT-SWAP",
            },
            clear=True,
        ):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.leader_follower_config.leader_mode == "fixed"
            assert cfg.leader_follower_config.fixed_leader_symbol == "ETH-USDT-SWAP"

    def test_leader_follower_config_dynamic_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PORTFOLIO_LEADER_MODE": "dynamic"},
            clear=True,
        ):
            cfg = __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()
            assert cfg.leader_follower_config.leader_mode == "dynamic"

    def test_leader_follower_config_invalid_mode_raises(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PORTFOLIO_LEADER_MODE": "invalid",
                "PORTFOLIO_FIXED_LEADER_SYMBOL": "",
            },
            clear=True,
        ):
            with pytest.raises(LeaderFollowerError):
                __import__("src.live.portfolio_allocator_shadow", fromlist=["PortfolioAllocatorShadowConfig"]).PortfolioAllocatorShadowConfig.from_env()


class TestShadowRunnerDisabled(unittest.IsolatedAsyncioTestCase):
    """Tests when shadow is disabled."""

    async def test_disabled_returns_without_journal(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=False)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        # Override ledger with a mock to verify it's never called
        runner.ledger = MagicMock()
        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG")
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        runner.ledger.read_locked.assert_not_called()
        assert len(journal.events) == 0

    async def test_disabled_does_not_raise(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=False)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        # Should not raise even with a broken ledger
        runner.ledger = None  # type: ignore[assignment]
        journal = FakeJournal()

        # This should not raise (returns before accessing ledger)
        await runner.run_entry_shadow_check(
            command=FakeTradeCommand(intent_type="ADD_LONG"),  # type: ignore[arg-type]
            trader=FakeTrader(),  # type: ignore[arg-type]
            strategy=make_strategy(),
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )
        assert len(journal.events) == 0


class TestShadowRunnerOpenMain(unittest.IsolatedAsyncioTestCase):
    """Tests for OPEN_MAIN shadow check."""

    async def test_open_main_allowed_records_journal(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        # Replace ledger with mock returning default snapshot
        snapshot = default_snapshot(updated_ms=1000)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(
            intent_type="OPEN_LONG",
            side="LONG",
            layer_index=1,
            margin_usdt=30.0,
            eth_qty=1.0,
        )
        trader = FakeTrader()
        strategy = make_strategy(max_layers=8, layer_multiplier_step=0.15)

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        payload = shadow_events[0][1]
        assert payload["allocator_action"] == "OPEN_MAIN"
        assert payload["allowed"] is True
        assert payload["reason"] == "OPEN_MAIN_ALLOWED"
        assert payload["shadow_mode"] is True
        assert payload["symbol"] == "ETH-USDT-SWAP"

    async def test_open_short_allowed_records_journal(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        snapshot = default_snapshot(updated_ms=1000)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(
            intent_type="OPEN_SHORT",
            side="SHORT",
            layer_index=1,
            margin_usdt=30.0,
            eth_qty=1.0,
        )
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        assert shadow_events[0][1]["allocator_action"] == "OPEN_MAIN"
        assert shadow_events[0][1]["allowed"] is True

    async def test_open_long_layer_not_1_still_checked(self) -> None:
        """OPEN_LONG with layer_index != 1 is forced to requested_layer=1 for shadow."""
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        snapshot = default_snapshot(updated_ms=1000)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        # Even if intent says layer_index=5, shadow maps to OPEN_MAIN with requested_layer=1
        command = FakeTradeCommand(
            intent_type="OPEN_LONG",
            side="LONG",
            layer_index=5,  # wrong, but shadow normalizes it
            margin_usdt=30.0,
            eth_qty=1.0,
        )
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        # G04 will reject because requested_layer=1 is required
        assert shadow_events[0][1]["allocator_action"] == "OPEN_MAIN"


class TestShadowRunnerAddMain(unittest.IsolatedAsyncioTestCase):
    """Tests for ADD_MAIN shadow check."""

    async def test_add_main_allowed_records_journal(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)

        # Build a snapshot where ETH is already OPEN layer1 with a plan
        eth_state = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=1,
            position_plan_id="plan-1",
            planned_main_contracts=("10", "11.5", "13", "14.5", "16", "17.5", "19", "20.5"),
            base_main_contracts="10",
            plan_max_layers=8,
            permission_max_layers=8,
            main_used_margin_usdt="30",
            sidecar_enabled=True,
        )
        snapshot = fake_ledger_snapshot(eth_state=eth_state)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(
            intent_type="ADD_LONG",
            side="LONG",
            layer_index=2,
            margin_usdt=34.5,
            eth_qty=1.15,
        )
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        payload = shadow_events[0][1]
        assert payload["allocator_action"] == "ADD_MAIN"
        assert payload["allowed"] is True
        assert payload["reason"] == "ADD_MAIN_ALLOWED"

    async def test_add_short_allowed_records_journal(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)

        eth_state = SymbolCapitalState(
            state="OPEN",
            side="SHORT",
            used_layers=1,
            position_plan_id="plan-s",
            planned_main_contracts=("10", "11.5", "13", "14.5", "16", "17.5", "19", "20.5"),
            base_main_contracts="10",
            plan_max_layers=8,
            permission_max_layers=8,
            main_used_margin_usdt="30",
            sidecar_enabled=True,
        )
        snapshot = fake_ledger_snapshot(eth_state=eth_state)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(
            intent_type="ADD_SHORT",
            side="SHORT",
            layer_index=2,
            margin_usdt=34.5,
            eth_qty=1.15,
        )
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        assert shadow_events[0][1]["allocator_action"] == "ADD_MAIN"
        assert shadow_events[0][1]["allowed"] is True


class TestShadowRunnerRejected(unittest.IsolatedAsyncioTestCase):
    """Tests for rejected shadow decisions."""

    async def test_global_no_new_entry_rejected(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)

        snapshot = fake_ledger_snapshot(global_no_new_entry=True)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG", side="LONG", layer_index=1)
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        payload = shadow_events[0][1]
        assert payload["allowed"] is False
        assert payload["reason"] == "GLOBAL_NO_NEW_ENTRY"
        # Should not raise
        assert True

    async def test_already_active_rejected(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)

        # ETH already OPEN
        eth_state = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=1,
            position_plan_id="plan-1",
            planned_main_contracts=("10",),
            base_main_contracts="10",
            plan_max_layers=8,
            permission_max_layers=8,
            main_used_margin_usdt="30",
        )
        snapshot = fake_ledger_snapshot(eth_state=eth_state)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG", side="LONG", layer_index=1)
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        shadow_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(shadow_events) == 1
        assert shadow_events[0][1]["allowed"] is False
        assert shadow_events[0][1]["reason"] == "SYMBOL_ALREADY_ACTIVE"


class TestShadowRunnerExceptionSwallowed(unittest.IsolatedAsyncioTestCase):
    """Tests that exceptions inside the shadow runner are swallowed."""

    async def test_ledger_read_exception_swallowed(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.side_effect = RuntimeError("disk full")

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG")
        trader = FakeTrader()
        strategy = make_strategy()

        # Must not raise
        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        failed_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW_FAILED"]
        assert len(failed_events) >= 1
        payload = failed_events[0][1]
        assert payload["intent_type"] == "OPEN_LONG"
        assert payload["shadow_mode"] is True
        assert "RuntimeError" in payload["error_type"] or "disk full" in payload["error"]

    async def test_non_entry_intent_returns_immediately(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        runner.ledger = MagicMock()

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="UPDATE_TP")
        trader = FakeTrader()
        strategy = make_strategy()

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # Ledger should NOT be read
        runner.ledger.read_locked.assert_not_called()
        assert len(journal.events) == 0


class TestShadowRunnerSidecar(unittest.IsolatedAsyncioTestCase):
    """Tests for sidecar shadow check."""

    async def test_sidecar_shadow_when_enabled(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)

        eth_state = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=1,
            position_plan_id="plan-1",
            planned_main_contracts=("10",),
            base_main_contracts="10",
            plan_max_layers=8,
            permission_max_layers=8,
            main_used_margin_usdt="30",
            sidecar_enabled=True,
        )
        snapshot = fake_ledger_snapshot(eth_state=eth_state)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG")
        trader = FakeTrader()
        strategy = make_strategy()

        # Create a simple sidecar plan mimic
        sidecar_plan = types.SimpleNamespace(enabled=True, sidecar_margin_pct=0.01)

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
            sidecar_plan=sidecar_plan,  # type: ignore[arg-type]
        )

        sidecar_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW_SIDECAR"]
        assert len(sidecar_events) == 1
        payload = sidecar_events[0][1]
        assert payload["allocator_action"] == "OPEN_SIDECAR"
        assert payload["shadow_mode"] is True

    async def test_sidecar_shadow_skipped_when_disabled(self) -> None:
        from src.live.portfolio_allocator_shadow import (
            PortfolioAllocatorShadowConfig,
            PortfolioAllocatorShadowRunner,
        )
        config = PortfolioAllocatorShadowConfig(enabled=True)
        runner = PortfolioAllocatorShadowRunner.from_config(config)
        snapshot = default_snapshot(updated_ms=1000)
        runner.ledger = MagicMock()
        runner.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG")
        trader = FakeTrader()
        strategy = make_strategy()

        # Sidecar plan says disabled
        sidecar_plan = types.SimpleNamespace(enabled=False, sidecar_margin_pct=0.01)

        await runner.run_entry_shadow_check(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
            sidecar_plan=sidecar_plan,  # type: ignore[arg-type]
        )

        # Main shadow event still recorded
        main_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW"]
        assert len(main_events) == 1
        # Sidecar shadow skipped
        sidecar_events = [e for e in journal.events if e[0] == "PORTFOLIO_ALLOCATOR_SHADOW_SIDECAR"]
        assert len(sidecar_events) == 0


if __name__ == "__main__":
    unittest.main()
