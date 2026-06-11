# -*- coding: utf-8 -*-
"""Tests for G06a portfolio allocator enforce mode."""

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

from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.portfolio.capital_allocator import AllocationCheckRequest, AllocationDecision
from src.portfolio.capital_ledger import (
    CapitalLedgerSnapshot,
    SymbolCapitalState,
    default_snapshot,
)
from src.portfolio.leader_follower import SymbolPermission
from src.portfolio.position_plan import PositionPlan
from src.risk.simple_position_sizer import (
    PositionSize,
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
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

    def append(
        self, event_name: str, payload: dict, position_id: str | None = None
    ) -> None:
        self.events.append((event_name, dict(payload), position_id))

    def new_position_id(
        self, symbol: str, side: str, ts_ms: int | None = None
    ) -> str:
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


def make_strategy(
    max_layers: int = 8, layer_multiplier_step: float = 0.15
) -> BollCvdShockReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(max_layers=max_layers)
    sizer_config = SimplePositionSizerConfig(
        layer_multiplier_step=layer_multiplier_step
    )
    return BollCvdShockReclaimStrategy(config, SimplePositionSizer(sizer_config))


def fake_ledger_snapshot(**overrides) -> CapitalLedgerSnapshot:
    """Build a snapshot with optional ETH state override and top-level overrides."""
    base = default_snapshot(updated_ms=1000)
    eth_state = overrides.pop("eth_state", None)
    global_no_new_entry = overrides.pop(
        "global_no_new_entry", base.global_no_new_entry
    )
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


class TestEnforceConfig:
    """Tests for PortfolioAllocatorEnforceConfig.from_env()."""

    def test_default_disabled(self) -> None:
        """1. PORTFOLIO_ALLOCATOR_ENFORCE_ENABLED default false."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__(
                "src.live.portfolio_allocator_enforcer",
                fromlist=["PortfolioAllocatorEnforceConfig"],
            ).PortfolioAllocatorEnforceConfig.from_env()
            assert cfg.enabled is False

    def test_default_lock_timeout(self) -> None:
        """lock timeout default 1.0."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__(
                "src.live.portfolio_allocator_enforcer",
                fromlist=["PortfolioAllocatorEnforceConfig"],
            ).PortfolioAllocatorEnforceConfig.from_env()
            assert cfg.lock_timeout_seconds == 1.0

    def test_default_ledger_paths(self) -> None:
        """default runtime/portfolio ledger path."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = __import__(
                "src.live.portfolio_allocator_enforcer",
                fromlist=["PortfolioAllocatorEnforceConfig"],
            ).PortfolioAllocatorEnforceConfig.from_env(runtime_dir="runtime")
            assert cfg.ledger_path == Path("runtime/portfolio/capital_ledger.json")
            assert cfg.lock_path == Path("runtime/portfolio/capital_ledger.lock")

    def test_env_true_enabled(self) -> None:
        """2. PORTFOLIO_ALLOCATOR_ENFORCE_ENABLED=true -> enabled True."""
        with patch.dict(
            os.environ,
            {"PORTFOLIO_ALLOCATOR_ENFORCE_ENABLED": "true"},
            clear=True,
        ):
            cfg = __import__(
                "src.live.portfolio_allocator_enforcer",
                fromlist=["PortfolioAllocatorEnforceConfig"],
            ).PortfolioAllocatorEnforceConfig.from_env()
            assert cfg.enabled is True

    def test_lock_timeout_env(self) -> None:
        """PORTFOLIO_ALLOCATOR_ENFORCE_LOCK_TIMEOUT_SECONDS=0.5 -> 0.5."""
        with patch.dict(
            os.environ,
            {"PORTFOLIO_ALLOCATOR_ENFORCE_LOCK_TIMEOUT_SECONDS": "0.5"},
            clear=True,
        ):
            cfg = __import__(
                "src.live.portfolio_allocator_enforcer",
                fromlist=["PortfolioAllocatorEnforceConfig"],
            ).PortfolioAllocatorEnforceConfig.from_env()
            assert cfg.lock_timeout_seconds == 0.5

    def test_custom_ledger_path_env(self) -> None:
        with patch.dict(
            os.environ,
            {"PORTFOLIO_LEDGER_PATH": "/tmp/test_ledger.json"},
            clear=True,
        ):
            cfg = __import__(
                "src.live.portfolio_allocator_enforcer",
                fromlist=["PortfolioAllocatorEnforceConfig"],
            ).PortfolioAllocatorEnforceConfig.from_env()
            assert cfg.ledger_path == Path("/tmp/test_ledger.json")


class TestEnforcerDisabled(unittest.IsolatedAsyncioTestCase):
    """Tests when enforce is disabled."""

    async def test_disabled_precheck_allowed(self) -> None:
        """3. enabled=False: returns enabled=False, allowed=True, reason=ENFORCE_DISABLED."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=False)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()
        journal = FakeJournal()

        result = await enforcer.precheck_entry_allocation(
            command=FakeTradeCommand(intent_type="OPEN_LONG"),  # type: ignore[arg-type]
            trader=FakeTrader(),  # type: ignore[arg-type]
            strategy=make_strategy(),
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        assert result.enabled is False
        assert result.allowed is True
        assert result.reason == "ENFORCE_DISABLED"
        # Should not read ledger
        enforcer.ledger.read_locked.assert_not_called()
        assert len(journal.events) == 0


class TestEnforcerNonEntry(unittest.IsolatedAsyncioTestCase):
    """Tests for non-entry intents."""

    async def test_update_tp_not_enforced(self) -> None:
        """4. UPDATE_TP: allowed=True, reason=NON_ENTRY_INTENT, no ledger read."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()
        journal = FakeJournal()

        result = await enforcer.precheck_entry_allocation(
            command=FakeTradeCommand(intent_type="UPDATE_TP"),  # type: ignore[arg-type]
            trader=FakeTrader(),  # type: ignore[arg-type]
            strategy=make_strategy(),
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        assert result.enabled is True
        assert result.allowed is True
        assert result.reason == "NON_ENTRY_INTENT"
        enforcer.ledger.read_locked.assert_not_called()


class TestEnforcerOpenMainAllowed(unittest.IsolatedAsyncioTestCase):
    """Tests for OPEN_MAIN allowed."""

    async def test_open_main_allowed(self) -> None:
        """5. OPEN_LONG layer1: allowed=True, projected_snapshot ETH state OPEN."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        snapshot = default_snapshot(updated_ms=1000)
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = snapshot

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

        result = await enforcer.precheck_entry_allocation(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        assert result.enabled is True
        assert result.allowed is True
        assert result.reason == "ALLOCATOR_ENFORCE_ALLOWED"
        assert result.main_decision is not None
        assert result.main_decision.reason == "OPEN_MAIN_ALLOWED"
        assert result.projected_snapshot is not None

        eth_state = result.projected_snapshot.symbols.get("ETH-USDT-SWAP")
        assert eth_state is not None
        assert eth_state.state == "OPEN"
        assert eth_state.used_layers == 1

        # Journal should have PRECHECK event
        precheck_events = [
            e
            for e in journal.events
            if e[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_PRECHECK"
        ]
        assert len(precheck_events) == 1
        assert precheck_events[0][1]["enforce_mode"] is True


class TestEnforcerOpenMainRejected(unittest.IsolatedAsyncioTestCase):
    """Tests for OPEN_MAIN rejected."""

    async def test_global_no_new_entry_rejected(self) -> None:
        """6. snapshot.global_no_new_entry=True: allowed=False, reason=GLOBAL_NO_NEW_ENTRY."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        snapshot = fake_ledger_snapshot(global_no_new_entry=True)
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG", side="LONG", layer_index=1)
        trader = FakeTrader()
        strategy = make_strategy()

        result = await enforcer.precheck_entry_allocation(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        assert result.allowed is False
        assert result.reason == "GLOBAL_NO_NEW_ENTRY"

        # Should have REJECTED journal event
        rejected_events = [
            e
            for e in journal.events
            if e[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED"
        ]
        assert len(rejected_events) == 1
        assert rejected_events[0][1]["reason"] == "GLOBAL_NO_NEW_ENTRY"


class TestEnforcerAddMain(unittest.IsolatedAsyncioTestCase):
    """Tests for ADD_MAIN."""

    async def test_add_main_allowed(self) -> None:
        """7. ADD_LONG layer2 with ETH OPEN layer1: allowed, used_layers=2."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)

        eth_state = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=1,
            position_plan_id="plan-1",
            planned_main_contracts=(
                "10", "11.5", "13", "14.5", "16", "17.5", "19", "20.5",
            ),
            base_main_contracts="10",
            plan_max_layers=8,
            permission_max_layers=8,
            main_used_margin_usdt="30",
            sidecar_enabled=True,
        )
        snapshot = fake_ledger_snapshot(eth_state=eth_state)
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = snapshot

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

        result = await enforcer.precheck_entry_allocation(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        assert result.allowed is True
        assert result.projected_snapshot is not None
        eth_state_after = result.projected_snapshot.symbols.get("ETH-USDT-SWAP")
        assert eth_state_after is not None
        assert eth_state_after.used_layers == 2

    async def test_add_main_permission_rejected(self) -> None:
        """8. ADD rejected due to permission (no_add_layer)."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        from src.portfolio.leader_follower import LeaderFollowerConfig

        config = PortfolioAllocatorEnforceConfig(
            enabled=True,
            leader_follower_config=LeaderFollowerConfig(leader_mode="dynamic"),
        )
        enforcer = PortfolioAllocatorEnforcer.from_config(config)

        # ETH has used_layers=1, leader is BTC with max_layers=1
        # This will make ETH a follower with permission_max_layers=1 and no_add_layer=True
        btc_state = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=5,
            position_plan_id="btc-plan",
            planned_main_contracts=("1", "2", "3", "4", "5"),
            base_main_contracts="1",
            plan_max_layers=5,
            permission_max_layers=5,
            main_used_margin_usdt="100",
        )
        eth_state = SymbolCapitalState(
            state="OPEN",
            side="LONG",
            used_layers=1,
            position_plan_id="plan-1",
            planned_main_contracts=("10", "11.5"),
            base_main_contracts="10",
            plan_max_layers=2,
            permission_max_layers=2,
            main_used_margin_usdt="30",
            sidecar_enabled=True,
        )
        snapshot = fake_ledger_snapshot(
            eth_state=eth_state,
            leader_symbol="BTC-USDT-SWAP",
        )
        # Also add BTC to the snapshot
        symbols = dict(snapshot.symbols)
        symbols["BTC-USDT-SWAP"] = btc_state
        snapshot = CapitalLedgerSnapshot(
            version=snapshot.version,
            updated_ms=snapshot.updated_ms,
            leader_symbol="BTC-USDT-SWAP",
            global_no_new_entry=snapshot.global_no_new_entry,
            symbols=symbols,
        )

        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = snapshot

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

        result = await enforcer.precheck_entry_allocation(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # This should be rejected for some reason (permission layer limit or no_add)
        assert result.allowed is False
        assert result.reason in (
            "PERMISSION_NO_ADD_LAYER",
            "PERMISSION_LAYER_LIMIT",
        )


class TestEnforcerSidecar(unittest.IsolatedAsyncioTestCase):
    """Tests for sidecar enforce."""

    async def test_sidecar_rejected_rejects_whole_entry(self) -> None:
        """9. sidecar rejected -> whole entry rejected, reason starts with SIDECAR_."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)

        # ETH is FLAT with sidecar_enabled=False — main OPEN_MAIN will succeed
        # (creating projected snapshot with sidecar_enabled=False),
        # but the sidecar OPEN_SIDECAR check will fail with SIDECAR_DISABLED.
        eth_state = SymbolCapitalState(
            state="FLAT",
            side=None,
            used_layers=0,
            main_used_margin_usdt="0",
            sidecar_enabled=False,  # sidecar disabled
        )
        snapshot = fake_ledger_snapshot(eth_state=eth_state)
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.return_value = snapshot

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG", side="LONG", layer_index=1)
        trader = FakeTrader()
        strategy = make_strategy()

        sidecar_plan = types.SimpleNamespace(enabled=True, sidecar_margin_pct=0.01)

        result = await enforcer.precheck_entry_allocation(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
            sidecar_plan=sidecar_plan,  # type: ignore[arg-type]
        )

        assert result.allowed is False
        assert result.reason.startswith("SIDECAR_")

        # Should have sidecar precheck journal event
        sidecar_events = [
            e
            for e in journal.events
            if e[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_PRECHECK_SIDECAR"
        ]
        assert len(sidecar_events) == 1


class TestEnforcerCommit(unittest.IsolatedAsyncioTestCase):
    """Tests for commit_projected_snapshot_after_fill."""

    async def test_commit_after_filled(self) -> None:
        """10. committed when entry_filled=True."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
            PortfolioAllocatorPrecheckResult,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()
        snapshot = default_snapshot(updated_ms=1000)
        enforcer.ledger.update_locked.return_value = snapshot

        precheck = PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=True,
            reason="ALLOCATOR_ENFORCE_ALLOWED",
            projected_snapshot=snapshot,
        )

        journal = FakeJournal()
        live_result = LiveTradeResult(
            ok=True,
            action="OPEN_LONG",
            order_id="ord-1",
            tp_order_id="tp-1",
            contracts="10",
            tp_price="101",
            message="ok",
            entry_filled=True,
            tp_ok=True,
        )

        await enforcer.commit_projected_snapshot_after_fill(
            precheck_result=precheck,
            live_result=live_result,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # Ledger update should have been called
        enforcer.ledger.update_locked.assert_called_once()

        # Should have COMMITTED journal event
        committed_events = [
            e
            for e in journal.events
            if e[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_COMMITTED"
        ]
        assert len(committed_events) == 1

    async def test_no_commit_when_order_failed_not_filled(self) -> None:
        """11. ok=False, entry_filled=False: no write."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
            PortfolioAllocatorPrecheckResult,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()

        precheck = PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=True,
            reason="ALLOCATOR_ENFORCE_ALLOWED",
            projected_snapshot=default_snapshot(),
        )

        journal = FakeJournal()
        live_result = LiveTradeResult(
            ok=False,
            action="OPEN_LONG",
            order_id="",
            tp_order_id="",
            contracts="",
            tp_price="",
            message="insufficient margin",
            entry_filled=False,
            tp_ok=False,
        )

        await enforcer.commit_projected_snapshot_after_fill(
            precheck_result=precheck,
            live_result=live_result,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # Should NOT write ledger
        enforcer.ledger.update_locked.assert_not_called()

    async def test_commit_when_entry_filled_even_if_not_ok(self) -> None:
        """12. ok=False but entry_filled=True: still commits."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
            PortfolioAllocatorPrecheckResult,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()
        snapshot = default_snapshot(updated_ms=1000)
        enforcer.ledger.update_locked.return_value = snapshot

        precheck = PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=True,
            reason="ALLOCATOR_ENFORCE_ALLOWED",
            projected_snapshot=snapshot,
        )

        journal = FakeJournal()
        live_result = LiveTradeResult(
            ok=False,
            action="OPEN_LONG",
            order_id="ord-1",
            tp_order_id="",
            contracts="10",
            tp_price="",
            message="tp placement failed",
            entry_filled=True,
            tp_ok=False,
        )

        await enforcer.commit_projected_snapshot_after_fill(
            precheck_result=precheck,
            live_result=live_result,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # Should write ledger even though ok=False
        enforcer.ledger.update_locked.assert_called_once()

    async def test_no_commit_when_enforce_disabled(self) -> None:
        """commit_projected_snapshot_after_fill returns early when config.enabled=False."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
            PortfolioAllocatorPrecheckResult,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=False)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()

        precheck = PortfolioAllocatorPrecheckResult(
            enabled=False,
            allowed=True,
            reason="ENFORCE_DISABLED",
            projected_snapshot=default_snapshot(),
        )

        journal = FakeJournal()
        live_result = LiveTradeResult(
            ok=True,
            action="OPEN_LONG",
            order_id="ord-1",
            tp_order_id="tp-1",
            contracts="10",
            tp_price="101",
            message="ok",
            entry_filled=True,
            tp_ok=True,
        )

        await enforcer.commit_projected_snapshot_after_fill(
            precheck_result=precheck,
            live_result=live_result,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # Should NOT write ledger
        enforcer.ledger.update_locked.assert_not_called()


class TestEnforcerFailClosed(unittest.IsolatedAsyncioTestCase):
    """Tests for fail-closed behavior."""

    async def test_ledger_read_exception_fail_closed(self) -> None:
        """13. ledger.read_locked raises: allowed=False, reason=ALLOCATOR_ENFORCE_ERROR."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()
        enforcer.ledger.read_locked.side_effect = RuntimeError("disk full")

        journal = FakeJournal()
        command = FakeTradeCommand(intent_type="OPEN_LONG")
        trader = FakeTrader()
        strategy = make_strategy()

        # Must not raise
        result = await enforcer.precheck_entry_allocation(
            command=command,  # type: ignore[arg-type]
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        assert result.allowed is False
        assert result.reason == "ALLOCATOR_ENFORCE_ERROR"

        # Should have FAILED journal event
        failed_events = [
            e
            for e in journal.events
            if e[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_FAILED"
        ]
        assert len(failed_events) >= 1

    async def test_commit_exception_swallowed(self) -> None:
        """14. ledger.update_locked raises: does not raise, journals COMMIT_FAILED."""
        from src.live.portfolio_allocator_enforcer import (
            PortfolioAllocatorEnforceConfig,
            PortfolioAllocatorEnforcer,
            PortfolioAllocatorPrecheckResult,
        )
        config = PortfolioAllocatorEnforceConfig(enabled=True)
        enforcer = PortfolioAllocatorEnforcer.from_config(config)
        enforcer.ledger = MagicMock()
        enforcer.ledger.update_locked.side_effect = RuntimeError("write failed")

        precheck = PortfolioAllocatorPrecheckResult(
            enabled=True,
            allowed=True,
            reason="ALLOCATOR_ENFORCE_ALLOWED",
            projected_snapshot=default_snapshot(),
        )

        journal = FakeJournal()
        live_result = LiveTradeResult(
            ok=True,
            action="OPEN_LONG",
            order_id="ord-1",
            tp_order_id="tp-1",
            contracts="10",
            tp_price="101",
            message="ok",
            entry_filled=True,
            tp_ok=True,
        )

        # Must not raise
        await enforcer.commit_projected_snapshot_after_fill(
            precheck_result=precheck,
            live_result=live_result,
            journal=journal,  # type: ignore[arg-type]
            position_id="pos-1",
        )

        # Should have COMMIT_FAILED journal event
        failed_events = [
            e
            for e in journal.events
            if e[0] == "PORTFOLIO_ALLOCATOR_ENFORCE_COMMIT_FAILED"
        ]
        assert len(failed_events) == 1


if __name__ == "__main__":
    unittest.main()
