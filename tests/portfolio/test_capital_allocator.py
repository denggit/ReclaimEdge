# -*- coding: utf-8 -*-
"""Tests for G04: CapitalAllocator dry-run checker."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.portfolio.capital_allocator import (
    AllocationAction,
    AllocationCheckRequest,
    AllocationDecision,
    CapitalAllocatorError,
    check_allocation_dry_run,
    decimal_from_string,
    is_exit_or_reduce_action,
    is_new_risk_action,
    total_main_used_margin_usdt,
    would_exceed_main_cap,
)
from src.portfolio.capital_ledger import CapitalLedgerSnapshot, SymbolCapitalState
from src.portfolio.position_plan import PositionPlan, create_main_position_plan

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_ETH = "ETH-USDT-SWAP"
_BTC = "BTC-USDT-SWAP"


def _state(**overrides) -> SymbolCapitalState:
    """Build a SymbolCapitalState with sensible test defaults."""
    defaults = dict(
        state="FLAT",
        side=None,
        used_layers=0,
        position_plan_id=None,
        planned_main_contracts=(),
        base_main_contracts="0",
        plan_max_layers=8,
        permission_max_layers=8,
        add_gap_multiplier="1.0",
        add_freeze_multiplier="1.0",
        main_used_margin_usdt="0",
        sidecar_enabled=False,
        sidecar_used_margin_usdt="0",
    )
    defaults.update(overrides)
    return SymbolCapitalState(**defaults)


def _snapshot(
    *,
    leader_symbol: str | None = None,
    global_no_new_entry: bool = False,
    symbols: dict[str, SymbolCapitalState] | None = None,
) -> CapitalLedgerSnapshot:
    """Build a CapitalLedgerSnapshot with test defaults."""
    if symbols is None:
        symbols = {
            _ETH: _state(sidecar_enabled=True),
            _BTC: _state(),
        }
    return CapitalLedgerSnapshot(
        version=1,
        updated_ms=0,
        leader_symbol=leader_symbol,
        global_no_new_entry=global_no_new_entry,
        symbols=symbols,
    )


def _make_plan(
    inst_id: str = _ETH,
    side: str = "LONG",
    max_layers: int = 8,
    base_main_contracts: str = "100",
) -> PositionPlan:
    """Create a test PositionPlan."""
    return create_main_position_plan(
        inst_id=inst_id,
        side=side,
        base_main_contracts=base_main_contracts,
        max_layers=max_layers,
        layer_multiplier_step="0.15",
        contract_precision="0.1",
        min_contracts="0.1",
    )


def _request(
    inst_id: str = _ETH,
    action: AllocationAction = "OPEN_MAIN",
    side: str | None = "LONG",
    requested_layer: int | None = 1,
    position_plan: PositionPlan | None = None,
    main_margin_delta_usdt: str = "0",
    sidecar_margin_delta_usdt: str = "0",
    account_equity_usdt: str = "0",
    global_main_cap_pct: str = "0.70",
) -> AllocationCheckRequest:
    """Build an AllocationCheckRequest with sensible defaults."""
    return AllocationCheckRequest(
        inst_id=inst_id,
        action=action,
        side=side,
        requested_layer=requested_layer,
        position_plan=position_plan,
        main_margin_delta_usdt=main_margin_delta_usdt,
        sidecar_margin_delta_usdt=sidecar_margin_delta_usdt,
        account_equity_usdt=account_equity_usdt,
        global_main_cap_pct=global_main_cap_pct,
    )


# ===================================================================
# 1. Exit / reduce always allowed
# ===================================================================


class TestExitReduceAlwaysAllowed:
    """CLOSE_MAIN, REDUCE_MAIN, CLOSE_SIDECAR are always allowed."""

    @pytest.mark.parametrize("action", ["CLOSE_MAIN", "REDUCE_MAIN", "CLOSE_SIDECAR"])
    def test_allowed_even_with_global_no_new_entry(self, action: AllocationAction):
        snap = _snapshot(global_no_new_entry=True)
        req = _request(action=action, side=None, requested_layer=None)
        decision = check_allocation_dry_run(snapshot=snap, request=req)

        assert decision.allowed is True
        assert decision.reason == "EXIT_OR_REDUCE_ALWAYS_ALLOWED"
        assert decision.projected_snapshot is snap

    @pytest.mark.parametrize("action", ["CLOSE_MAIN", "REDUCE_MAIN", "CLOSE_SIDECAR"])
    def test_allowed_even_with_zero_equity(self, action: AllocationAction):
        snap = _snapshot()
        req = _request(
            action=action,
            side=None,
            requested_layer=None,
            account_equity_usdt="0",
        )
        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.reason == "EXIT_OR_REDUCE_ALWAYS_ALLOWED"
        assert decision.projected_snapshot is snap

    @pytest.mark.parametrize("action", ["CLOSE_MAIN", "REDUCE_MAIN", "CLOSE_SIDECAR"])
    def test_allowed_even_with_leader_layer_5(self, action: AllocationAction):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN", side="LONG", used_layers=5, sidecar_enabled=True
                ),
                _BTC: _state(),
            },
        )
        req = _request(action=action, side=None, requested_layer=None)
        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.reason == "EXIT_OR_REDUCE_ALWAYS_ALLOWED"
        assert decision.projected_snapshot is snap

    def test_projected_snapshot_is_original_for_exit(self):
        snap = _snapshot()
        req = _request(action="CLOSE_MAIN", side=None, requested_layer=None)
        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.projected_snapshot is snap


# ===================================================================
# 2. OPEN_MAIN allowed
# ===================================================================


class TestOpenMainAllowed:
    def test_eth_flat_open_main_allowed(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
            global_main_cap_pct="0.70",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)

        assert decision.allowed is True
        assert decision.reason == "OPEN_MAIN_ALLOWED"

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.state == "OPEN"
        assert projected_eth.side == "LONG"
        assert projected_eth.used_layers == 1
        assert projected_eth.position_plan_id == plan.plan_id
        assert projected_eth.planned_main_contracts == plan.planned_main_contracts
        assert projected_eth.plan_max_layers == 8
        assert projected_eth.main_used_margin_usdt == "10"

    def test_open_main_leader_symbol_is_none_when_only_layer_1(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.leader_symbol is None

    def test_open_main_base_main_contracts_matches_plan(self):
        plan = _make_plan(max_layers=8, base_main_contracts="150")
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.base_main_contracts == "150"

    def test_open_main_preserves_btc_state(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(used_layers=0),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.projected_snapshot.symbols[_BTC].used_layers == 0
        assert decision.projected_snapshot.symbols[_BTC].state == "FLAT"


# ===================================================================
# 3. OPEN_MAIN uses plan max_layers, not hardcoded
# ===================================================================


class TestOpenMainMaxLayersFromPlan:
    def test_plan_max_layers_10(self):
        plan = _make_plan(max_layers=10)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.plan_max_layers == 10
        assert len(projected_eth.planned_main_contracts) == 10

    def test_plan_max_layers_3(self):
        plan = _make_plan(max_layers=3)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.plan_max_layers == 3
        assert len(projected_eth.planned_main_contracts) == 3

    def test_plan_max_layers_5(self):
        plan = _make_plan(max_layers=5)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.plan_max_layers == 5
        assert len(projected_eth.planned_main_contracts) == 5


# ===================================================================
# 4. OPEN_MAIN rejected if symbol active
# ===================================================================


class TestOpenMainRejectedSymbolActive:
    def test_already_open_layer_1(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN", side="LONG", used_layers=1, sidecar_enabled=True
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "SYMBOL_ALREADY_ACTIVE"

    def test_already_open_layer_2(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN", side="LONG", used_layers=2, sidecar_enabled=True
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "SYMBOL_ALREADY_ACTIVE"


# ===================================================================
# 5. OPEN_MAIN rejected by global_no_new_entry
# ===================================================================


class TestOpenMainRejectedGlobalNoNewEntry:
    def test_global_no_new_entry_blocks_open_main(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(global_no_new_entry=True)
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "GLOBAL_NO_NEW_ENTRY"


# ===================================================================
# 6. OPEN_MAIN rejected by follower no_new_entry
# ===================================================================


class TestOpenMainRejectedByPermission:
    def test_follower_flat_blocked_by_no_new_entry(self):
        """Leader ETH at layer5 → follower BTC no_new_entry=True."""
        plan = _make_plan(inst_id=_BTC, max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN", side="LONG", used_layers=5, sidecar_enabled=True
                ),
                _BTC: _state(state="FLAT", used_layers=0),
            },
        )
        req = _request(
            inst_id=_BTC,
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "PERMISSION_NO_NEW_ENTRY"

    def test_follower_permission_layer_limit(self):
        """Leader ETH at layer5+ → follower BTC flat.
        permission_max_layers=0, no_new_entry=True.
        OPEN_MAIN with layer 1 is rejected (no_new_entry fires first)."""
        plan = _make_plan(inst_id=_BTC, max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN", side="LONG", used_layers=5, sidecar_enabled=True
                ),
                _BTC: _state(state="FLAT", used_layers=0),
            },
        )
        req = _request(
            inst_id=_BTC,
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        # no_new_entry fires before layer limit check
        assert decision.reason == "PERMISSION_NO_NEW_ENTRY"


# ===================================================================
# 7. OPEN_MAIN cap check
# ===================================================================


class TestOpenMainCapCheck:
    def test_exactly_at_cap_allowed(self):
        """account_equity=100, cap_pct=0.70 → limit=70.
        current_margin=60, delta=10 → total=70 (at limit, allowed)."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True, main_used_margin_usdt="60"),
                _BTC: _state(main_used_margin_usdt="0"),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="100",
            global_main_cap_pct="0.70",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.reason == "OPEN_MAIN_ALLOWED"

    def test_slightly_over_cap_rejected(self):
        """account_equity=100, cap_pct=0.70 → limit=70.
        current_margin=60, delta=10.01 → total=70.01 (> limit, rejected)."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True, main_used_margin_usdt="60"),
                _BTC: _state(main_used_margin_usdt="0"),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10.01",
            account_equity_usdt="100",
            global_main_cap_pct="0.70",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "MAIN_CAP_EXCEEDED"

    def test_cap_with_multi_symbol_margin(self):
        """ETH margin=30, BTC margin=20 (total=50).
        account_equity=100, cap=0.70 → limit=70.
        delta=25 → total=75 (>70, rejected)."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True, main_used_margin_usdt="30"),
                _BTC: _state(main_used_margin_usdt="20"),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="25",
            account_equity_usdt="100",
            global_main_cap_pct="0.70",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "MAIN_CAP_EXCEEDED"

    def test_cap_with_multi_symbol_margin_allowed(self):
        """ETH margin=30, BTC margin=20 (total=50).
        account_equity=100, cap=0.70 → limit=70.
        delta=20 → total=70 (at limit, allowed)."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True, main_used_margin_usdt="30"),
                _BTC: _state(main_used_margin_usdt="20"),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="20",
            account_equity_usdt="100",
            global_main_cap_pct="0.70",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.reason == "OPEN_MAIN_ALLOWED"


# ===================================================================
# 8. ADD_MAIN allowed from layer1 to layer2
# ===================================================================


class TestAddMainAllowed:
    def test_layer1_to_layer2(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    base_main_contracts="100",
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)

        assert decision.allowed is True
        assert decision.reason == "ADD_MAIN_ALLOWED"

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.used_layers == 2
        assert projected_eth.main_used_margin_usdt == "25"  # 10 + 15
        assert projected_eth.state == "OPEN"
        assert projected_eth.side == "LONG"

    def test_leader_still_none_after_layer2(self):
        """Layer 1→2 should NOT create a leader (needs ≥3 layers)."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.leader_symbol is None

    def test_permission_fields_updated_on_add_main(self):
        """ADD_MAIN should overlay permission fields on projected state."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        projected = decision.projected_snapshot.symbols[_ETH]
        assert projected.permission_max_layers == 8  # NEUTRAL: plan_max_layers
        assert projected.add_gap_multiplier == "1.0"
        assert projected.add_freeze_multiplier == "1.0"


# ===================================================================
# 9. ADD_MAIN layer2 -> layer3 creates projected leader
# ===================================================================


class TestAddMainCreatesLeader:
    def test_layer2_to_layer3_creates_leader(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=2,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="25",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=3,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.reason == "ADD_MAIN_ALLOWED"
        assert decision.leader_symbol == _ETH

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.used_layers == 3

    def test_sticky_leader_kept(self):
        """ETH is already leader at layer3, ADD_MAIN → layer4 keeps leader."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=3,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="40",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=4,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.leader_symbol == _ETH


# ===================================================================
# 10. ADD_MAIN rejected if not active
# ===================================================================


class TestAddMainRejectedNotActive:
    def test_flat_symbol_add_main_rejected(self):
        snap = _snapshot()
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "SYMBOL_NOT_ACTIVE"

    def test_missing_symbol_add_main_rejected(self):
        """Symbol not in snapshot at all."""
        snap = _snapshot(
            symbols={
                _BTC: _state(),
            },
        )
        req = _request(
            inst_id=_ETH,
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "SYMBOL_NOT_ACTIVE"


# ===================================================================
# 11. ADD_MAIN rejected side mismatch
# ===================================================================


class TestAddMainRejectedSideMismatch:
    def test_long_state_short_request(self):
        plan = _make_plan(max_layers=8, side="LONG")
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="SHORT",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "SIDE_MISMATCH"


# ===================================================================
# 12. ADD_MAIN rejected non-sequential
# ===================================================================


class TestAddMainRejectedNonSequential:
    def test_skip_layer_from_1_to_3(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=3,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "NON_SEQUENTIAL_LAYER"

    def test_skip_layer_from_2_to_5(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=2,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="25",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=5,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "NON_SEQUENTIAL_LAYER"


# ===================================================================
# 13. ADD_MAIN rejected missing plan
# ===================================================================


class TestAddMainRejectedMissingPlan:
    def test_no_position_plan_id(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=None,
                    planned_main_contracts=(),
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "MISSING_POSITION_PLAN"

    def test_empty_planned_main_contracts(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id="plan-xxx",
                    planned_main_contracts=(),
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "MISSING_POSITION_PLAN"

    def test_plan_max_layers_zero(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id="plan-xxx",
                    planned_main_contracts=("100",),
                    plan_max_layers=0,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "MISSING_POSITION_PLAN"


# ===================================================================
# 14. ADD_MAIN rejected if target exceeds plan
# ===================================================================


class TestAddMainRejectedExceedsPlan:
    def test_layer_4_requested_plan_only_3(self):
        plan = _make_plan(max_layers=3)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=3,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=3,
                    main_used_margin_usdt="40",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=4,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "REQUESTED_LAYER_EXCEEDS_PLAN"


# ===================================================================
# 15. ADD_MAIN rejected by permission layer limit
# ===================================================================


class TestAddMainRejectedPermissionLayerLimit:
    def test_follower_exceeds_permission_max_layers(self):
        """ETH leader layer3 → BTC follower permission_max_layers=5.
        BTC used_layers=5, request layer6 → rejected.
        Must set sticky leader_symbol so BTC is a follower, not leader."""
        plan = _make_plan(inst_id=_BTC, max_layers=8)
        snap = _snapshot(
            leader_symbol=_ETH,  # sticky leader
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=3,
                    main_used_margin_usdt="50",
                    sidecar_enabled=True,
                ),
                _BTC: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=5,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="60",
                ),
            },
        )
        req = _request(
            inst_id=_BTC,
            action="ADD_MAIN",
            side="LONG",
            requested_layer=6,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "PERMISSION_LAYER_LIMIT"


# ===================================================================
# 16. ADD_MAIN rejected by no_add_layer
# ===================================================================


class TestAddMainRejectedNoAddLayer:
    def test_follower_blocked_by_no_add_layer(self):
        """ETH leader layer5 → BTC follower no_add_layer=True.
        BTC used_layers=2, request layer3 → rejected."""
        plan = _make_plan(inst_id=_BTC, max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=5,
                    main_used_margin_usdt="80",
                    sidecar_enabled=True,
                ),
                _BTC: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=2,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="30",
                ),
            },
        )
        req = _request(
            inst_id=_BTC,
            action="ADD_MAIN",
            side="LONG",
            requested_layer=3,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "PERMISSION_NO_ADD_LAYER"

    def test_add_main_also_blocked_by_global_no_new_entry(self):
        """ADD_MAIN is new risk, blocked by global_no_new_entry."""
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            global_no_new_entry=True,
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=2,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "GLOBAL_NO_NEW_ENTRY"


# ===================================================================
# 17. OPEN_SIDECAR allowed
# ===================================================================


class TestOpenSidecarAllowed:
    def test_eth_sidecar_allowed_with_delta(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                    sidecar_used_margin_usdt="5",
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)

        assert decision.allowed is True
        assert decision.reason == "OPEN_SIDECAR_ALLOWED"

        projected_eth = decision.projected_snapshot.symbols[_ETH]
        assert projected_eth.sidecar_used_margin_usdt == "8"  # 5 + 3
        assert projected_eth.sidecar_enabled is True

    def test_sidecar_with_zero_delta_allowed(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    sidecar_enabled=True,
                    sidecar_used_margin_usdt="5",
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="0",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        assert decision.reason == "OPEN_SIDECAR_ALLOWED"

    def test_sidecar_other_fields_preserved(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=2,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                    sidecar_used_margin_usdt="5",
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True
        projected = decision.projected_snapshot.symbols[_ETH]
        assert projected.state == "OPEN"
        assert projected.side == "LONG"
        assert projected.used_layers == 2
        assert projected.main_used_margin_usdt == "10"


# ===================================================================
# 18. OPEN_SIDECAR rejected if disabled
# ===================================================================


class TestOpenSidecarRejectedDisabled:
    def test_btc_sidecar_disabled(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(sidecar_enabled=False),
            },
        )
        req = _request(
            inst_id=_BTC,
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "SIDECAR_DISABLED"

    def test_unknown_symbol_rejected(self):
        snap = _snapshot()
        req = _request(
            inst_id="SOL-USDT-SWAP",
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "UNKNOWN_SYMBOL"


# ===================================================================
# 19. OPEN_SIDECAR rejected by no_new_sidecar_leg
# ===================================================================


class TestOpenSidecarRejectedByPermission:
    def test_follower_blocked_by_no_new_sidecar_leg(self):
        """ETH leader layer5 → BTC follower no_new_sidecar_leg=True."""
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=5,
                    main_used_margin_usdt="80",
                    sidecar_enabled=True,
                ),
                _BTC: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=2,
                    main_used_margin_usdt="30",
                    sidecar_enabled=True,
                    sidecar_used_margin_usdt="0",
                ),
            },
        )
        req = _request(
            inst_id=_BTC,
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="5",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "PERMISSION_NO_NEW_SIDECAR_LEG"

    def test_sidecar_blocked_by_global_no_new_entry(self):
        snap = _snapshot(
            global_no_new_entry=True,
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "GLOBAL_NO_NEW_ENTRY"


# ===================================================================
# 20. Invalid inputs
# ===================================================================


class TestInvalidInputs:
    def test_open_main_requested_layer_not_1(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=2,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_OPEN_MAIN_REQUEST"

    def test_open_main_no_position_plan(self):
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=None,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_OPEN_MAIN_REQUEST"

    def test_open_main_plan_inst_id_mismatch(self):
        plan = _make_plan(inst_id=_ETH, max_layers=8)
        snap = _snapshot()
        req = _request(
            inst_id=_BTC,
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_OPEN_MAIN_REQUEST"

    def test_open_main_plan_side_mismatch(self):
        plan = _make_plan(side="LONG", max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="SHORT",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_OPEN_MAIN_REQUEST"

    def test_open_main_missing_side(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side=None,
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_OPEN_MAIN_REQUEST"

    def test_add_main_no_requested_layer(self):
        snap = _snapshot()
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=None,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_ADD_MAIN_REQUEST"

    def test_account_equity_zero_for_open_main(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="0",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_ACCOUNT_EQUITY"

    def test_account_equity_negative_for_open_main(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="-5",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_ACCOUNT_EQUITY"

    def test_global_main_cap_pct_negative(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
            global_main_cap_pct="-0.1",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_GLOBAL_MAIN_CAP_PCT"

    def test_global_main_cap_pct_greater_than_1(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
            global_main_cap_pct="1.5",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_GLOBAL_MAIN_CAP_PCT"

    def test_global_main_cap_pct_zero(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True, main_used_margin_usdt="0"),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
            global_main_cap_pct="0",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_GLOBAL_MAIN_CAP_PCT"

    def test_main_margin_delta_negative(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="-10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_MARGIN_DELTA"

    def test_sidecar_margin_delta_negative(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(),
            },
        )
        req = _request(
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="-3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_MARGIN_DELTA"

    def test_float_input_rejected(self):
        with pytest.raises(CapitalAllocatorError, match="float"):
            decimal_from_string(1.5, "test_field")  # type: ignore[arg-type]

    def test_unknown_action_rejected(self):
        snap = _snapshot()
        req = _request(
            action="UNKNOWN_ACTION",  # type: ignore[arg-type]
            side=None,
            requested_layer=None,
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "UNKNOWN_ACTION"

    def test_open_main_invalid_side_string(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot()
        req = _request(
            action="OPEN_MAIN",
            side="NEUTRAL",  # not LONG or SHORT
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_OPEN_MAIN_REQUEST"

    def test_add_main_negative_layer_rejected(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=-1,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False
        assert decision.reason == "INVALID_ADD_MAIN_REQUEST"


# ===================================================================
# 21. Projected snapshot does not mutate original
# ===================================================================


class TestSnapshotImmutability:
    def test_original_snapshot_unchanged_after_allowed_decision(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(),
            },
        )
        original_eth = snap.symbols[_ETH]
        original_btc = snap.symbols[_BTC]

        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True

        # Original snapshot unchanged
        assert snap.symbols[_ETH] is original_eth
        assert snap.symbols[_BTC] is original_btc
        assert snap.symbols[_ETH].state == "FLAT"
        assert snap.symbols[_ETH].used_layers == 0

        # Projected is a NEW object
        assert decision.projected_snapshot is not snap
        assert decision.projected_snapshot.symbols[_ETH] is not original_eth
        assert decision.projected_snapshot.symbols[_ETH].state == "OPEN"
        assert decision.projected_snapshot.symbols[_ETH].used_layers == 1

    def test_original_snapshot_unchanged_after_rejected_decision(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            global_no_new_entry=True,
            symbols={
                _ETH: _state(sidecar_enabled=True),
                _BTC: _state(),
            },
        )
        original_eth = snap.symbols[_ETH]

        req = _request(
            action="OPEN_MAIN",
            side="LONG",
            requested_layer=1,
            position_plan=plan,
            main_margin_delta_usdt="10",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is False

        # Original snapshot unchanged
        assert snap.symbols[_ETH] is original_eth
        assert snap.symbols[_ETH].state == "FLAT"

        # For rejected, projected is the same as original (by design for these tests)
        # but could also be a copy - verify at minimum it's not mutated
        assert decision.projected_snapshot.symbols[_ETH].state == "FLAT"

    def test_add_main_projected_does_not_mutate_original(self):
        plan = _make_plan(max_layers=8)
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=2,
                    position_plan_id=plan.plan_id,
                    planned_main_contracts=plan.planned_main_contracts,
                    plan_max_layers=8,
                    main_used_margin_usdt="25",
                    sidecar_enabled=True,
                ),
                _BTC: _state(),
            },
        )
        original_eth = snap.symbols[_ETH]

        req = _request(
            action="ADD_MAIN",
            side="LONG",
            requested_layer=3,
            main_margin_delta_usdt="15",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True

        # Original unchanged
        assert snap.symbols[_ETH] is original_eth
        assert snap.symbols[_ETH].used_layers == 2
        assert snap.symbols[_ETH].main_used_margin_usdt == "25"

        # Projected updated
        assert decision.projected_snapshot.symbols[_ETH].used_layers == 3
        assert decision.projected_snapshot.symbols[_ETH].main_used_margin_usdt == "40"

    def test_open_sidecar_projected_does_not_mutate_original(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(
                    state="OPEN",
                    side="LONG",
                    used_layers=1,
                    main_used_margin_usdt="10",
                    sidecar_enabled=True,
                    sidecar_used_margin_usdt="5",
                ),
                _BTC: _state(),
            },
        )
        original_eth = snap.symbols[_ETH]

        req = _request(
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            sidecar_margin_delta_usdt="3",
            account_equity_usdt="1000",
        )

        decision = check_allocation_dry_run(snapshot=snap, request=req)
        assert decision.allowed is True

        # Original unchanged
        assert snap.symbols[_ETH] is original_eth
        assert snap.symbols[_ETH].sidecar_used_margin_usdt == "5"

        # Projected updated
        assert decision.projected_snapshot.symbols[_ETH].sidecar_used_margin_usdt == "8"


# ===================================================================
# 22. Helper function tests
# ===================================================================


class TestHelpers:
    def test_is_new_risk_action(self):
        assert is_new_risk_action("OPEN_MAIN") is True
        assert is_new_risk_action("ADD_MAIN") is True
        assert is_new_risk_action("OPEN_SIDECAR") is True
        assert is_new_risk_action("CLOSE_MAIN") is False
        assert is_new_risk_action("REDUCE_MAIN") is False
        assert is_new_risk_action("CLOSE_SIDECAR") is False

    def test_is_exit_or_reduce_action(self):
        assert is_exit_or_reduce_action("CLOSE_MAIN") is True
        assert is_exit_or_reduce_action("REDUCE_MAIN") is True
        assert is_exit_or_reduce_action("CLOSE_SIDECAR") is True
        assert is_exit_or_reduce_action("OPEN_MAIN") is False
        assert is_exit_or_reduce_action("ADD_MAIN") is False
        assert is_exit_or_reduce_action("OPEN_SIDECAR") is False

    def test_total_main_used_margin_usdt(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(main_used_margin_usdt="30", sidecar_enabled=True),
                _BTC: _state(main_used_margin_usdt="20"),
            },
        )
        total = total_main_used_margin_usdt(snap)
        assert total == Decimal("50")

    def test_total_main_used_margin_usdt_zero(self):
        snap = _snapshot()
        total = total_main_used_margin_usdt(snap)
        assert total == Decimal("0")

    def test_would_exceed_main_cap_false(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(main_used_margin_usdt="60", sidecar_enabled=True),
                _BTC: _state(main_used_margin_usdt="0"),
            },
        )
        result = would_exceed_main_cap(
            snapshot=snap,
            margin_delta_usdt=Decimal("10"),
            account_equity_usdt=Decimal("100"),
            global_main_cap_pct=Decimal("0.70"),
        )
        assert result is False  # 60+10=70, limit=70 → not exceed

    def test_would_exceed_main_cap_true(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(main_used_margin_usdt="60", sidecar_enabled=True),
                _BTC: _state(main_used_margin_usdt="0"),
            },
        )
        result = would_exceed_main_cap(
            snapshot=snap,
            margin_delta_usdt=Decimal("10.01"),
            account_equity_usdt=Decimal("100"),
            global_main_cap_pct=Decimal("0.70"),
        )
        assert result is True  # 60+10.01=70.01 > 70

    def test_decimal_from_string_accepts_decimal(self):
        d = Decimal("123.45")
        result = decimal_from_string(d, "test")
        assert result is d

    def test_decimal_from_string_accepts_str(self):
        result = decimal_from_string("123.45", "test")
        assert result == Decimal("123.45")

    def test_decimal_from_string_rejects_float(self):
        with pytest.raises(CapitalAllocatorError, match="float"):
            decimal_from_string(1.5, "test_field")  # type: ignore[arg-type]

    def test_decimal_from_string_rejects_invalid_str(self):
        with pytest.raises(CapitalAllocatorError, match="not a valid decimal"):
            decimal_from_string("not-a-number", "test_field")


# ===================================================================
# 23. DTO immutability
# ===================================================================


class TestDtoImmutability:
    def test_allocation_check_request_is_frozen(self):
        req = _request(action="OPEN_MAIN")
        with pytest.raises(Exception):
            req.action = "CLOSE_MAIN"  # type: ignore[misc]

    def test_allocation_decision_is_frozen(self):
        snap = _snapshot()
        decision = AllocationDecision(
            allowed=True,
            reason="test",
            inst_id=_ETH,
            action="OPEN_MAIN",
            requested_layer=1,
            leader_symbol=None,
            permission=None,
            projected_snapshot=snap,
        )
        with pytest.raises(Exception):
            decision.allowed = False  # type: ignore[misc]


# ===================================================================
# 24. No forbidden imports / live dependencies
# ===================================================================


class TestSourcePurity:
    def test_capital_allocator_source_has_no_forbidden_imports(self):
        import ast
        from pathlib import Path

        source_path = (
            Path(__file__).parents[2]
            / "src" / "portfolio" / "capital_allocator.py"
        )
        source = source_path.read_text()

        forbidden = [
            "CapitalLedger.read_locked",
            "CapitalLedger.update_locked",
            "write_json_atomic",
            "read_json_or_none",
            "Trader",
            "Strategy",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "EmailSender",
            "os.getenv",
            "load_dotenv",
            "redis",
            "sqlite",
        ]
        for token in forbidden:
            assert token not in source, (
                f"capital_allocator.py must not import/use {token}"
            )

    def test_capital_allocator_does_not_import_capital_ledger_class(self):
        """G04 should not import CapitalLedger (the class, only the DTOs)."""
        import ast
        from pathlib import Path

        source_path = (
            Path(__file__).parents[2]
            / "src" / "portfolio" / "capital_allocator.py"
        )
        source = source_path.read_text()

        # It may import CapitalLedgerSnapshot and SymbolCapitalState, but NOT CapitalLedger the class
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "src.portfolio.capital_ledger":
                    for alias in node.names:
                        assert alias.name != "CapitalLedger", (
                            "capital_allocator.py must not import CapitalLedger class"
                        )
