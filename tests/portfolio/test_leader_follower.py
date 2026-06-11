# -*- coding: utf-8 -*-
"""Tests for G03: Leader/Follower Permission logic."""

from __future__ import annotations

import pytest

from src.portfolio.capital_ledger import CapitalLedgerSnapshot, SymbolCapitalState
from src.portfolio.leader_follower import (
    LeaderFollowerError,
    LeaderFollowerPermissions,
    SymbolPermission,
    apply_permission_overlay,
    build_leader_follower_permissions,
    is_active_symbol_state,
    resolve_leader_symbol,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_ETH = "ETH-USDT-SWAP"
_BTC = "BTC-USDT-SWAP"


def _state(**overrides) -> SymbolCapitalState:
    """Build a ``SymbolCapitalState`` with sensible test defaults."""
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
    """Build a ``CapitalLedgerSnapshot`` with test defaults."""
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


# ---------------------------------------------------------------------------
# is_active_symbol_state
# ---------------------------------------------------------------------------


class TestIsActiveSymbolState:
    def test_flat_with_zero_layers_is_inactive(self):
        s = _state(state="FLAT", used_layers=0)
        assert not is_active_symbol_state(s)

    def test_open_with_zero_layers_is_inactive(self):
        s = _state(state="OPEN", used_layers=0)
        assert not is_active_symbol_state(s)

    def test_flat_with_layers_is_inactive(self):
        s = _state(state="FLAT", used_layers=3)
        assert not is_active_symbol_state(s)

    def test_open_with_layers_is_active(self):
        s = _state(state="OPEN", used_layers=2)
        assert is_active_symbol_state(s)

    def test_case_insensitive_flat(self):
        s = _state(state="flat", used_layers=1)
        assert not is_active_symbol_state(s)

    def test_unknown_state_with_layers_is_active(self):
        s = _state(state="RECLAIMING", used_layers=1)
        assert is_active_symbol_state(s)


# ---------------------------------------------------------------------------
# resolve_leader_symbol
# ---------------------------------------------------------------------------


class TestResolveLeaderSymbol:
    def test_no_active_symbols(self):
        snap = _snapshot()
        assert resolve_leader_symbol(snap) is None

    def test_active_but_no_pressure(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=1, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=2),
            },
        )
        assert resolve_leader_symbol(snap) is None

    def test_first_pressure_becomes_leader(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=1),
            },
        )
        assert resolve_leader_symbol(snap) == _ETH

    def test_sticky_leader_kept_even_if_other_higher(self):
        """Sticky rule: existing leader stays even if another symbol gets more layers."""
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=5),
            },
        )
        assert resolve_leader_symbol(snap) == _ETH

    def test_sticky_leader_released_when_flat(self):
        """When old leader goes flat, a new leader is selected."""
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="FLAT", used_layers=0, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=4),
            },
        )
        assert resolve_leader_symbol(snap) == _BTC

    def test_sticky_leader_released_when_below_3_layers(self):
        """When old leader drops below 3 layers, it's released."""
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="OPEN", used_layers=2, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=4),
            },
        )
        assert resolve_leader_symbol(snap) == _BTC

    def test_old_leader_flat_no_new_candidate(self):
        """Old leader flat, no symbol >= 3 → no leader."""
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="FLAT", used_layers=0, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=2),
            },
        )
        assert resolve_leader_symbol(snap) is None

    def test_new_leader_picks_highest_layers(self):
        """Without existing leader, pick the symbol with most used_layers."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=5),
            },
        )
        assert resolve_leader_symbol(snap) == _BTC

    def test_tie_break_uses_dict_order_eth_first(self):
        """When tied, first symbol in dict iteration wins."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=3),
            },
        )
        assert resolve_leader_symbol(snap) == _ETH

    def test_tie_break_uses_dict_order_btc_first(self):
        """When tied, first symbol in dict iteration wins."""
        snap = _snapshot(
            symbols={
                _BTC: _state(state="OPEN", used_layers=3),
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
            },
        )
        assert resolve_leader_symbol(snap) == _BTC

    def test_leader_symbol_not_in_symbols_ignored(self):
        """If leader_symbol references a missing inst_id, it's ignored."""
        snap = _snapshot(
            leader_symbol="SOL-USDT-SWAP",
            symbols={
                _ETH: _state(state="OPEN", used_layers=4, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=1),
            },
        )
        assert resolve_leader_symbol(snap) == _ETH


# ---------------------------------------------------------------------------
# build_leader_follower_permissions
# ---------------------------------------------------------------------------


class TestBuildLeaderFollowerPermissions:
    # -- No leader scenarios --------------------------------------------------

    def test_no_active_symbols_all_neutral(self):
        snap = _snapshot()
        result = build_leader_follower_permissions(snap)

        assert result.leader_symbol is None

        eth = result.permission_for(_ETH)
        assert eth.role == "NEUTRAL"
        assert eth.permission_max_layers == 8  # plan_max_layers
        assert eth.add_gap_multiplier == "1.0"
        assert eth.add_freeze_multiplier == "1.0"
        assert eth.no_new_entry is False
        assert eth.no_add_layer is False
        assert eth.no_new_sidecar_leg is False
        assert eth.reason == "NO_PRESSURE_LEADER"

        btc = result.permission_for(_BTC)
        assert btc.role == "NEUTRAL"
        assert btc.permission_max_layers == 8

    def test_active_but_no_pressure_neutral(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=1, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=2),
            },
        )
        result = build_leader_follower_permissions(snap)

        assert result.leader_symbol is None
        assert result.permission_for(_ETH).role == "NEUTRAL"
        assert result.permission_for(_BTC).role == "NEUTRAL"

    # -- Leader scenarios -----------------------------------------------------

    def test_first_pressure_leader(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=1),
            },
        )
        result = build_leader_follower_permissions(snap)

        assert result.leader_symbol == _ETH

        eth = result.permission_for(_ETH)
        assert eth.role == "LEADER"
        assert eth.leader_used_layers == 3
        assert eth.permission_max_layers == 8  # plan_max_layers unchanged
        assert eth.add_gap_multiplier == "1.0"
        assert eth.add_freeze_multiplier == "1.0"
        assert eth.reason == "ACTIVE_LEADER"

        btc = result.permission_for(_BTC)
        assert btc.role == "FOLLOWER"
        assert btc.leader_symbol == _ETH
        assert btc.leader_used_layers == 3
        assert btc.permission_max_layers == 5  # min(8, 5)
        assert btc.add_gap_multiplier == "1.5"
        assert btc.add_freeze_multiplier == "1.5"
        assert btc.no_new_entry is False
        assert btc.reason == "LEADER_LAYER_3_FOLLOWER_CAUTION"

    def test_sticky_leader(self):
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=5),
            },
        )
        result = build_leader_follower_permissions(snap)

        assert result.leader_symbol == _ETH
        assert result.permission_for(_ETH).role == "LEADER"
        assert result.permission_for(_BTC).role == "FOLLOWER"

    def test_old_leader_flat_new_leader_selected(self):
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="FLAT", used_layers=0, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=4),
            },
        )
        result = build_leader_follower_permissions(snap)

        assert result.leader_symbol == _BTC
        assert result.permission_for(_BTC).role == "LEADER"

        # Flat ETH is still FOLLOWER when there's a leader
        eth = result.permission_for(_ETH)
        assert eth.role == "FOLLOWER"
        # leader layer 4 → follower cap = min(8, 4) = 4
        assert eth.permission_max_layers == 4
        assert eth.add_gap_multiplier == "2.0"
        assert eth.no_new_entry is False

    def test_old_leader_flat_no_new_candidate(self):
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="FLAT", used_layers=0, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=2),
            },
        )
        result = build_leader_follower_permissions(snap)

        assert result.leader_symbol is None
        assert result.permission_for(_ETH).role == "NEUTRAL"
        assert result.permission_for(_BTC).role == "NEUTRAL"

    # -- Follower restriction levels ------------------------------------------

    def test_leader_layer3_follower_caution(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=1, plan_max_layers=8),
            },
        )
        result = build_leader_follower_permissions(snap)

        btc = result.permission_for(_BTC)
        assert btc.role == "FOLLOWER"
        assert btc.permission_max_layers == 5
        assert btc.add_gap_multiplier == "1.5"
        assert btc.add_freeze_multiplier == "1.5"
        assert btc.no_new_entry is False
        assert btc.no_add_layer is False
        assert btc.no_new_sidecar_leg is False
        assert btc.reason == "LEADER_LAYER_3_FOLLOWER_CAUTION"

    def test_leader_layer4_follower_defensive(self):
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=4, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=1, plan_max_layers=8),
            },
        )
        result = build_leader_follower_permissions(snap)

        btc = result.permission_for(_BTC)
        assert btc.role == "FOLLOWER"
        assert btc.permission_max_layers == 4
        assert btc.add_gap_multiplier == "2.0"
        assert btc.add_freeze_multiplier == "2.0"
        assert btc.no_new_entry is False
        assert btc.no_add_layer is False
        assert btc.no_new_sidecar_leg is False
        assert btc.reason == "LEADER_LAYER_4_FOLLOWER_DEFENSIVE"

    def test_leader_layer5_follower_flat_no_new_risk(self):
        """Leader at 5 layers, follower flat → permission_max_layers=0, all blocked."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=5, sidecar_enabled=True),
                _BTC: _state(state="FLAT", used_layers=0, plan_max_layers=8),
            },
        )
        result = build_leader_follower_permissions(snap)

        btc = result.permission_for(_BTC)
        assert btc.role == "FOLLOWER"
        assert btc.permission_max_layers == 0  # min(8, 0)
        assert btc.add_gap_multiplier == "2.0"
        assert btc.add_freeze_multiplier == "2.0"
        assert btc.no_new_entry is True
        assert btc.no_add_layer is True
        assert btc.no_new_sidecar_leg is True
        assert btc.reason == "LEADER_LAYER_5_PLUS_FOLLOWER_NO_NEW_RISK"

    def test_leader_layer6_follower_open_no_new_risk(self):
        """Leader at 6 layers, follower has 2 layers → frozen at 2, no new risk."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=6, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=2, plan_max_layers=8),
            },
        )
        result = build_leader_follower_permissions(snap)

        btc = result.permission_for(_BTC)
        assert btc.role == "FOLLOWER"
        assert btc.permission_max_layers == 2  # min(8, 2)
        assert btc.no_new_entry is True
        assert btc.no_add_layer is True
        assert btc.no_new_sidecar_leg is True
        assert btc.reason == "LEADER_LAYER_5_PLUS_FOLLOWER_NO_NEW_RISK"

    def test_leader_layer5_leader_still_full_permissions(self):
        """Leader itself always has full permissions."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=5, plan_max_layers=8, sidecar_enabled=True),
                _BTC: _state(state="FLAT", used_layers=0),
            },
        )
        result = build_leader_follower_permissions(snap)

        eth = result.permission_for(_ETH)
        assert eth.role == "LEADER"
        assert eth.permission_max_layers == 8
        assert eth.add_gap_multiplier == "1.0"
        assert eth.add_freeze_multiplier == "1.0"
        assert eth.no_new_entry is False
        assert eth.no_add_layer is False
        assert eth.no_new_sidecar_leg is False

    # -- plan_max_layers not hardcoded ----------------------------------------

    def test_permission_respects_plan_max_layers_3(self):
        """follower plan_max_layers=3, leader layer3 → permission=3 (not 5)."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=0, plan_max_layers=3),
            },
        )
        result = build_leader_follower_permissions(snap)
        btc = result.permission_for(_BTC)
        assert btc.permission_max_layers == 3  # min(3, 5) = 3

    def test_permission_respects_plan_max_layers_10_leader_3(self):
        """follower plan_max_layers=10, leader layer3 → permission=5."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=0, plan_max_layers=10),
            },
        )
        result = build_leader_follower_permissions(snap)
        btc = result.permission_for(_BTC)
        assert btc.permission_max_layers == 5  # min(10, 5) = 5

    def test_permission_respects_plan_max_layers_10_leader_4(self):
        """follower plan_max_layers=10, leader layer4 → permission=4."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=4, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=0, plan_max_layers=10),
            },
        )
        result = build_leader_follower_permissions(snap)
        btc = result.permission_for(_BTC)
        assert btc.permission_max_layers == 4  # min(10, 4) = 4

    def test_permission_respects_plan_max_layers_6_leader_3(self):
        """follower plan_max_layers=6, leader layer3 → permission=5."""
        snap = _snapshot(
            symbols={
                _ETH: _state(state="OPEN", used_layers=3, sidecar_enabled=True),
                _BTC: _state(state="OPEN", used_layers=0, plan_max_layers=6),
            },
        )
        result = build_leader_follower_permissions(snap)
        btc = result.permission_for(_BTC)
        assert btc.permission_max_layers == 5  # min(6, 5) = 5

    # -- permission_for error -------------------------------------------------

    def test_permission_for_missing_symbol_raises_error(self):
        snap = _snapshot()
        result = build_leader_follower_permissions(snap)

        with pytest.raises(LeaderFollowerError, match="SOL-USDT-SWAP"):
            result.permission_for("SOL-USDT-SWAP")

    # -- every symbol gets a permission entry --------------------------------

    def test_all_symbols_have_permission_entries(self):
        snap = _snapshot(
            leader_symbol=_ETH,
            symbols={
                _ETH: _state(state="OPEN", used_layers=4, sidecar_enabled=True),
                _BTC: _state(state="FLAT", used_layers=0),
            },
        )
        result = build_leader_follower_permissions(snap)

        assert _ETH in result.permissions
        assert _BTC in result.permissions
        assert len(result.permissions) == 2


# ---------------------------------------------------------------------------
# apply_permission_overlay
# ---------------------------------------------------------------------------


class TestApplyPermissionOverlay:
    def test_only_permission_fields_are_updated(self):
        original = _state(
            state="OPEN",
            side="LONG",
            used_layers=2,
            position_plan_id="plan-123",
            planned_main_contracts=("100", "115", "130"),
            base_main_contracts="100",
            plan_max_layers=10,
            permission_max_layers=10,
            add_gap_multiplier="1.0",
            add_freeze_multiplier="1.0",
            main_used_margin_usdt="5000.0",
            sidecar_enabled=True,
            sidecar_used_margin_usdt="1000.0",
        )

        perm = SymbolPermission(
            inst_id=_ETH,
            role="FOLLOWER",
            leader_symbol=_ETH,
            leader_used_layers=3,
            permission_max_layers=5,
            add_gap_multiplier="1.5",
            add_freeze_multiplier="1.5",
            no_new_entry=False,
            no_add_layer=False,
            no_new_sidecar_leg=False,
            reason="LEADER_LAYER_3_FOLLOWER_CAUTION",
        )

        result = apply_permission_overlay(original, perm)

        # Updated fields
        assert result.permission_max_layers == 5
        assert result.add_gap_multiplier == "1.5"
        assert result.add_freeze_multiplier == "1.5"

        # Unchanged fields
        assert result.state == "OPEN"
        assert result.side == "LONG"
        assert result.used_layers == 2
        assert result.position_plan_id == "plan-123"
        assert result.planned_main_contracts == ("100", "115", "130")
        assert result.base_main_contracts == "100"
        assert result.plan_max_layers == 10
        assert result.main_used_margin_usdt == "5000.0"
        assert result.sidecar_enabled is True
        assert result.sidecar_used_margin_usdt == "1000.0"

    def test_overlay_preserves_frozen_dataclass_behavior(self):
        """Result is still a frozen dataclass."""
        original = _state()
        perm = SymbolPermission(
            inst_id=_ETH,
            role="NEUTRAL",
            leader_symbol=None,
            leader_used_layers=0,
            permission_max_layers=8,
            add_gap_multiplier="1.0",
            add_freeze_multiplier="1.0",
            no_new_entry=False,
            no_add_layer=False,
            no_new_sidecar_leg=False,
            reason="NO_PRESSURE_LEADER",
        )
        result = apply_permission_overlay(original, perm)
        assert isinstance(result, SymbolCapitalState)
        # frozen dataclass: cannot set attributes
        with pytest.raises(Exception):
            result.permission_max_layers = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LeaderFollowerPermissions dataclass
# ---------------------------------------------------------------------------


class TestLeaderFollowerPermissionsDataclass:
    def test_permission_for_returns_correct_permission(self):
        perm = SymbolPermission(
            inst_id=_ETH,
            role="LEADER",
            leader_symbol=_ETH,
            leader_used_layers=3,
            permission_max_layers=8,
            add_gap_multiplier="1.0",
            add_freeze_multiplier="1.0",
            no_new_entry=False,
            no_add_layer=False,
            no_new_sidecar_leg=False,
            reason="ACTIVE_LEADER",
        )
        lfp = LeaderFollowerPermissions(
            leader_symbol=_ETH,
            permissions={_ETH: perm},
        )
        assert lfp.permission_for(_ETH) is perm

    def test_permission_for_missing_raises(self):
        lfp = LeaderFollowerPermissions(
            leader_symbol=None,
            permissions={},
        )
        with pytest.raises(LeaderFollowerError, match="SOL-USDT-SWAP"):
            lfp.permission_for("SOL-USDT-SWAP")


# ---------------------------------------------------------------------------
# SymbolPermission dataclass is frozen
# ---------------------------------------------------------------------------


class TestSymbolPermissionFrozen:
    def test_cannot_mutate(self):
        perm = SymbolPermission(
            inst_id=_ETH,
            role="NEUTRAL",
            leader_symbol=None,
            leader_used_layers=0,
            permission_max_layers=8,
            add_gap_multiplier="1.0",
            add_freeze_multiplier="1.0",
            no_new_entry=False,
            no_add_layer=False,
            no_new_sidecar_leg=False,
            reason="NO_PRESSURE_LEADER",
        )
        with pytest.raises(Exception):
            perm.permission_max_layers = 99  # type: ignore[misc]
