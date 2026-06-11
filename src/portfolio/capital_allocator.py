# -*- coding: utf-8 -*-
"""
G04: Allocator dry-run checker —— 仅判断，不接真实下单。

纯逻辑模块，无 env / 文件 IO / OKX / 邮件 / live runtime 依赖。

职责:
  - 接收 snapshot + request，返回 AllocationDecision
  - 判断新风险 action 是否允许（OPEN_MAIN / ADD_MAIN / OPEN_SIDECAR）
  - exit / reduce action 永远 allowed

不负责:
  - 读写 CapitalLedger 文件
  - 下单 / OKX 请求 / 邮件发送 / 策略信号判断
  - live path 接入
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Mapping

from src.portfolio.capital_ledger import CapitalLedgerSnapshot, SymbolCapitalState
from src.portfolio.leader_follower import (
    LeaderFollowerConfig,
    LeaderFollowerPermissions,
    SymbolPermission,
    apply_permission_overlay,
    build_leader_follower_permissions,
    resolve_leader_symbol,
)
from src.portfolio.position_plan import PositionPlan, decimal_to_plain_str

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CapitalAllocatorError(ValueError):
    """CapitalAllocator 模块基础异常。"""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AllocationAction = Literal[
    "OPEN_MAIN",
    "ADD_MAIN",
    "OPEN_SIDECAR",
    "CLOSE_MAIN",
    "REDUCE_MAIN",
    "CLOSE_SIDECAR",
]

_EXIT_REDUCE_ACTIONS: frozenset[AllocationAction] = frozenset({
    "CLOSE_MAIN",
    "REDUCE_MAIN",
    "CLOSE_SIDECAR",
})

_NEW_RISK_ACTIONS: frozenset[AllocationAction] = frozenset({
    "OPEN_MAIN",
    "ADD_MAIN",
    "OPEN_SIDECAR",
})

_VALID_SIDES: frozenset[str] = frozenset({"LONG", "SHORT"})

# Sentinel for _keep_leader marker
_KEEP_LEADER = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decimal_from_string(value: str | Decimal, field: str) -> Decimal:
    """Coerce *value* to Decimal, rejecting float for precision safety.

    Raises ``CapitalAllocatorError`` on float input or invalid decimal string.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        raise CapitalAllocatorError(
            f"{field}: float not accepted (use str or Decimal), got {value!r}"
        )
    if not isinstance(value, str):
        raise CapitalAllocatorError(
            f"{field} must be str or Decimal, got {type(value).__name__}: {value!r}"
        )
    try:
        return Decimal(value)
    except InvalidOperation:
        raise CapitalAllocatorError(
            f"{field} is not a valid decimal string: {value!r}"
        ) from None


def is_new_risk_action(action: AllocationAction) -> bool:
    """Return True for actions that increase risk exposure."""
    return action in _NEW_RISK_ACTIONS


def is_exit_or_reduce_action(action: AllocationAction) -> bool:
    """Return True for actions that reduce or exit risk."""
    return action in _EXIT_REDUCE_ACTIONS


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationCheckRequest:
    """A dry-run allocation check request."""

    inst_id: str
    action: AllocationAction
    side: str | None = None
    requested_layer: int | None = None
    requested_main_contracts: str | None = None
    position_plan: PositionPlan | None = None
    main_margin_delta_usdt: str = "0"
    sidecar_margin_delta_usdt: str = "0"
    account_equity_usdt: str = "0"
    global_main_cap_pct: str = "0.70"


@dataclass(frozen=True)
class AllocationDecision:
    """Result of a dry-run allocation check."""

    allowed: bool
    reason: str
    inst_id: str
    action: AllocationAction
    requested_layer: int | None
    leader_symbol: str | None
    permission: SymbolPermission | None
    projected_snapshot: CapitalLedgerSnapshot
    message: str = ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_allocation_dry_run(
    *,
    snapshot: CapitalLedgerSnapshot,
    request: AllocationCheckRequest,
    leader_follower_config: LeaderFollowerConfig | None = None,
) -> AllocationDecision:
    """Core entry point for G04: check an allocation intent against *snapshot*.

    Parameters
    ----------
    snapshot:
        Current capital ledger snapshot.
    request:
        Allocation check request.
    leader_follower_config:
        Leader/follower config.  ``None`` defaults to dynamic mode for backward
        compatibility.

    Returns an ``AllocationDecision``.  This function is pure — it never
    reads/writes files or calls external services.
    """
    cfg = leader_follower_config  # local alias for passing through

    # -- Exit / reduce: always allowed ----------------------------------------
    if is_exit_or_reduce_action(request.action):
        return _decision(
            allowed=True,
            reason="EXIT_OR_REDUCE_ALWAYS_ALLOWED",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Unknown action -------------------------------------------------------
    if request.action not in _NEW_RISK_ACTIONS:
        return _decision(
            allowed=False,
            reason="UNKNOWN_ACTION",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Check valid Decimal inputs for cap-related fields --------------------
    # Validate main_margin_delta_usdt
    main_delta = decimal_from_string(
        request.main_margin_delta_usdt, "main_margin_delta_usdt"
    )
    if main_delta < 0:
        return _decision(
            allowed=False,
            reason="INVALID_MARGIN_DELTA",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # Build permissions (needed for OPEN_MAIN, ADD_MAIN, OPEN_SIDECAR)
    permissions = build_leader_follower_permissions(snapshot, config=cfg)

    # -- Dispatch by action ---------------------------------------------------
    if request.action == "OPEN_MAIN":
        return _check_open_main(snapshot, request, main_delta, permissions, cfg)
    elif request.action == "ADD_MAIN":
        return _check_add_main(snapshot, request, main_delta, permissions, cfg)
    elif request.action == "OPEN_SIDECAR":
        return _check_open_sidecar(snapshot, request, permissions)

    # Should not reach here
    return _decision(
        allowed=False,
        reason="UNKNOWN_ACTION",
        snapshot=snapshot,
        request=request,
        permission=None,
    )


# ---------------------------------------------------------------------------
# OPEN_MAIN
# ---------------------------------------------------------------------------


def _check_open_main(
    snapshot: CapitalLedgerSnapshot,
    request: AllocationCheckRequest,
    main_delta: Decimal,
    permissions: LeaderFollowerPermissions,
    leader_follower_config: LeaderFollowerConfig | None = None,
) -> AllocationDecision:
    """Validate an OPEN_MAIN request."""
    inst_id = request.inst_id

    # -- Validate required fields ---------------------------------------------
    if request.side not in _VALID_SIDES:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    if request.requested_layer != 1:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    position_plan = request.position_plan
    if position_plan is None:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    if position_plan.inst_id != inst_id:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    if position_plan.side != request.side:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    if position_plan.max_layers < 1:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    if len(position_plan.planned_main_contracts) != position_plan.max_layers:
        return _decision(
            allowed=False,
            reason="INVALID_OPEN_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Current symbol must be flat / inactive -------------------------------
    current = snapshot.symbols.get(inst_id)
    if current is None or _is_active(current):
        return _decision(
            allowed=False,
            reason="SYMBOL_ALREADY_ACTIVE",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- global_no_new_entry --------------------------------------------------
    if snapshot.global_no_new_entry:
        return _decision(
            allowed=False,
            reason="GLOBAL_NO_NEW_ENTRY",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Permission check -----------------------------------------------------
    permission = permissions.permission_for(inst_id)

    if permission.no_new_entry:
        return _decision(
            allowed=False,
            reason="PERMISSION_NO_NEW_ENTRY",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if request.requested_layer > permission.permission_max_layers:
        return _decision(
            allowed=False,
            reason="PERMISSION_LAYER_LIMIT",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    # -- Main cap check -------------------------------------------------------
    equity = decimal_from_string(request.account_equity_usdt, "account_equity_usdt")
    cap_pct = decimal_from_string(request.global_main_cap_pct, "global_main_cap_pct")

    if equity <= 0:
        return _decision(
            allowed=False,
            reason="INVALID_ACCOUNT_EQUITY",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if cap_pct <= 0 or cap_pct > 1:
        return _decision(
            allowed=False,
            reason="INVALID_GLOBAL_MAIN_CAP_PCT",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if _would_exceed(snapshot, main_delta, equity, cap_pct):
        return _decision(
            allowed=False,
            reason="MAIN_CAP_EXCEEDED",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    # -- Build projected snapshot ---------------------------------------------
    base_state = current if current is not None else SymbolCapitalState()
    old_main_margin = decimal_from_string(
        base_state.main_used_margin_usdt, "main_used_margin_usdt"
    )
    new_main_margin = old_main_margin + main_delta

    new_state = SymbolCapitalState(
        state="OPEN",
        side=request.side,
        used_layers=1,
        position_plan_id=position_plan.plan_id,
        planned_main_contracts=position_plan.planned_main_contracts,
        base_main_contracts=position_plan.base_main_contracts,
        plan_max_layers=position_plan.max_layers,
        permission_max_layers=permission.permission_max_layers,
        add_gap_multiplier=permission.add_gap_multiplier,
        add_freeze_multiplier=permission.add_freeze_multiplier,
        main_used_margin_usdt=decimal_to_plain_str(new_main_margin),
        sidecar_enabled=base_state.sidecar_enabled,
        sidecar_used_margin_usdt=base_state.sidecar_used_margin_usdt,
    )

    projected = _replace_symbol_state(snapshot, inst_id, new_state)

    # Resolve leader on the projected snapshot (without sticky leader bias),
    # then sync it back into the projected snapshot.
    projected_leader = resolve_leader_symbol(projected, config=leader_follower_config)
    projected = _replace_leader_symbol(projected, projected_leader)

    return _decision(
        allowed=True,
        reason="OPEN_MAIN_ALLOWED",
        snapshot=projected,
        request=request,
        permission=permission,
        leader_symbol=projected_leader,
    )


# ---------------------------------------------------------------------------
# ADD_MAIN
# ---------------------------------------------------------------------------


def _check_add_main(
    snapshot: CapitalLedgerSnapshot,
    request: AllocationCheckRequest,
    main_delta: Decimal,
    permissions: LeaderFollowerPermissions,
    leader_follower_config: LeaderFollowerConfig | None = None,
) -> AllocationDecision:
    """Validate an ADD_MAIN request."""
    inst_id = request.inst_id

    # -- Validate required fields ---------------------------------------------
    if request.side not in _VALID_SIDES:
        return _decision(
            allowed=False,
            reason="INVALID_ADD_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    requested_layer = request.requested_layer
    if requested_layer is None or not isinstance(requested_layer, int):
        return _decision(
            allowed=False,
            reason="INVALID_ADD_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )
    if requested_layer < 1:
        return _decision(
            allowed=False,
            reason="INVALID_ADD_MAIN_REQUEST",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Current symbol must exist and be active ------------------------------
    current = snapshot.symbols.get(inst_id)
    if current is None or not _is_active(current):
        return _decision(
            allowed=False,
            reason="SYMBOL_NOT_ACTIVE",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Side must match ------------------------------------------------------
    if current.side != request.side:
        return _decision(
            allowed=False,
            reason="SIDE_MISMATCH",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Must be sequential layer ---------------------------------------------
    if requested_layer != current.used_layers + 1:
        return _decision(
            allowed=False,
            reason="NON_SEQUENTIAL_LAYER",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Must have a position plan --------------------------------------------
    if (
        current.position_plan_id is None
        or len(current.planned_main_contracts) == 0
        or current.plan_max_layers < 1
    ):
        return _decision(
            allowed=False,
            reason="MISSING_POSITION_PLAN",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    if requested_layer > current.plan_max_layers:
        return _decision(
            allowed=False,
            reason="REQUESTED_LAYER_EXCEEDS_PLAN",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- global_no_new_entry --------------------------------------------------
    if snapshot.global_no_new_entry:
        return _decision(
            allowed=False,
            reason="GLOBAL_NO_NEW_ENTRY",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Permission check -----------------------------------------------------
    permission = permissions.permission_for(inst_id)

    if permission.no_add_layer:
        return _decision(
            allowed=False,
            reason="PERMISSION_NO_ADD_LAYER",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if requested_layer > permission.permission_max_layers:
        return _decision(
            allowed=False,
            reason="PERMISSION_LAYER_LIMIT",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    # -- Planned contract quantity check -------------------------------------
    expected_contracts = _expected_main_contracts_for_layer(current, requested_layer)
    if expected_contracts is None:
        return _decision(
            allowed=False,
            reason="MISSING_EXPECTED_MAIN_CONTRACTS",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    try:
        requested_contracts = _requested_main_contracts_decimal(
            request.requested_main_contracts
        )
    except CapitalAllocatorError:
        return _decision(
            allowed=False,
            reason="INVALID_REQUESTED_MAIN_CONTRACTS",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if requested_contracts is None:
        return _decision(
            allowed=False,
            reason="MISSING_REQUESTED_MAIN_CONTRACTS",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if requested_contracts != expected_contracts:
        return _decision(
            allowed=False,
            reason="ADD_MAIN_CONTRACT_MISMATCH",
            snapshot=snapshot,
            request=request,
            permission=permission,
            message=(
                f"requested_main_contracts={requested_contracts} "
                f"expected_main_contracts={expected_contracts}"
            ),
        )

    # -- Main cap check -------------------------------------------------------
    equity = decimal_from_string(request.account_equity_usdt, "account_equity_usdt")
    cap_pct = decimal_from_string(request.global_main_cap_pct, "global_main_cap_pct")

    if equity <= 0:
        return _decision(
            allowed=False,
            reason="INVALID_ACCOUNT_EQUITY",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if cap_pct <= 0 or cap_pct > 1:
        return _decision(
            allowed=False,
            reason="INVALID_GLOBAL_MAIN_CAP_PCT",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    if _would_exceed(snapshot, main_delta, equity, cap_pct):
        return _decision(
            allowed=False,
            reason="MAIN_CAP_EXCEEDED",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    # -- Build projected snapshot ---------------------------------------------
    old_main_margin = decimal_from_string(
        current.main_used_margin_usdt, "main_used_margin_usdt"
    )
    new_main_margin = old_main_margin + main_delta

    new_state = SymbolCapitalState(
        state=current.state,
        side=current.side,
        used_layers=requested_layer,
        position_plan_id=current.position_plan_id,
        planned_main_contracts=current.planned_main_contracts,
        base_main_contracts=current.base_main_contracts,
        plan_max_layers=current.plan_max_layers,
        permission_max_layers=permission.permission_max_layers,
        add_gap_multiplier=permission.add_gap_multiplier,
        add_freeze_multiplier=permission.add_freeze_multiplier,
        main_used_margin_usdt=decimal_to_plain_str(new_main_margin),
        sidecar_enabled=current.sidecar_enabled,
        sidecar_used_margin_usdt=current.sidecar_used_margin_usdt,
    )

    projected = _replace_symbol_state(snapshot, inst_id, new_state)

    # Resolve leader on the projected snapshot, then sync it back.
    projected_leader = resolve_leader_symbol(projected, config=leader_follower_config)
    projected = _replace_leader_symbol(projected, projected_leader)

    return _decision(
        allowed=True,
        reason="ADD_MAIN_ALLOWED",
        snapshot=projected,
        request=request,
        permission=permission,
        leader_symbol=projected_leader,
    )


# ---------------------------------------------------------------------------
# OPEN_SIDECAR
# ---------------------------------------------------------------------------


def _check_open_sidecar(
    snapshot: CapitalLedgerSnapshot,
    request: AllocationCheckRequest,
    permissions: LeaderFollowerPermissions,
) -> AllocationDecision:
    """Validate an OPEN_SIDECAR request."""
    inst_id = request.inst_id

    # -- Symbol must exist ----------------------------------------------------
    current = snapshot.symbols.get(inst_id)
    if current is None:
        return _decision(
            allowed=False,
            reason="UNKNOWN_SYMBOL",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- sidecar_enabled ------------------------------------------------------
    if not current.sidecar_enabled:
        return _decision(
            allowed=False,
            reason="SIDECAR_DISABLED",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Margin delta validation ----------------------------------------------
    sidecar_delta = decimal_from_string(
        request.sidecar_margin_delta_usdt, "sidecar_margin_delta_usdt"
    )
    if sidecar_delta < 0:
        return _decision(
            allowed=False,
            reason="INVALID_MARGIN_DELTA",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- global_no_new_entry --------------------------------------------------
    if snapshot.global_no_new_entry:
        return _decision(
            allowed=False,
            reason="GLOBAL_NO_NEW_ENTRY",
            snapshot=snapshot,
            request=request,
            permission=None,
        )

    # -- Permission check -----------------------------------------------------
    permission = permissions.permission_for(inst_id)

    if permission.no_new_sidecar_leg:
        return _decision(
            allowed=False,
            reason="PERMISSION_NO_NEW_SIDECAR_LEG",
            snapshot=snapshot,
            request=request,
            permission=permission,
        )

    # -- Build projected snapshot ---------------------------------------------
    old_sidecar_margin = decimal_from_string(
        current.sidecar_used_margin_usdt, "sidecar_used_margin_usdt"
    )
    new_sidecar_margin = old_sidecar_margin + sidecar_delta

    new_state = SymbolCapitalState(
        state=current.state,
        side=current.side,
        used_layers=current.used_layers,
        position_plan_id=current.position_plan_id,
        planned_main_contracts=current.planned_main_contracts,
        base_main_contracts=current.base_main_contracts,
        plan_max_layers=current.plan_max_layers,
        permission_max_layers=permission.permission_max_layers,
        add_gap_multiplier=permission.add_gap_multiplier,
        add_freeze_multiplier=permission.add_freeze_multiplier,
        main_used_margin_usdt=current.main_used_margin_usdt,
        sidecar_enabled=current.sidecar_enabled,
        sidecar_used_margin_usdt=decimal_to_plain_str(new_sidecar_margin),
    )

    projected = _replace_symbol_state(snapshot, inst_id, new_state)

    return _decision(
        allowed=True,
        reason="OPEN_SIDECAR_ALLOWED",
        snapshot=projected,
        request=request,
        permission=permission,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_active(state: SymbolCapitalState) -> bool:
    """Return True if the symbol state represents an active position."""
    return state.state.upper() != "FLAT" and state.used_layers > 0


def total_main_used_margin_usdt(snapshot: CapitalLedgerSnapshot) -> Decimal:
    """Sum of main_used_margin_usdt across all symbols."""
    total = Decimal("0")
    for state in snapshot.symbols.values():
        total += decimal_from_string(state.main_used_margin_usdt, "main_used_margin_usdt")
    return total


def would_exceed_main_cap(
    *,
    snapshot: CapitalLedgerSnapshot,
    margin_delta_usdt: Decimal,
    account_equity_usdt: Decimal,
    global_main_cap_pct: Decimal,
) -> bool:
    """Return True if adding *margin_delta_usdt* would exceed the main cap."""
    current_total = total_main_used_margin_usdt(snapshot)
    cap_limit = account_equity_usdt * global_main_cap_pct
    return current_total + margin_delta_usdt > cap_limit


def _would_exceed(
    snapshot: CapitalLedgerSnapshot,
    margin_delta_usdt: Decimal,
    account_equity_usdt: Decimal,
    global_main_cap_pct: Decimal,
) -> bool:
    """Shorthand for cap check used internally."""
    return would_exceed_main_cap(
        snapshot=snapshot,
        margin_delta_usdt=margin_delta_usdt,
        account_equity_usdt=account_equity_usdt,
        global_main_cap_pct=global_main_cap_pct,
    )


def _expected_main_contracts_for_layer(
    state: SymbolCapitalState,
    requested_layer: int,
) -> Decimal | None:
    """Return planned main contracts for a 1-based layer, if present."""
    index = requested_layer - 1
    if not state.planned_main_contracts or index < 0:
        return None
    if index >= len(state.planned_main_contracts):
        return None
    return decimal_from_string(
        state.planned_main_contracts[index],
        "planned_main_contracts",
    )


def _requested_main_contracts_decimal(value: str | None) -> Decimal | None:
    """Parse requested main contracts from a request, treating empty as missing."""
    if value is None or value == "":
        return None
    return decimal_from_string(value, "requested_main_contracts")


def _replace_symbol_state(
    snapshot: CapitalLedgerSnapshot,
    inst_id: str,
    new_state: SymbolCapitalState,
    *,
    leader_symbol: str | None | object = _KEEP_LEADER,
) -> CapitalLedgerSnapshot:
    """Return a new snapshot with *inst_id* replaced by *new_state*.

    The original ``snapshot.symbols`` dict is never mutated.
    """
    new_symbols = dict(snapshot.symbols)
    new_symbols[inst_id] = new_state

    _leader = (
        snapshot.leader_symbol
        if leader_symbol is _KEEP_LEADER
        else leader_symbol
    )

    return CapitalLedgerSnapshot(
        version=snapshot.version,
        updated_ms=snapshot.updated_ms,
        leader_symbol=_leader,
        global_no_new_entry=snapshot.global_no_new_entry,
        symbols=new_symbols,
    )


def _replace_leader_symbol(
    snapshot: CapitalLedgerSnapshot,
    leader_symbol: str | None,
) -> CapitalLedgerSnapshot:
    """Return a new snapshot with *leader_symbol* replaced.

    All other fields are copied verbatim.  The original snapshot is never mutated.
    """
    return CapitalLedgerSnapshot(
        version=snapshot.version,
        updated_ms=snapshot.updated_ms,
        leader_symbol=leader_symbol,
        global_no_new_entry=snapshot.global_no_new_entry,
        symbols=snapshot.symbols,
    )


def _decision(
    *,
    allowed: bool,
    reason: str,
    snapshot: CapitalLedgerSnapshot,
    request: AllocationCheckRequest,
    permission: SymbolPermission | None,
    leader_symbol: str | None = None,
    message: str = "",
) -> AllocationDecision:
    """Construct an ``AllocationDecision`` with consistent defaults."""
    return AllocationDecision(
        allowed=allowed,
        reason=reason,
        inst_id=request.inst_id,
        action=request.action,
        requested_layer=request.requested_layer,
        leader_symbol=(
            leader_symbol
            if leader_symbol is not None
            else (snapshot.leader_symbol if allowed else None)
        ),
        permission=permission,
        projected_snapshot=snapshot if allowed else snapshot,
        message=message,
    )
