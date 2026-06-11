# -*- coding: utf-8 -*-
"""
Portfolio infrastructure layer.

G01: 账户资金账本 CapitalLedger —— JSON 文件 + 文件锁。
G02: PositionPlan —— 开仓时生成完整 layer 计划。
G03: LeaderFollower —— leader/follower 动态权限。
G04: CapitalAllocator —— allocator dry-run checker（纯逻辑）。
"""

from src.portfolio.capital_allocator import (
    AllocationAction,
    AllocationCheckRequest,
    AllocationDecision,
    CapitalAllocatorError,
    check_allocation_dry_run,
    decimal_from_string,
    is_new_risk_action,
    total_main_used_margin_usdt,
    would_exceed_main_cap,
)

from src.portfolio.capital_ledger import (
    CapitalLedger,
    CapitalLedgerError,
    CapitalLedgerLockTimeout,
    CapitalLedgerSchemaError,
    CapitalLedgerSnapshot,
    SymbolCapitalState,
    default_snapshot,
    default_symbol_state,
)

from src.portfolio.leader_follower import (
    LeaderFollowerConfig,
    LeaderFollowerError,
    LeaderFollowerPermissions,
    LeaderMode,
    SymbolPermission,
    apply_permission_overlay,
    build_leader_follower_permissions,
    is_active_symbol_state,
    resolve_leader_symbol,
)

from src.portfolio.position_plan import (
    PositionPlan,
    PositionPlanError,
    create_main_position_plan,
)

__all__ = [
    # -- G01 --
    "CapitalLedger",
    "CapitalLedgerError",
    "CapitalLedgerLockTimeout",
    "CapitalLedgerSchemaError",
    "CapitalLedgerSnapshot",
    "SymbolCapitalState",
    "default_snapshot",
    "default_symbol_state",
    # -- G02 --
    "PositionPlan",
    "PositionPlanError",
    "create_main_position_plan",
    # -- G03 --
    "LeaderFollowerConfig",
    "LeaderFollowerError",
    "LeaderFollowerPermissions",
    "LeaderMode",
    "SymbolPermission",
    "apply_permission_overlay",
    "build_leader_follower_permissions",
    "is_active_symbol_state",
    "resolve_leader_symbol",
    # -- G04 --
    "AllocationAction",
    "AllocationCheckRequest",
    "AllocationDecision",
    "CapitalAllocatorError",
    "check_allocation_dry_run",
    "decimal_from_string",
    "is_new_risk_action",
    "total_main_used_margin_usdt",
    "would_exceed_main_cap",
]
