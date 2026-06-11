# -*- coding: utf-8 -*-
"""
Portfolio infrastructure layer.

G01: 账户资金账本 CapitalLedger —— JSON 文件 + 文件锁。
G02: PositionPlan —— 开仓时生成完整 layer 计划。
G03: LeaderFollower —— leader/follower 动态权限。
"""

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
    LeaderFollowerError,
    LeaderFollowerPermissions,
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
    "LeaderFollowerError",
    "LeaderFollowerPermissions",
    "SymbolPermission",
    "apply_permission_overlay",
    "build_leader_follower_permissions",
    "is_active_symbol_state",
    "resolve_leader_symbol",
]
