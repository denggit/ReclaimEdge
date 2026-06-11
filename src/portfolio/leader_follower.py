# -*- coding: utf-8 -*-
"""
G03: Leader/Follower Permission —— 动态限制 follower 最大层数、加仓距离、冻结时间。

纯逻辑模块，无 IO / 网络 / 环境变量 / live runtime 依赖。

职责:
  - 判断当前谁是 leader（第一个进入压力状态的 symbol）
  - 为每个 symbol 分配 LEADER / FOLLOWER / NEUTRAL 角色
  - 根据 leader 层数动态生成 follower 的 permission 限制
  - apply_permission_overlay 将 permission 覆写到 SymbolCapitalState

不负责:
  - 写入 CapitalLedger
  - 下单 / OKX 请求 / 邮件发送 / 策略信号判断
  - live path 接入（G04 才会接入 allocator dry-run）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from src.portfolio.capital_ledger import CapitalLedgerSnapshot, SymbolCapitalState

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LeaderFollowerError(ValueError):
    """LeaderFollower 模块基础异常。"""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SymbolRole = Literal["LEADER", "FOLLOWER", "NEUTRAL"]

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolPermission:
    """单个 symbol 的 leader/follower permission 快照。"""

    inst_id: str
    role: SymbolRole
    leader_symbol: str | None
    leader_used_layers: int
    permission_max_layers: int
    add_gap_multiplier: str
    add_freeze_multiplier: str
    no_new_entry: bool
    no_add_layer: bool
    no_new_sidecar_leg: bool
    reason: str


@dataclass(frozen=True)
class LeaderFollowerPermissions:
    """一次完整的 leader/follower permission 解析结果。"""

    leader_symbol: str | None
    permissions: Mapping[str, SymbolPermission]

    def permission_for(self, inst_id: str) -> SymbolPermission:
        """Return the ``SymbolPermission`` for *inst_id*.

        Raises ``LeaderFollowerError`` if *inst_id* is not present.
        """
        perm = self.permissions.get(inst_id)
        if perm is None:
            raise LeaderFollowerError(
                f"symbol '{inst_id}' not found in permissions"
            )
        return perm


# ---------------------------------------------------------------------------
# Active symbol helper
# ---------------------------------------------------------------------------


def is_active_symbol_state(state: SymbolCapitalState) -> bool:
    """Return True if *state* represents an active (non-flat, in-use) symbol.

    A symbol is active when:
      - ``state`` is not ``"FLAT"`` (case-insensitive)
      - ``used_layers > 0``
    """
    return state.state.upper() != "FLAT" and state.used_layers > 0


# ---------------------------------------------------------------------------
# Leader resolution
# ---------------------------------------------------------------------------


def resolve_leader_symbol(snapshot: CapitalLedgerSnapshot) -> str | None:
    """Determine the current leader symbol from *snapshot*.

    Rules (in order):

    1. **Sticky leader**: If ``snapshot.leader_symbol`` is set AND that symbol
       is still active with ``used_layers >= 3``, keep it.
    2. **New leader**: Otherwise, scan all active symbols for any with
       ``used_layers >= 3``.  Pick the one with the highest ``used_layers``.
       On tie, the first symbol in ``snapshot.symbols`` iteration order wins.
    3. **No leader**: If no active symbol has ``used_layers >= 3``, return
       ``None``.

    Returns
    -------
    str or None
        The ``inst_id`` of the leader, or ``None`` if no leader exists.
    """
    # 1. Sticky leader check
    if snapshot.leader_symbol is not None:
        leader_state = snapshot.symbols.get(snapshot.leader_symbol)
        if (
            leader_state is not None
            and is_active_symbol_state(leader_state)
            and leader_state.used_layers >= 3
        ):
            return snapshot.leader_symbol

    # 2. Scan for new leader candidate
    best_symbol: str | None = None
    best_layers = 0

    for inst_id, state in snapshot.symbols.items():
        if is_active_symbol_state(state) and state.used_layers >= 3:
            if state.used_layers > best_layers:
                best_layers = state.used_layers
                best_symbol = inst_id

    return best_symbol


# ---------------------------------------------------------------------------
# Build permissions
# ---------------------------------------------------------------------------


def build_leader_follower_permissions(
    snapshot: CapitalLedgerSnapshot,
) -> LeaderFollowerPermissions:
    """Build leader/follower permissions for every symbol in *snapshot*.

    Returns a ``LeaderFollowerPermissions`` containing a ``SymbolPermission``
    for each symbol in ``snapshot.symbols``.
    """
    leader_symbol = resolve_leader_symbol(snapshot)
    leader_state = (
        snapshot.symbols[leader_symbol] if leader_symbol is not None else None
    )

    permissions: dict[str, SymbolPermission] = {}

    for inst_id, state in snapshot.symbols.items():
        if leader_symbol is None:
            # -- No leader → every symbol is NEUTRAL -------------------------
            permissions[inst_id] = SymbolPermission(
                inst_id=inst_id,
                role="NEUTRAL",
                leader_symbol=None,
                leader_used_layers=0,
                permission_max_layers=state.plan_max_layers,
                add_gap_multiplier="1.0",
                add_freeze_multiplier="1.0",
                no_new_entry=False,
                no_add_layer=False,
                no_new_sidecar_leg=False,
                reason="NO_PRESSURE_LEADER",
            )
        elif inst_id == leader_symbol:
            # -- Leader -------------------------------------------------------
            permissions[inst_id] = SymbolPermission(
                inst_id=inst_id,
                role="LEADER",
                leader_symbol=leader_symbol,
                leader_used_layers=leader_state.used_layers,
                permission_max_layers=state.plan_max_layers,
                add_gap_multiplier="1.0",
                add_freeze_multiplier="1.0",
                no_new_entry=False,
                no_add_layer=False,
                no_new_sidecar_leg=False,
                reason="ACTIVE_LEADER",
            )
        else:
            # -- Follower — restrictions depend on leader layer count --------
            leader_layers = leader_state.used_layers
            if leader_layers == 3:
                permissions[inst_id] = _follower_permission(
                    inst_id=inst_id,
                    leader_symbol=leader_symbol,
                    leader_used_layers=3,
                    state=state,
                    cap=5,
                    gap="1.5",
                    freeze="1.5",
                    no_new_entry=False,
                    no_add_layer=False,
                    no_new_sidecar_leg=False,
                    reason="LEADER_LAYER_3_FOLLOWER_CAUTION",
                )
            elif leader_layers == 4:
                permissions[inst_id] = _follower_permission(
                    inst_id=inst_id,
                    leader_symbol=leader_symbol,
                    leader_used_layers=4,
                    state=state,
                    cap=4,
                    gap="2.0",
                    freeze="2.0",
                    no_new_entry=False,
                    no_add_layer=False,
                    no_new_sidecar_leg=False,
                    reason="LEADER_LAYER_4_FOLLOWER_DEFENSIVE",
                )
            else:  # leader_layers >= 5
                permissions[inst_id] = _follower_permission(
                    inst_id=inst_id,
                    leader_symbol=leader_symbol,
                    leader_used_layers=leader_layers,
                    state=state,
                    cap=state.used_layers,  # freeze at current layers
                    gap="2.0",
                    freeze="2.0",
                    no_new_entry=True,
                    no_add_layer=True,
                    no_new_sidecar_leg=True,
                    reason="LEADER_LAYER_5_PLUS_FOLLOWER_NO_NEW_RISK",
                )

    return LeaderFollowerPermissions(
        leader_symbol=leader_symbol,
        permissions=permissions,
    )


def _follower_permission(
    *,
    inst_id: str,
    leader_symbol: str,
    leader_used_layers: int,
    state: SymbolCapitalState,
    cap: int,
    gap: str,
    freeze: str,
    no_new_entry: bool,
    no_add_layer: bool,
    no_new_sidecar_leg: bool,
    reason: str,
) -> SymbolPermission:
    """Build a ``SymbolPermission`` for a follower symbol.

    ``cap`` is the absolute upper bound for ``permission_max_layers``; the
    actual value is ``min(state.plan_max_layers, cap)``.
    """
    return SymbolPermission(
        inst_id=inst_id,
        role="FOLLOWER",
        leader_symbol=leader_symbol,
        leader_used_layers=leader_used_layers,
        permission_max_layers=min(state.plan_max_layers, cap),
        add_gap_multiplier=gap,
        add_freeze_multiplier=freeze,
        no_new_entry=no_new_entry,
        no_add_layer=no_add_layer,
        no_new_sidecar_leg=no_new_sidecar_leg,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Permission overlay helper
# ---------------------------------------------------------------------------


def apply_permission_overlay(
    state: SymbolCapitalState,
    permission: SymbolPermission,
) -> SymbolCapitalState:
    """Return a new ``SymbolCapitalState`` with permission fields overlaid.

    Only these three fields are updated from *permission*:
      - ``permission_max_layers``
      - ``add_gap_multiplier``
      - ``add_freeze_multiplier``

    All other fields are copied verbatim from *state*.
    """
    return SymbolCapitalState(
        state=state.state,
        side=state.side,
        used_layers=state.used_layers,
        position_plan_id=state.position_plan_id,
        planned_main_contracts=state.planned_main_contracts,
        base_main_contracts=state.base_main_contracts,
        plan_max_layers=state.plan_max_layers,
        permission_max_layers=permission.permission_max_layers,
        add_gap_multiplier=permission.add_gap_multiplier,
        add_freeze_multiplier=permission.add_freeze_multiplier,
        main_used_margin_usdt=state.main_used_margin_usdt,
        sidecar_enabled=state.sidecar_enabled,
        sidecar_used_margin_usdt=state.sidecar_used_margin_usdt,
    )
