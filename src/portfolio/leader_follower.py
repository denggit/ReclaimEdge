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

G08c: 新增 fixed leader mode —— 固定 leader symbol，保护实盘 ETH。
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

LeaderMode = Literal["dynamic", "fixed"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaderFollowerConfig:
    """Immutable configuration for leader/follower mode.

    Attributes
    ----------
    leader_mode:
        ``"dynamic"`` — first symbol reaching layer3 becomes leader (current behaviour).
        ``"fixed"`` — *fixed_leader_symbol* is the only leader when active.
    fixed_leader_symbol:
        The inst_id of the fixed leader.  Required when *leader_mode* is
        ``"fixed"``; ignored (may be ``None``) for ``"dynamic"`` mode.
    """

    leader_mode: LeaderMode = "fixed"
    fixed_leader_symbol: str | None = "ETH-USDT-SWAP"

    _VALID_MODES: tuple[str, ...] = ("dynamic", "fixed")

    def __post_init__(self) -> None:
        if self.leader_mode not in self._VALID_MODES:
            raise LeaderFollowerError(
                f"leader_mode must be 'dynamic' or 'fixed', got {self.leader_mode!r}"
            )
        if self.leader_mode == "fixed":
            if not self.fixed_leader_symbol or not self.fixed_leader_symbol.strip():
                raise LeaderFollowerError(
                    "fixed_leader_symbol must be a non-empty string when "
                    "leader_mode='fixed'"
                )
            # Strip and re-set via object.__setattr__ (frozen dataclass)
            stripped = self.fixed_leader_symbol.strip()
            if stripped != self.fixed_leader_symbol:
                object.__setattr__(self, "fixed_leader_symbol", stripped)

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


def resolve_leader_symbol(
    snapshot: CapitalLedgerSnapshot,
    *,
    config: LeaderFollowerConfig | None = None,
) -> str | None:
    """Determine the current leader symbol from *snapshot*.

    **Dynamic mode** (``config is None`` or ``config.leader_mode == "dynamic"``):

    1. **Sticky leader**: If ``snapshot.leader_symbol`` is set AND that symbol
       is still active with ``used_layers >= 3``, keep it.
    2. **New leader**: Otherwise, scan all active symbols for any with
       ``used_layers >= 3``.  Pick the one with the highest ``used_layers``.
       On tie, the first symbol in ``snapshot.symbols`` iteration order wins.
    3. **No leader**: If no active symbol has ``used_layers >= 3``, return
       ``None``.

    **Fixed mode** (``config.leader_mode == "fixed"``):

    1. ``config.fixed_leader_symbol`` **must** exist in ``snapshot.symbols``,
       otherwise ``LeaderFollowerError`` is raised.
    2. If the fixed leader is active (``is_active_symbol_state``) **and**
       ``used_layers > 0``, return ``config.fixed_leader_symbol``.
    3. If the fixed leader is flat or ``used_layers == 0``, return ``None``
       (no pressure — the fixed leader does not restrict followers when flat).

    Returns
    -------
    str or None
        The ``inst_id`` of the leader, or ``None`` if no leader exists.

    Raises
    ------
    LeaderFollowerError
        If fixed mode and *fixed_leader_symbol* is not in *snapshot.symbols*.
    """
    cfg = config if config is not None else LeaderFollowerConfig(leader_mode="dynamic")

    # ── Fixed mode ──────────────────────────────────────────────────────────
    if cfg.leader_mode == "fixed":
        fixed_symbol = cfg.fixed_leader_symbol  # type: ignore[assignment]
        if fixed_symbol not in snapshot.symbols:
            raise LeaderFollowerError(
                f"fixed_leader_symbol '{fixed_symbol}' not found in "
                f"snapshot symbols: {list(snapshot.symbols)}"
            )
        fixed_state = snapshot.symbols[fixed_symbol]
        if is_active_symbol_state(fixed_state) and fixed_state.used_layers > 0:
            return fixed_symbol
        return None

    # ── Dynamic mode (original logic) ──────────────────────────────────────
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
    config: LeaderFollowerConfig | None = None,
) -> LeaderFollowerPermissions:
    """Build leader/follower permissions for every symbol in *snapshot*.

    Parameters
    ----------
    snapshot:
        Current capital ledger snapshot.
    config:
        Leader/follower config.  ``None`` defaults to dynamic mode for backward
        compatibility.

    Returns a ``LeaderFollowerPermissions`` containing a ``SymbolPermission``
    for each symbol in ``snapshot.symbols``.
    """
    cfg = config if config is not None else LeaderFollowerConfig(leader_mode="dynamic")
    is_fixed = cfg.leader_mode == "fixed"

    leader_symbol = resolve_leader_symbol(snapshot, config=cfg)
    leader_state = (
        snapshot.symbols[leader_symbol] if leader_symbol is not None else None
    )

    permissions: dict[str, SymbolPermission] = {}

    for inst_id, state in snapshot.symbols.items():
        if leader_symbol is None:
            # -- No leader → every symbol is NEUTRAL -------------------------
            reason = (
                "FIXED_LEADER_FLAT_NO_PRESSURE"
                if is_fixed
                else "NO_PRESSURE_LEADER"
            )
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
                reason=reason,
            )
        elif inst_id == leader_symbol:
            # -- Leader -------------------------------------------------------
            reason = (
                "FIXED_LEADER_ACTIVE_BELOW_PRESSURE"
                if is_fixed and leader_state.used_layers < 3
                else "ACTIVE_LEADER"
            )
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
                reason=reason,
            )
        else:
            # -- Follower — restrictions depend on leader layer count --------
            leader_layers = leader_state.used_layers

            if is_fixed and leader_layers < 3:
                # Fixed leader active but below pressure threshold →
                # followers are NEUTRAL-like with no restrictions.
                permissions[inst_id] = _follower_permission(
                    inst_id=inst_id,
                    leader_symbol=leader_symbol,
                    leader_used_layers=leader_layers,
                    state=state,
                    cap=state.plan_max_layers,
                    gap="1.0",
                    freeze="1.0",
                    no_new_entry=False,
                    no_add_layer=False,
                    no_new_sidecar_leg=False,
                    reason="FIXED_LEADER_ACTIVE_BELOW_PRESSURE",
                )
            elif leader_layers == 3:
                reason = (
                    "FIXED_LEADER_LAYER_3_FOLLOWER_CAUTION"
                    if is_fixed
                    else "LEADER_LAYER_3_FOLLOWER_CAUTION"
                )
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
                    reason=reason,
                )
            elif leader_layers == 4:
                reason = (
                    "FIXED_LEADER_LAYER_4_FOLLOWER_DEFENSIVE"
                    if is_fixed
                    else "LEADER_LAYER_4_FOLLOWER_DEFENSIVE"
                )
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
                    reason=reason,
                )
            else:  # leader_layers >= 5
                reason = (
                    "FIXED_LEADER_LAYER_5_PLUS_FOLLOWER_NO_NEW_RISK"
                    if is_fixed
                    else "LEADER_LAYER_5_PLUS_FOLLOWER_NO_NEW_RISK"
                )
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
                    reason=reason,
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
