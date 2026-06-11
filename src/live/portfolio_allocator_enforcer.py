# -*- coding: utf-8 -*-
"""
G06a: Portfolio allocator enforce mode —— 下单前同步检查，成功成交后写 ledger。

与 G05 shadow mode 的区别:
  - enforce mode **同步等待** allocator precheck 结果。
  - rejected 时**跳过真实下单**（fail closed）。
  - allowed 时继续下单，并在成交后把 projected_snapshot 写回 CapitalLedger。

延迟原则:
  - 只在 ExecutionCommandProcessor 的 OPEN/ADD/SIDECAR 执行路径里做。
  - 不在 strategy_tick_worker / on_tick / 策略信号计算路径里做。
  - ledger lock timeout 有上限，超时 fail closed。

默认关闭，不影响当前 ETH 实盘。
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from src.portfolio import (
    AllocationCheckRequest,
    AllocationDecision,
    CapitalLedger,
    CapitalLedgerSnapshot,
    LeaderFollowerConfig,
    check_allocation_dry_run,
    create_main_position_plan,
)
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import LiveTradeResult, Trader
    from src.position_management.sidecar.planner import SidecarExecutionPlan
    from src.reporting.trade_journal import LiveTradeJournal
    from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy

    from src.live import runtime_types as _rt

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_PREFIX = "PORTFOLIO_ALLOCATOR"
_DEFAULT_GLOBAL_MAIN_CAP_PCT = "0.70"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 1.0

# G08c: leader/follower fixed mode env keys
_ENV_LEADER_MODE = "PORTFOLIO_LEADER_MODE"
_ENV_FIXED_LEADER_SYMBOL = "PORTFOLIO_FIXED_LEADER_SYMBOL"


def _leader_follower_config_from_env() -> LeaderFollowerConfig:
    """Build a ``LeaderFollowerConfig`` from environment variables."""
    from src.portfolio.leader_follower import LeaderFollowerConfig as LFC

    mode_raw = os.getenv(_ENV_LEADER_MODE, "fixed").strip().lower()
    fixed_symbol_raw = os.getenv(_ENV_FIXED_LEADER_SYMBOL, "").strip()

    if mode_raw not in ("fixed", "dynamic"):
        return LFC(leader_mode=mode_raw)  # type: ignore[arg-type]

    if mode_raw == "fixed":
        if fixed_symbol_raw:
            return LFC(leader_mode="fixed", fixed_leader_symbol=fixed_symbol_raw)
        return LFC(leader_mode="fixed")

    return LFC(leader_mode="dynamic")

_ENTRY_INTENT_TYPES: frozenset[str] = frozenset({
    "OPEN_LONG",
    "OPEN_SHORT",
    "ADD_LONG",
    "ADD_SHORT",
})

_OPEN_INTENT_TYPES: frozenset[str] = frozenset({"OPEN_LONG", "OPEN_SHORT"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioAllocatorEnforceConfig:
    """Immutable configuration for the enforce allocator.

    All fields default to disabled / safe values so that importing this module
    never triggers filesystem access.
    """

    enabled: bool = False
    global_main_cap_pct: str = _DEFAULT_GLOBAL_MAIN_CAP_PCT
    ledger_path: Path = Path("runtime/portfolio/capital_ledger.json")
    lock_path: Path = Path("runtime/portfolio/capital_ledger.lock")
    lock_timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS
    leader_follower_config: LeaderFollowerConfig = field(
        default_factory=_leader_follower_config_from_env,
    )

    @classmethod
    def from_env(
        cls,
        *,
        runtime_dir: str | Path = "runtime",
    ) -> "PortfolioAllocatorEnforceConfig":
        """Build config from environment variables.

        Parameters
        ----------
        runtime_dir:
            Base runtime directory.  Used as the parent of the default
            ``portfolio/capital_ledger.json`` and ``.lock`` paths when the
            corresponding env vars are empty.
        """
        _rd = Path(runtime_dir)

        enabled = os.getenv(
            f"{_ENV_PREFIX}_ENFORCE_ENABLED", "false"
        ).strip().lower() in ("true", "1", "yes", "on")

        global_main_cap_pct = os.getenv(
            f"{_ENV_PREFIX}_GLOBAL_MAIN_CAP_PCT",
            _DEFAULT_GLOBAL_MAIN_CAP_PCT,
        ).strip() or _DEFAULT_GLOBAL_MAIN_CAP_PCT

        ledger_path_str = os.getenv("PORTFOLIO_LEDGER_PATH", "").strip()
        lock_path_str = os.getenv("PORTFOLIO_LEDGER_LOCK_PATH", "").strip()

        lock_timeout_str = os.getenv(
            f"{_ENV_PREFIX}_ENFORCE_LOCK_TIMEOUT_SECONDS",
            str(_DEFAULT_LOCK_TIMEOUT_SECONDS),
        ).strip()

        try:
            lock_timeout_seconds = float(lock_timeout_str)
        except (ValueError, TypeError):
            lock_timeout_seconds = _DEFAULT_LOCK_TIMEOUT_SECONDS

        return cls(
            enabled=enabled,
            global_main_cap_pct=global_main_cap_pct,
            ledger_path=Path(ledger_path_str) if ledger_path_str else _rd / "portfolio" / "capital_ledger.json",
            lock_path=Path(lock_path_str) if lock_path_str else _rd / "portfolio" / "capital_ledger.lock",
            lock_timeout_seconds=lock_timeout_seconds,
            leader_follower_config=_leader_follower_config_from_env(),
        )


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioAllocatorPrecheckResult:
    """Result of an enforce precheck.

    Fields
    ------
    enabled:
        Whether enforce mode is enabled.  ``False`` means no interception.
    allowed:
        Whether the order may proceed to real execution.
    reason:
        Human-readable reason string (e.g. ``"ENFORCE_DISABLED"``,
        ``"ALLOCATOR_ENFORCE_ALLOWED"``, ``"GLOBAL_NO_NEW_ENTRY"``).
    main_decision:
        The raw AllocationDecision from the main check (``None`` when not run).
    sidecar_decision:
        The raw AllocationDecision from the sidecar check (``None`` when not run).
    projected_snapshot:
        The snapshot to write to ledger after a successful fill.
        Must be non-``None`` when ``allowed=True``.
    message:
        Optional human-readable additional detail.
    """

    enabled: bool
    allowed: bool
    reason: str
    main_decision: AllocationDecision | None = None
    sidecar_decision: AllocationDecision | None = None
    projected_snapshot: CapitalLedgerSnapshot | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class PortfolioAllocatorEnforcer:
    """Synchronous (awaited) enforce runner wrapping G04 ``check_allocation_dry_run``.

    Construct via :meth:`from_config` rather than directly.

    Unlike the G05 shadow runner, this module:
      - Is **awaited** synchronously before order placement.
      - **Blocks** the real order path when a new-risk intent is rejected.
      - **Writes** the projected snapshot back to CapitalLedger after a successful fill.
      - Never raises to the caller (fail closed).
    """

    config: PortfolioAllocatorEnforceConfig
    ledger: CapitalLedger

    @classmethod
    def from_config(
        cls,
        config: PortfolioAllocatorEnforceConfig,
    ) -> "PortfolioAllocatorEnforcer":
        """Create an enforcer from a validated config."""
        return cls(
            config=config,
            ledger=CapitalLedger(
                ledger_path=config.ledger_path,
                lock_path=config.lock_path,
                lock_timeout_seconds=config.lock_timeout_seconds,
            ),
        )

    # -----------------------------------------------------------------------
    # Main entry: precheck
    # -----------------------------------------------------------------------

    async def precheck_entry_allocation(
        self,
        *,
        command: "_rt.TradeCommand",
        trader: "Trader",
        strategy: "BollCvdShockReclaimStrategy",
        journal: "LiveTradeJournal",
        position_id: str | None,
        sidecar_plan: "SidecarExecutionPlan | None" = None,
    ) -> PortfolioAllocatorPrecheckResult:
        """Run an enforce allocation precheck for a single entry command.

        This method is **awaited synchronously** before order placement.
        It never raises — all errors result in fail-closed (allowed=False).

        Parameters
        ----------
        command:
            The raw entry TradeCommand (before sidecar combination).
        trader:
            The live Trader instance (read-only access to equity, metadata).
        strategy:
            The strategy instance (read-only access to config, sizer).
        journal:
            The live trade journal for recording enforce events.
        position_id:
            The current position ID (may be ``None`` for first entry).
        sidecar_plan:
            Optional sidecar execution plan from the combined entry build.

        Returns
        -------
        PortfolioAllocatorPrecheckResult
        """
        # ── Guard: disabled ──────────────────────────────────────────────
        if not self.config.enabled:
            return PortfolioAllocatorPrecheckResult(
                enabled=False,
                allowed=True,
                reason="ENFORCE_DISABLED",
            )

        # ── Guard: non-entry intent ──────────────────────────────────────
        intent = command.intent
        intent_type = getattr(intent, "intent_type", None)
        if intent_type not in _ENTRY_INTENT_TYPES:
            return PortfolioAllocatorPrecheckResult(
                enabled=True,
                allowed=True,
                reason="NON_ENTRY_INTENT",
            )

        try:
            # ── Read ledger snapshot ─────────────────────────────────────
            snapshot = await asyncio.to_thread(self.ledger.read_locked)

            # ── Main check ──────────────────────────────────────────────
            main_request = _build_main_request(
                command=command,
                trader=trader,
                strategy=strategy,
                config=self.config,
                position_id=position_id,
            )

            if main_request is None:
                # Position plan build failed
                journal.append(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_FAILED",
                    {
                        "symbol": getattr(trader, "symbol", "?"),
                        "intent_type": intent_type,
                        "error_type": "PositionPlanBuildError",
                        "error": "Failed to build position plan for enforce check",
                        "enforce_mode": True,
                    },
                    position_id=position_id,
                )
                return PortfolioAllocatorPrecheckResult(
                    enabled=True,
                    allowed=False,
                    reason="POSITION_PLAN_BUILD_FAILED",
                )

            main_decision = check_allocation_dry_run(
                snapshot=snapshot, request=main_request,
                leader_follower_config=self.config.leader_follower_config,
            )

            # ── Journal main precheck ────────────────────────────────────
            permission = main_decision.permission
            journal.append(
                "PORTFOLIO_ALLOCATOR_ENFORCE_PRECHECK",
                {
                    "symbol": getattr(trader, "symbol", ""),
                    "intent_type": intent_type,
                    "side": getattr(intent, "side", ""),
                    "layer_index": getattr(intent, "layer_index", 1),
                    "allocator_action": main_request.action,
                    "allowed": main_decision.allowed,
                    "reason": main_decision.reason,
                    "leader_symbol": main_decision.leader_symbol,
                    "projected_leader_symbol": main_decision.projected_snapshot.leader_symbol,
                    "permission_role": permission.role if permission else None,
                    "permission_max_layers": permission.permission_max_layers if permission else None,
                    "account_equity_usdt": str(getattr(trader, "account_equity_usdt", 0)),
                    "main_margin_delta_usdt": main_request.main_margin_delta_usdt,
                    "global_main_cap_pct": self.config.global_main_cap_pct,
                    "enforce_mode": True,
                },
                position_id=position_id,
            )

            # ── Main rejected ────────────────────────────────────────────
            if not main_decision.allowed:
                journal.append(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED",
                    {
                        "symbol": getattr(trader, "symbol", ""),
                        "intent_type": intent_type,
                        "reason": main_decision.reason,
                        "enforce_mode": True,
                    },
                    position_id=position_id,
                )
                logger.warning(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED | symbol=%s intent_type=%s reason=%s",
                    getattr(trader, "symbol", ""),
                    intent_type,
                    main_decision.reason,
                )
                return PortfolioAllocatorPrecheckResult(
                    enabled=True,
                    allowed=False,
                    reason=main_decision.reason,
                    main_decision=main_decision,
                )

            projected = main_decision.projected_snapshot

            # ── Sidecar check (optional) ─────────────────────────────────
            sidecar_decision = None
            if sidecar_plan is not None and getattr(sidecar_plan, "enabled", False):
                sidecar_request = _build_sidecar_request(
                    trader=trader,
                    config=self.config,
                    sidecar_plan=sidecar_plan,
                )
                sidecar_decision = check_allocation_dry_run(
                    snapshot=projected, request=sidecar_request,
                    leader_follower_config=self.config.leader_follower_config,
                )

                # ── Journal sidecar precheck ────────────────────────────
                sidecar_permission = sidecar_decision.permission
                journal.append(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_PRECHECK_SIDECAR",
                    {
                        "symbol": getattr(trader, "symbol", ""),
                        "allocator_action": "OPEN_SIDECAR",
                        "allowed": sidecar_decision.allowed,
                        "reason": sidecar_decision.reason,
                        "leader_symbol": sidecar_decision.leader_symbol,
                        "projected_leader_symbol": sidecar_decision.projected_snapshot.leader_symbol,
                        "permission_role": sidecar_permission.role if sidecar_permission else None,
                        "permission_max_layers": sidecar_permission.permission_max_layers if sidecar_permission else None,
                        "account_equity_usdt": str(getattr(trader, "account_equity_usdt", 0)),
                        "enforce_mode": True,
                    },
                    position_id=position_id,
                )

                # ── Sidecar rejected ────────────────────────────────────
                if not sidecar_decision.allowed:
                    sidecar_reason = f"SIDECAR_{sidecar_decision.reason}"
                    journal.append(
                        "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED",
                        {
                            "symbol": getattr(trader, "symbol", ""),
                            "intent_type": intent_type,
                            "reason": sidecar_reason,
                            "enforce_mode": True,
                        },
                        position_id=position_id,
                    )
                    logger.warning(
                        "PORTFOLIO_ALLOCATOR_ENFORCE_REJECTED | symbol=%s intent_type=%s reason=%s",
                        getattr(trader, "symbol", ""),
                        intent_type,
                        sidecar_reason,
                    )
                    return PortfolioAllocatorPrecheckResult(
                        enabled=True,
                        allowed=False,
                        reason=sidecar_reason,
                        main_decision=main_decision,
                        sidecar_decision=sidecar_decision,
                    )

                projected = sidecar_decision.projected_snapshot

            # ── Allowed ──────────────────────────────────────────────────
            logger.info(
                "PORTFOLIO_ALLOCATOR_ENFORCE_ALLOWED | symbol=%s intent_type=%s action=%s",
                getattr(trader, "symbol", ""),
                intent_type,
                main_request.action,
            )
            return PortfolioAllocatorPrecheckResult(
                enabled=True,
                allowed=True,
                reason="ALLOCATOR_ENFORCE_ALLOWED",
                main_decision=main_decision,
                sidecar_decision=sidecar_decision,
                projected_snapshot=projected,
            )

        except Exception as exc:
            logger.exception(
                "PORTFOLIO_ALLOCATOR_ENFORCE_FAILED | symbol=%s intent_type=%s error=%s",
                getattr(trader, "symbol", "?"),
                intent_type,
                exc,
            )
            try:
                journal.append(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_FAILED",
                    {
                        "symbol": getattr(trader, "symbol", "?"),
                        "intent_type": intent_type,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "enforce_mode": True,
                    },
                    position_id=position_id,
                )
            except Exception:
                logger.exception("PORTFOLIO_ALLOCATOR_ENFORCE_FAILED_JOURNAL_FAILED")

            return PortfolioAllocatorPrecheckResult(
                enabled=True,
                allowed=False,
                reason="ALLOCATOR_ENFORCE_ERROR",
            )

    # -----------------------------------------------------------------------
    # Commit projected snapshot after fill
    # -----------------------------------------------------------------------

    async def commit_projected_snapshot_after_fill(
        self,
        *,
        precheck_result: PortfolioAllocatorPrecheckResult,
        live_result: "LiveTradeResult",
        journal: "LiveTradeJournal",
        position_id: str | None,
    ) -> None:
        """Write the projected snapshot back to CapitalLedger after a successful fill.

        This method is called **after** the real order has been placed.
        It catches all exceptions internally — a commit failure must never
        affect the existing error handling in the caller.

        Rules
        -----
        1. Only writes when enforce mode is enabled.
        2. Only writes when the precheck allowed the order.
        3. Only writes when ``projected_snapshot`` is not ``None``.
        4. Only writes when ``live_result.entry_filled=True`` or ``live_result.ok=True``.
        5. If ``live_result.ok=False`` and ``entry_filled=False``, does NOT write.
        6. Write failure is caught, journaled, and never raised.
        """
        if not self.config.enabled:
            return
        if not precheck_result.enabled:
            return
        if not precheck_result.allowed:
            return
        if precheck_result.projected_snapshot is None:
            return

        entry_filled = getattr(live_result, "entry_filled", False)
        ok = getattr(live_result, "ok", False)

        if not entry_filled and not ok:
            return

        try:
            # The mutator simply returns the pre-computed projected snapshot.
            def _mutator(_current: CapitalLedgerSnapshot) -> CapitalLedgerSnapshot:
                return precheck_result.projected_snapshot  # type: ignore[return-value]

            await asyncio.to_thread(self.ledger.update_locked, _mutator)

            try:
                journal.append(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_COMMITTED",
                    {
                        "symbol": getattr(live_result, "action", "?"),
                        "entry_filled": entry_filled,
                        "ok": ok,
                        "enforce_mode": True,
                    },
                    position_id=position_id,
                )
            except Exception:
                logger.exception("PORTFOLIO_ALLOCATOR_ENFORCE_COMMITTED_JOURNAL_FAILED")

            logger.info(
                "PORTFOLIO_ALLOCATOR_ENFORCE_COMMITTED | entry_filled=%s ok=%s",
                entry_filled,
                ok,
            )

        except Exception as exc:
            logger.exception(
                "PORTFOLIO_ALLOCATOR_ENFORCE_COMMIT_FAILED | error=%s",
                exc,
            )
            try:
                journal.append(
                    "PORTFOLIO_ALLOCATOR_ENFORCE_COMMIT_FAILED",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "entry_filled": entry_filled,
                        "ok": ok,
                        "enforce_mode": True,
                    },
                    position_id=position_id,
                )
            except Exception:
                logger.exception("PORTFOLIO_ALLOCATOR_ENFORCE_COMMIT_FAILED_JOURNAL_FAILED")


# ---------------------------------------------------------------------------
# Request builders (internal)
# ---------------------------------------------------------------------------


def _build_main_request(
    *,
    command: "_rt.TradeCommand",
    trader: "Trader",
    strategy: "BollCvdShockReclaimStrategy",
    config: PortfolioAllocatorEnforceConfig,
    position_id: str | None,
) -> AllocationCheckRequest | None:
    """Build an AllocationCheckRequest for the main OPEN or ADD check.

    Returns ``None`` if the position plan build fails (OPEN_MAIN only).
    """
    intent = command.intent
    intent_type = getattr(intent, "intent_type", "")

    # --- action mapping ----------------------------------------------------
    if intent_type in _OPEN_INTENT_TYPES:
        action = "OPEN_MAIN"
    else:
        action = "ADD_MAIN"

    # --- side --------------------------------------------------------------
    side = str(getattr(intent, "side", ""))

    # --- requested layer ---------------------------------------------------
    layer_index = getattr(intent, "layer_index", 1)
    if action == "OPEN_MAIN":
        requested_layer = 1
    else:
        requested_layer = layer_index

    # --- main margin delta --------------------------------------------------
    size = getattr(intent, "size", None)
    margin_usdt = getattr(size, "margin_usdt", 0.0) if size is not None else 0.0
    main_margin_delta_usdt = str(margin_usdt)

    # --- account equity -----------------------------------------------------
    account_equity_usdt = str(getattr(trader, "account_equity_usdt", 0))

    # --- position plan (OPEN_MAIN only) -------------------------------------
    position_plan = None
    if action == "OPEN_MAIN":
        try:
            position_plan = _build_position_plan(
                command=command,
                trader=trader,
                strategy=strategy,
                position_id=position_id,
            )
        except Exception:
            logger.exception(
                "PORTFOLIO_ALLOCATOR_ENFORCE_PLAN_BUILD_FAILED | "
                "symbol=%s intent_type=%s",
                getattr(trader, "symbol", "?"),
                intent_type,
            )
            return None

    # --- build request ------------------------------------------------------
    return AllocationCheckRequest(
        inst_id=str(getattr(trader, "symbol", "")),
        action=action,  # type: ignore[arg-type]
        side=side,
        requested_layer=requested_layer,
        requested_main_contracts=_intent_requested_main_contracts(
            intent=intent,
            trader=trader,
        ),
        position_plan=position_plan,
        main_margin_delta_usdt=main_margin_delta_usdt,
        account_equity_usdt=account_equity_usdt,
        global_main_cap_pct=config.global_main_cap_pct,
    )


def _intent_requested_main_contracts(
    *,
    intent: object,
    trader: object,
) -> str | None:
    size = getattr(intent, "size", None)
    eth_qty = getattr(size, "eth_qty", None) if size is not None else None
    if eth_qty is None:
        return None

    multiplier = getattr(trader, "contract_multiplier", None)
    if multiplier is None:
        return None

    contracts = Decimal(str(eth_qty)) / Decimal(str(multiplier))
    return format(contracts, "f")


def _build_sidecar_request(
    *,
    trader: "Trader",
    config: PortfolioAllocatorEnforceConfig,
    sidecar_plan: "SidecarExecutionPlan",
) -> AllocationCheckRequest:
    """Build an AllocationCheckRequest for the OPEN_SIDECAR check."""
    account_equity = Decimal(str(getattr(trader, "account_equity_usdt", 1)))
    sidecar_margin_pct = Decimal(
        str(getattr(sidecar_plan, "sidecar_margin_pct", "0"))
    )
    sidecar_margin_delta = account_equity * sidecar_margin_pct

    return AllocationCheckRequest(
        inst_id=str(getattr(trader, "symbol", "")),
        action="OPEN_SIDECAR",
        side=None,
        requested_layer=None,
        position_plan=None,
        sidecar_margin_delta_usdt=str(sidecar_margin_delta),
        account_equity_usdt=str(account_equity),
        global_main_cap_pct=config.global_main_cap_pct,
    )


# ---------------------------------------------------------------------------
# Position plan helper
# ---------------------------------------------------------------------------


def _build_position_plan(
    *,
    command: "_rt.TradeCommand",
    trader: "Trader",
    strategy: "BollCvdShockReclaimStrategy",
    position_id: str | None,
) -> "PositionPlan":
    """Build a PositionPlan from live objects for an OPEN_MAIN enforce check."""
    intent = command.intent

    # side
    side = str(getattr(intent, "side", ""))

    # base_main_contracts = eth_qty / contract_multiplier
    size = getattr(intent, "size", None)
    eth_qty = getattr(size, "eth_qty", 0.0) if size is not None else 0.0
    multiplier = getattr(trader, "contract_multiplier", Decimal("0.1"))
    base_main_contracts = Decimal(str(eth_qty)) / Decimal(str(multiplier))

    # strategy config — MUST come from strategy, not hardcoded
    max_layers = int(getattr(strategy.config, "max_layers", 8))
    layer_multiplier_step = float(
        getattr(strategy.sizer.config, "layer_multiplier_step", 0.15)
    )
    contract_precision = getattr(trader, "contract_precision", Decimal("0.01"))
    min_contracts = getattr(trader, "min_contracts", Decimal("0.01"))

    plan = create_main_position_plan(
        inst_id=str(getattr(trader, "symbol", "")),
        side=side,
        base_main_contracts=base_main_contracts,
        max_layers=max_layers,
        layer_multiplier_step=str(layer_multiplier_step),
        contract_precision=contract_precision,
        min_contracts=min_contracts,
        plan_id=position_id or None,
        created_ms=getattr(intent, "ts_ms", 0),
    )
    return plan  # type: ignore[return-value]
