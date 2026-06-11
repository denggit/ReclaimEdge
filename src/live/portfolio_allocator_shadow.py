# -*- coding: utf-8 -*-
"""
G05: Portfolio allocator shadow mode —— 只记录判断，不改变 ETH 实盘行为。

Live adapter that wraps G04 ``check_allocation_dry_run()`` and records the
decision as a journal event.  This module is **fire-and-forget** by design:

- Never blocks the real order path.
- Never writes the CapitalLedger.
- Never raises to the caller.
- Disabled by default; only activates when ``PORTFOLIO_ALLOCATOR_SHADOW_ENABLED=true``.

Intended usage (inside ExecutionCommandProcessor)::

    self._schedule_portfolio_allocator_shadow(
        command=raw_entry_command,
        sidecar_plan=combined_plan.sidecar_plan,
        position_id=current_position_id,
    )
    # ... then immediately proceed to execute_intent without awaiting.

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
    CapitalLedger,
    LeaderFollowerConfig,
    check_allocation_dry_run,
    create_main_position_plan,
)
from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.execution.trader import Trader
    from src.position_management.sidecar.planner import SidecarExecutionPlan
    from src.reporting.trade_journal import LiveTradeJournal
    from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy

    # Late-import from live runtime types
    from src.live import runtime_types as _rt

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ENV_PREFIX = "PORTFOLIO_ALLOCATOR"
_DEFAULT_GLOBAL_MAIN_CAP_PCT = "0.70"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 0.25

# G08c: leader/follower fixed mode env keys
_ENV_LEADER_MODE = "PORTFOLIO_LEADER_MODE"
_ENV_FIXED_LEADER_SYMBOL = "PORTFOLIO_FIXED_LEADER_SYMBOL"


def _leader_follower_config_from_env() -> LeaderFollowerConfig:
    """Build a ``LeaderFollowerConfig`` from environment variables."""
    from src.portfolio.leader_follower import LeaderFollowerConfig as LFC

    mode_raw = os.getenv(_ENV_LEADER_MODE, "fixed").strip().lower()
    fixed_symbol_raw = os.getenv(_ENV_FIXED_LEADER_SYMBOL, "").strip()
    if fixed_symbol_raw:
        return LFC(leader_mode=mode_raw, fixed_leader_symbol=fixed_symbol_raw)  # type: ignore[arg-type]
    elif mode_raw == "fixed":
        return LFC(leader_mode="fixed")
    else:
        return LFC(leader_mode="dynamic")

_ENTRY_INTENT_TYPES: frozenset[str] = frozenset({
    "OPEN_LONG",
    "OPEN_SHORT",
    "ADD_LONG",
    "ADD_SHORT",
})

_OPEN_INTENT_TYPES: frozenset[str] = frozenset({"OPEN_LONG", "OPEN_SHORT"})


@dataclass(frozen=True)
class PortfolioAllocatorShadowConfig:
    """Immutable configuration for the shadow allocator.

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
    ) -> "PortfolioAllocatorShadowConfig":
        """Build config from environment variables.

        Parameters
        ----------
        runtime_dir:
            Base runtime directory.  Used as the parent of the default
            ``portfolio/capital_ledger.json`` and ``.lock`` paths when the
            corresponding env vars are empty.
        """
        _rd = Path(runtime_dir)

        enabled = os.getenv(f"{_ENV_PREFIX}_SHADOW_ENABLED", "false").strip().lower() in (
            "true", "1", "yes", "on",
        )

        global_main_cap_pct = os.getenv(
            f"{_ENV_PREFIX}_GLOBAL_MAIN_CAP_PCT",
            _DEFAULT_GLOBAL_MAIN_CAP_PCT,
        ).strip() or _DEFAULT_GLOBAL_MAIN_CAP_PCT

        ledger_path_str = os.getenv("PORTFOLIO_LEDGER_PATH", "").strip()
        lock_path_str = os.getenv("PORTFOLIO_LEDGER_LOCK_PATH", "").strip()

        lock_timeout_str = os.getenv(
            f"{_ENV_PREFIX}_SHADOW_LOCK_TIMEOUT_SECONDS",
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
# Runner
# ---------------------------------------------------------------------------


@dataclass
class PortfolioAllocatorShadowRunner:
    """Fire-and-forget shadow runner that wraps G04 ``check_allocation_dry_run``.

    Construct via :meth:`from_config` rather than directly.
    """

    config: PortfolioAllocatorShadowConfig
    ledger: CapitalLedger

    @classmethod
    def from_config(
        cls,
        config: PortfolioAllocatorShadowConfig,
    ) -> "PortfolioAllocatorShadowRunner":
        """Create a runner from a validated config."""
        return cls(
            config=config,
            ledger=CapitalLedger(
                ledger_path=config.ledger_path,
                lock_path=config.lock_path,
                lock_timeout_seconds=config.lock_timeout_seconds,
            ),
        )

    # -- main entry ------------------------------------------------------------

    async def run_entry_shadow_check(
        self,
        *,
        command: "_rt.TradeCommand",
        trader: "Trader",
        strategy: "BollCvdShockReclaimStrategy",
        journal: "LiveTradeJournal",
        position_id: str | None,
        sidecar_plan: "SidecarExecutionPlan | None" = None,
    ) -> None:
        """Run a shadow allocation check for a single entry command.

        This method is **fire-and-forget**: it never raises, never blocks the
        caller, and never mutates any live state.

        Parameters
        ----------
        command:
            The raw entry TradeCommand (before sidecar combination).
        trader:
            The live Trader instance (read-only access to equity, metadata).
        strategy:
            The strategy instance (read-only access to config, sizer).
        journal:
            The live trade journal for recording shadow events.
        position_id:
            The current position ID (may be ``None`` for first entry).
        sidecar_plan:
            Optional sidecar execution plan from the combined entry build.
        """
        # ── Guard: disabled ──────────────────────────────────────────────────
        if not self.config.enabled:
            return

        intent = command.intent
        intent_type = getattr(intent, "intent_type", None)
        if intent_type not in _ENTRY_INTENT_TYPES:
            return

        try:
            await self._run_main_check(
                command=command,
                trader=trader,
                strategy=strategy,
                journal=journal,
                position_id=position_id,
                intent_type=str(intent_type),
            )

            # ── Sidecar shadow (optional) ──────────────────────────────────
            if sidecar_plan is not None and getattr(sidecar_plan, "enabled", False):
                await self._run_sidecar_check(
                    trader=trader,
                    journal=journal,
                    position_id=position_id,
                    sidecar_plan=sidecar_plan,
                )

        except Exception as exc:
            logger.exception(
                "PORTFOLIO_ALLOCATOR_SHADOW_FAILED | symbol=%s intent_type=%s error=%s",
                getattr(trader, "symbol", "?"),
                intent_type,
                exc,
            )
            try:
                journal.append(
                    "PORTFOLIO_ALLOCATOR_SHADOW_FAILED",
                    {
                        "symbol": getattr(trader, "symbol", "?"),
                        "intent_type": intent_type,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "shadow_mode": True,
                    },
                    position_id=position_id,
                )
            except Exception:
                logger.exception("PORTFOLIO_ALLOCATOR_SHADOW_FAILED_JOURNAL_FAILED")

    # -- internal helpers ------------------------------------------------------

    async def _run_main_check(
        self,
        *,
        command: "_rt.TradeCommand",
        trader: "Trader",
        strategy: "BollCvdShockReclaimStrategy",
        journal: "LiveTradeJournal",
        position_id: str | None,
        intent_type: str,
    ) -> None:
        """Build request → run check → record journal for main entry."""
        intent = command.intent

        # --- action mapping --------------------------------------------------
        if intent_type in _OPEN_INTENT_TYPES:
            action = "OPEN_MAIN"
        else:
            action = "ADD_MAIN"

        # --- side ------------------------------------------------------------
        side = str(getattr(intent, "side", ""))

        # --- requested layer -------------------------------------------------
        layer_index = getattr(intent, "layer_index", 1)
        if action == "OPEN_MAIN":
            requested_layer = 1
        else:
            requested_layer = layer_index

        # --- main margin delta ------------------------------------------------
        size = getattr(intent, "size", None)
        margin_usdt = getattr(size, "margin_usdt", 0.0) if size is not None else 0.0
        main_margin_delta_usdt = str(margin_usdt)

        # --- account equity ---------------------------------------------------
        account_equity_usdt = str(getattr(trader, "account_equity_usdt", 0))

        # --- position plan (OPEN_MAIN only) -----------------------------------
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
                    "PORTFOLIO_ALLOCATOR_SHADOW_PLAN_BUILD_FAILED | "
                    "symbol=%s intent_type=%s",
                    getattr(trader, "symbol", "?"),
                    intent_type,
                )
                # Don't raise — record a failed journal entry for this sub-step
                journal.append(
                    "PORTFOLIO_ALLOCATOR_SHADOW_FAILED",
                    {
                        "symbol": getattr(trader, "symbol", "?"),
                        "intent_type": intent_type,
                        "error_type": "PositionPlanBuildError",
                        "error": "Failed to build position plan for shadow check",
                        "shadow_mode": True,
                    },
                    position_id=position_id,
                )
                return

        # --- build request ----------------------------------------------------
        request = AllocationCheckRequest(
            inst_id=str(getattr(trader, "symbol", "")),
            action=action,  # type: ignore[arg-type]
            side=side,
            requested_layer=requested_layer,
            position_plan=position_plan,
            main_margin_delta_usdt=main_margin_delta_usdt,
            account_equity_usdt=account_equity_usdt,
            global_main_cap_pct=self.config.global_main_cap_pct,
        )

        # --- read snapshot (off thread) ---------------------------------------
        snapshot = await asyncio.to_thread(self.ledger.read_locked)

        # --- run check --------------------------------------------------------
        decision = check_allocation_dry_run(
            snapshot=snapshot,
            request=request,
            leader_follower_config=self.config.leader_follower_config,
        )

        # --- record journal ---------------------------------------------------
        permission = decision.permission
        journal.append(
            "PORTFOLIO_ALLOCATOR_SHADOW",
            {
                "symbol": getattr(trader, "symbol", ""),
                "intent_type": intent_type,
                "side": side,
                "layer_index": layer_index,
                "allocator_action": request.action,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "leader_symbol": decision.leader_symbol,
                "projected_leader_symbol": decision.projected_snapshot.leader_symbol,
                "permission_role": permission.role if permission else None,
                "permission_max_layers": permission.permission_max_layers if permission else None,
                "no_new_entry": permission.no_new_entry if permission else None,
                "no_add_layer": permission.no_add_layer if permission else None,
                "no_new_sidecar_leg": permission.no_new_sidecar_leg if permission else None,
                "account_equity_usdt": account_equity_usdt,
                "main_margin_delta_usdt": main_margin_delta_usdt,
                "global_main_cap_pct": self.config.global_main_cap_pct,
                "shadow_mode": True,
            },
            position_id=position_id,
        )

        logger.info(
            "PORTFOLIO_ALLOCATOR_SHADOW | symbol=%s intent_type=%s action=%s "
            "allowed=%s reason=%s leader=%s",
            getattr(trader, "symbol", ""),
            intent_type,
            request.action,
            decision.allowed,
            decision.reason,
            decision.leader_symbol,
        )

    async def _run_sidecar_check(
        self,
        *,
        trader: "Trader",
        journal: "LiveTradeJournal",
        position_id: str | None,
        sidecar_plan: "SidecarExecutionPlan",
    ) -> None:
        """Run a shadow OPEN_SIDECAR check and record a separate journal event."""
        account_equity = Decimal(str(getattr(trader, "account_equity_usdt", 1)))
        sidecar_margin_pct = Decimal(
            str(getattr(sidecar_plan, "sidecar_margin_pct", "0"))
        )
        sidecar_margin_delta = account_equity * sidecar_margin_pct

        request = AllocationCheckRequest(
            inst_id=str(getattr(trader, "symbol", "")),
            action="OPEN_SIDECAR",
            side=None,
            requested_layer=None,
            position_plan=None,
            sidecar_margin_delta_usdt=str(sidecar_margin_delta),
            account_equity_usdt=str(account_equity),
            global_main_cap_pct=self.config.global_main_cap_pct,
        )

        snapshot = await asyncio.to_thread(self.ledger.read_locked)
        decision = check_allocation_dry_run(
            snapshot=snapshot,
            request=request,
            leader_follower_config=self.config.leader_follower_config,
        )

        permission = decision.permission
        journal.append(
            "PORTFOLIO_ALLOCATOR_SHADOW_SIDECAR",
            {
                "symbol": getattr(trader, "symbol", ""),
                "allocator_action": "OPEN_SIDECAR",
                "allowed": decision.allowed,
                "reason": decision.reason,
                "leader_symbol": decision.leader_symbol,
                "projected_leader_symbol": decision.projected_snapshot.leader_symbol,
                "permission_role": permission.role if permission else None,
                "permission_max_layers": permission.permission_max_layers if permission else None,
                "no_new_entry": permission.no_new_entry if permission else None,
                "no_new_sidecar_leg": permission.no_new_sidecar_leg if permission else None,
                "sidecar_margin_delta_usdt": str(sidecar_margin_delta),
                "account_equity_usdt": str(account_equity),
                "global_main_cap_pct": self.config.global_main_cap_pct,
                "shadow_mode": True,
            },
            position_id=position_id,
        )

        logger.info(
            "PORTFOLIO_ALLOCATOR_SHADOW_SIDECAR | symbol=%s allowed=%s reason=%s",
            getattr(trader, "symbol", ""),
            decision.allowed,
            decision.reason,
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
    """Build a PositionPlan from live objects for an OPEN_MAIN shadow check."""
    intent = command.intent

    # side
    side = str(getattr(intent, "side", ""))

    # base_main_contracts = eth_qty / contract_multiplier
    size = getattr(intent, "size", None)
    eth_qty = getattr(size, "eth_qty", 0.0) if size is not None else 0.0
    multiplier = getattr(trader, "contract_multiplier", Decimal("0.1"))
    base_main_contracts = Decimal(str(eth_qty)) / Decimal(str(multiplier))

    # strategy config
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
