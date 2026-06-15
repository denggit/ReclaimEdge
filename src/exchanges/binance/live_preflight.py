#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : live_preflight.py
@Description: Binance live trading preflight / confirmation guard (exchange layer).

This module is intentionally free of side-effects:
- No network connections
- No API key reading
- No signing
- No order placement
- No imports of strategy / execution / exchange clients / semantic executor
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Primary (exchange-neutral) confirmation phrase — use this for new deployments.
LIVE_CONFIRMATION_PHRASE: str = "I_UNDERSTAND_EXCHANGE_LIVE_TRADING"
# Legacy Binance confirmation phrase — accepted for backward compatibility.
BINANCE_LIVE_CONFIRMATION_PHRASE: str = "I_UNDERSTAND_BINANCE_LIVE_TRADING"

# Set of all accepted confirmation phrases (both new and legacy).
_ACCEPTED_CONFIRMATION_PHRASES: frozenset[str] = frozenset({
    LIVE_CONFIRMATION_PHRASE,
    BINANCE_LIVE_CONFIRMATION_PHRASE,
})

BINANCE_LIVE_HARD_MAX_LEVERAGE: int = 20

# ---------------------------------------------------------------------------
# Exchange-neutral env var names (primary) with Binance backward-compat aliases
# ---------------------------------------------------------------------------

_ENV_PRIMARY_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("SIGNAL_ONLY", "BINANCE_SIGNAL_ONLY"),
    ("LIVE_ENABLED", "BINANCE_LIVE_ENABLED"),
    ("LIVE_ALLOW_ORDERS", "BINANCE_LIVE_ALLOW_ORDERS"),
    ("LIVE_CONFIRMATION", "BINANCE_LIVE_CONFIRMATION"),
    ("LIVE_MAX_ORDER_NOTIONAL_USDT", "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT"),
    ("LIVE_MAX_POSITION_NOTIONAL_USDT", "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT"),
    ("LIVE_LEVERAGE", "BINANCE_LIVE_LEVERAGE"),
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "y", "on"})


def _resolve_env(
    env: Mapping[str, str],
    primary: str,
    alias: str,
) -> str:
    """Read *primary* env var, falling back to *alias*.

    Returns the raw (un-stripped) value from *primary* if set,
    otherwise the value from *alias*.
    """
    value = env.get(primary, "")
    if value.strip():
        return value
    return env.get(alias, "")


def _detect_env_conflicts(env: Mapping[str, str]) -> list[str]:
    """Return a list of conflict descriptions for dual-name env pairs.

    A conflict exists when both the primary (LIVE_*) and the alias
    (BINANCE_*) names are set to different non-empty values.
    """
    conflicts: list[str] = []
    for primary, alias in _ENV_PRIMARY_ALIAS_PAIRS:
        p_val = env.get(primary, "").strip()
        a_val = env.get(alias, "").strip()
        if p_val and a_val and p_val != a_val:
            conflicts.append(
                f"{primary}={p_val!r} vs {alias}={a_val!r} — "
                f"use only {primary}"
            )
    return conflicts


def _parse_decimal(raw: str) -> Decimal | None:
    """Parse a Decimal from a trimmed string, returning None on failure."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _parse_int(raw: str) -> int | None:
    """Parse an int from a trimmed string, returning None on failure."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinanceLivePreflightConfig:
    """Parsed Binance live trading preflight configuration.

    Only reads public / control environment variables — no secrets.
    """

    exchange: str
    signal_only: bool
    live_enabled: bool
    allow_orders: bool
    confirmation: str
    max_order_notional_usdt: Decimal | None
    max_position_notional_usdt: Decimal | None
    leverage: int | None


@dataclass(frozen=True)
class BinanceLivePreflightReport:
    """Result of the Binance live trading preflight check.

    Attributes:
        ok: ``True`` when there are zero blocking reasons.
        config: The parsed preflight configuration.
        blocking_reasons: Tuple of reason codes that prevent launch.
    """

    ok: bool
    config: BinanceLivePreflightConfig
    blocking_reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_binance_live_preflight_config(
    env: Mapping[str, str] | None = None,
) -> BinanceLivePreflightConfig:
    """Parse Binance live preflight configuration from environment variables.

    Parameters
    ----------
    env:
        Optional mapping of environment variables.  When ``None`` (the
        default) the real ``os.environ`` is used.

    Returns
    -------
    BinanceLivePreflightConfig
        A frozen configuration object with all parsed values.
    """
    if env is None:
        env = os.environ

    exchange: str = env.get("EXCHANGE", "okx").strip().lower()
    if not exchange:
        exchange = "okx"

    signal_only_raw: str = _resolve_env(env, "SIGNAL_ONLY", "BINANCE_SIGNAL_ONLY").strip().lower()
    signal_only: bool = signal_only_raw in _TRUTHY

    live_enabled_raw: str = _resolve_env(env, "LIVE_ENABLED", "BINANCE_LIVE_ENABLED").strip().lower()
    live_enabled: bool = live_enabled_raw in _TRUTHY

    allow_orders_raw: str = _resolve_env(env, "LIVE_ALLOW_ORDERS", "BINANCE_LIVE_ALLOW_ORDERS").strip().lower()
    allow_orders: bool = allow_orders_raw in _TRUTHY

    confirmation: str = _resolve_env(env, "LIVE_CONFIRMATION", "BINANCE_LIVE_CONFIRMATION").strip()

    max_order_notional_usdt: Decimal | None = _parse_decimal(
        _resolve_env(env, "LIVE_MAX_ORDER_NOTIONAL_USDT", "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT")
    )
    max_position_notional_usdt: Decimal | None = _parse_decimal(
        _resolve_env(env, "LIVE_MAX_POSITION_NOTIONAL_USDT", "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT")
    )
    leverage: int | None = _parse_int(
        _resolve_env(env, "LIVE_LEVERAGE", "BINANCE_LIVE_LEVERAGE")
    )

    return BinanceLivePreflightConfig(
        exchange=exchange,
        signal_only=signal_only,
        live_enabled=live_enabled,
        allow_orders=allow_orders,
        confirmation=confirmation,
        max_order_notional_usdt=max_order_notional_usdt,
        max_position_notional_usdt=max_position_notional_usdt,
        leverage=leverage,
    )


def build_binance_live_preflight_report(
    env: Mapping[str, str] | None = None,
    *,
    orders_globally_enabled: bool = False,
) -> BinanceLivePreflightReport:
    """Build a preflight report for Binance live trading.

    Parameters
    ----------
    env:
        Optional mapping of environment variables.
    orders_globally_enabled:
        Code-level gate.  When ``False`` (the default) the report will
        always include ``binance_live_orders_disabled_by_build``, ensuring
        Binance live trading cannot be launched by accident.

    Returns
    -------
    BinanceLivePreflightReport
        A frozen report with ``ok=True`` only when every gate is satisfied.
    """
    config = load_binance_live_preflight_config(env)
    blocking: list[str] = []

    # ── Gate 0: no conflicting env var pairs ────────────────────────────
    _env_conflicts = _detect_env_conflicts(env if env is not None else os.environ)
    if _env_conflicts:
        blocking.append("live_env_var_conflict")
        # Print details to stderr so the user can see exactly what conflicts
        import sys as _sys
        for _msg in _env_conflicts:
            print(f"ERROR: conflicting env vars — {_msg}", file=_sys.stderr)

    # ── Gate 1: exchange must be binance ────────────────────────────────
    if config.exchange != "binance":
        blocking.append("exchange_is_not_binance")
        return BinanceLivePreflightReport(
            ok=False,
            config=config,
            blocking_reasons=tuple(blocking),
        )

    # ── Gate 2: must NOT be signal-only ─────────────────────────────────
    if config.signal_only:
        blocking.append("binance_signal_only_enabled")
        return BinanceLivePreflightReport(
            ok=False,
            config=config,
            blocking_reasons=tuple(blocking),
        )

    # ── Gate 3: BINANCE_LIVE_ENABLED must be truthy ─────────────────────
    if not config.live_enabled:
        blocking.append("binance_live_enabled_not_true")

    # ── Gate 4: BINANCE_LIVE_ALLOW_ORDERS must be truthy ────────────────
    if not config.allow_orders:
        blocking.append("binance_live_allow_orders_not_true")

    # ── Gate 5: LIVE_CONFIRMATION must match an accepted phrase ─────────
    if config.confirmation not in _ACCEPTED_CONFIRMATION_PHRASES:
        blocking.append("binance_live_confirmation_missing_or_invalid")

    # ── Gate 6: LIVE_MAX_ORDER_NOTIONAL_USDT — must exist and be > 0 ─────
    if config.max_order_notional_usdt is None or not (config.max_order_notional_usdt > Decimal("0")):
        blocking.append("binance_live_max_order_notional_invalid")

    # ── Gate 7: LIVE_MAX_POSITION_NOTIONAL_USDT — must exist and be > 0 ──
    if config.max_position_notional_usdt is None or not (config.max_position_notional_usdt > Decimal("0")):
        blocking.append("binance_live_max_position_notional_invalid")

    # ── Gate 8: BINANCE_LIVE_LEVERAGE ───────────────────────────────────
    if config.leverage is None or not (
        1 <= config.leverage <= BINANCE_LIVE_HARD_MAX_LEVERAGE
    ):
        blocking.append("binance_live_leverage_invalid")

    # ── Gate 9: code-level orders gate ──────────────────────────────────
    if not orders_globally_enabled:
        blocking.append("binance_live_orders_disabled_by_build")

    return BinanceLivePreflightReport(
        ok=len(blocking) == 0,
        config=config,
        blocking_reasons=tuple(blocking),
    )


def format_binance_live_blocked_message(
    report: BinanceLivePreflightReport,
) -> str:
    """Format a human-readable blocked message from a preflight report.

    The message never includes secret values, API keys, or credentials.
    """
    reasons: str = ",".join(report.blocking_reasons)
    return (
        "Binance live trading runtime is not wired yet. "
        "Set SIGNAL_ONLY=true for signal-only observation. "
        f"Binance live preflight blocking_reasons={reasons}"
    )
