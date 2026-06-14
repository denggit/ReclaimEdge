#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : live_runtime_selector.py
@Description: Pure selector that decides which live runtime to launch.

This module is intentionally free of side-effects:
- No dotenv loading
- No network connections
- No API key reading
- No imports of strategy / execution / exchange clients / order placement
- No imports of OKX / Binance broker / signed REST
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Public env var keys for signal-only gating.
# PRIMARY is the exchange-neutral name; ALIAS is the legacy Binance name
# kept for backward compatibility.  Boundary tests reference both.
# These are NOT secrets — they are public env var names.
# ---------------------------------------------------------------------------
_SIGNAL_ONLY_KEY: str = "SIGNAL_ONLY"
_SIGNAL_ONLY_ALIAS_KEY: str = "BINANCE_SIGNAL_ONLY"

# Set of values recognised as boolean *true* for SIGNAL_ONLY.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "y", "on"})


class LiveRuntimeKind(str, Enum):
    """Kinds of live runtime the selector can choose."""

    OKX_LEGACY = "okx_legacy"
    BINANCE_SIGNAL_ONLY = "binance_signal_only"
    BINANCE_LIVE_BLOCKED = "binance_live_blocked"


@dataclass(frozen=True)
class LiveRuntimeSelection:
    """Result of selecting a live runtime.

    Attributes:
        kind: Which runtime to launch.
        exchange: Normalised exchange name (``okx`` or ``binance``).
        signal_only: Whether this is a signal-only (no-order) path.
        reason: Human-readable explanation of the decision.
    """

    kind: LiveRuntimeKind
    exchange: str
    signal_only: bool
    reason: str


def select_live_runtime(
    env: Mapping[str, str] | None = None,
) -> LiveRuntimeSelection:
    """Select a live runtime based on environment variables.

    Parameters
    ----------
    env:
        Optional mapping of environment variables.  When ``None`` (the
        default) the real ``os.environ`` is used — this makes the function
        ergonomic for production while keeping it testable.

    Returns
    -------
    LiveRuntimeSelection
        A frozen selection result.

    Raises
    ------
    ValueError
        When ``EXCHANGE`` names an unsupported exchange.
    """
    if env is None:
        env = os.environ

    raw_exchange: str = env.get("EXCHANGE", "okx").strip().lower()
    exchange: str = raw_exchange if raw_exchange else "okx"

    # ── OKX path (default) ──────────────────────────────────────────────
    if exchange == "okx":
        return LiveRuntimeSelection(
            kind=LiveRuntimeKind.OKX_LEGACY,
            exchange="okx",
            signal_only=False,
            reason="okx_legacy_default",
        )

    # ── Binance path ────────────────────────────────────────────────────
    if exchange == "binance":
        signal_only_primary: str = env.get(_SIGNAL_ONLY_KEY, "").strip()
        signal_only_alias: str = env.get(_SIGNAL_ONLY_ALIAS_KEY, "").strip()

        # ── Conflict detection ─────────────────────────────────────────
        if (
            signal_only_primary
            and signal_only_alias
            and signal_only_primary.lower() != signal_only_alias.lower()
        ):
            raise ValueError(
                f"Conflicting env vars: {_SIGNAL_ONLY_KEY}={signal_only_primary!r} vs "
                f"{_SIGNAL_ONLY_ALIAS_KEY}={signal_only_alias!r}. "
                f"Use only {_SIGNAL_ONLY_KEY}."
            )

        signal_only_raw: str = (
            signal_only_primary if signal_only_primary else signal_only_alias
        ).lower()
        signal_only: bool = signal_only_raw in _TRUTHY

        if signal_only:
            return LiveRuntimeSelection(
                kind=LiveRuntimeKind.BINANCE_SIGNAL_ONLY,
                exchange="binance",
                signal_only=True,
                reason="binance_signal_only",
            )

        return LiveRuntimeSelection(
            kind=LiveRuntimeKind.BINANCE_LIVE_BLOCKED,
            exchange="binance",
            signal_only=False,
            reason="binance_live_not_wired",
        )

    # ── Unsupported exchange ────────────────────────────────────────────
    raise ValueError(
        f"Unsupported exchange: {exchange!r}. "
        f"Supported values are 'okx' and 'binance'."
    )
