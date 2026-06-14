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


class LiveRuntimeKind(str, Enum):
    """Kinds of live runtime the selector can choose."""

    OKX_LEGACY = "okx_legacy"
    BINANCE_LIVE_BLOCKED = "binance_live_blocked"


@dataclass(frozen=True)
class LiveRuntimeSelection:
    """Result of selecting a live runtime.

    Attributes:
        kind: Which runtime to launch.
        exchange: Normalised exchange name (``okx`` or ``binance``).
        reason: Human-readable explanation of the decision.
    """

    kind: LiveRuntimeKind
    exchange: str
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
            reason="okx_legacy_default",
        )

    # ── Binance path (always blocked) ───────────────────────────────────
    if exchange == "binance":
        return LiveRuntimeSelection(
            kind=LiveRuntimeKind.BINANCE_LIVE_BLOCKED,
            exchange="binance",
            reason="binance_live_not_wired",
        )

    # ── Unsupported exchange ────────────────────────────────────────────
    raise ValueError(
        f"Unsupported exchange: {exchange!r}. "
        f"Supported values are 'okx' and 'binance'."
    )
