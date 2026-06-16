"""Sidecar runtime removed.

Only legacy type definitions (PositionSide, SidecarLegStatus) remain
for backward compatibility with non-sidecar code that references them.
All runtime functions have been removed.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

PositionSide = Literal["LONG", "SHORT"]


class SidecarLegStatus(str, Enum):
    """Legacy enum — sidecar runtime removed. Retained for backward compat."""
    OPEN = "OPEN"
    OPEN_UNPROTECTED = "OPEN_UNPROTECTED"
    TP_FILLED = "TP_FILLED"
    CLOSED = "CLOSED"
    FORCE_CLOSED = "FORCE_CLOSED"
    UNKNOWN = "UNKNOWN"
    UNKNOWN_HALTED = "UNKNOWN_HALTED"


# ── Stubs: removed runtime functions return safe defaults ──────────────

def calculate_core_margin_pct(layer_margin_pct: float, sidecar_enabled: bool, sidecar_margin_pct: float) -> float:
    """Stub — sidecar runtime removed. Returns layer_margin_pct unchanged."""
    return float(layer_margin_pct)


def sidecar_open_qty(legs: list[dict] | list) -> float:
    """Stub — sidecar runtime removed. Returns 0.0."""
    return 0.0


def sidecar_open_contracts(legs: list[dict] | list) -> float:
    """Stub — sidecar runtime removed. Returns 0.0."""
    return 0.0


def calculate_sidecar_margin(sidecar_margin_pct: float, layer_multiplier: float) -> float:
    """Stub — sidecar runtime removed. Returns 0.0."""
    return 0.0


def sanitize_okx_client_order_id(raw: str) -> str:
    """Stub — sidecar runtime removed. Returns input unchanged."""
    return raw


def trim_sidecar_legs_for_state(legs: list[dict], max_legs: int) -> list[dict]:
    """Stub — sidecar runtime removed. Returns empty list."""
    return []
