#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervisor symbol selection (F04).

This module is responsible for selecting which symbols the single-child
supervisor should launch — purely based on per-symbol TOML configuration,
not on runtime trading state.

Design rules:
  - No trading-executor import.
  - No symbol worker / factory import.
  - No entry/exit/take-profit/sidecar/DME import.
  - No network I/O.
  - No file writing.
  - No email sending.
  - No logging / print.
  - Only reads TOML files via the existing config loader and validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.symbol_config import SymbolConfig
from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_validator import validate_symbol_config


# ---------------------------------------------------------------------------
# Data object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorSymbolSelection:
    """Immutable result of selecting enabled symbols for the supervisor.

    Attributes:
        requested_symbols: All symbols parsed from ``RECLAIM_SYMBOLS``, in
            declaration order.
        enabled_symbols: Symbols whose TOML ``enabled`` is ``true``.
        skipped_disabled_symbols: Symbols whose TOML ``enabled`` is ``false``.
    """

    requested_symbols: tuple[str, ...]
    enabled_symbols: tuple[str, ...]
    skipped_disabled_symbols: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_enabled_supervisor_symbols(
    *,
    symbols: tuple[str, ...],
    symbol_config_dir: Path,
) -> SupervisorSymbolSelection:
    """Select which of the requested *symbols* are enabled in their TOML configs.

    For each symbol:
    1. Load ``SymbolConfig`` from ``<symbol_config_dir>/<symbol>.toml``.
    2. Run ``validate_symbol_config()`` on the loaded config.
    3. If ``config.is_enabled`` is ``True`` → add to ``enabled_symbols``.
    4. Otherwise → add to ``skipped_disabled_symbols``.

    Args:
        symbols: Non-empty ordered tuple of instrument IDs (e.g.
            ``("ETH-USDT-SWAP", "BTC-USDT-SWAP")``).
        symbol_config_dir: Directory containing ``<inst_id>.toml`` files.

    Returns:
        ``SupervisorSymbolSelection`` with the classified symbols.

    Raises:
        ValueError: If *symbols* is empty.
        FileNotFoundError: If a TOML file is missing for a requested symbol.
        SymbolConfigValidationError: If a loaded config fails validation.
    """
    if not symbols:
        raise ValueError("symbols must not be empty")

    enabled: list[str] = []
    skipped: list[str] = []

    for sym in symbols:
        config: SymbolConfig = load_symbol_config_from_dir(symbol_config_dir, sym)
        validate_symbol_config(config)

        if config.is_enabled:
            enabled.append(sym)
        else:
            skipped.append(sym)

    return SupervisorSymbolSelection(
        requested_symbols=symbols,
        enabled_symbols=tuple(enabled),
        skipped_disabled_symbols=tuple(skipped),
    )


def require_single_enabled_symbol(
    selection: SupervisorSymbolSelection,
) -> str:
    """Return the single enabled symbol, or raise if 0 or >1 are enabled.

    This enforces the single-child supervisor invariant.  Once the
    supervisor is upgraded to multi-child, this check can be relaxed.

    Args:
        selection: Result from ``select_enabled_supervisor_symbols()``.

    Returns:
        The single enabled symbol instrument ID (e.g. ``"ETH-USDT-SWAP"``).

    Raises:
        RuntimeError: If zero or more than one symbols are enabled.
    """
    enabled = selection.enabled_symbols

    if len(enabled) == 0:
        raise RuntimeError(
            "No enabled symbols selected for supervisor. "
            f"requested_symbols={selection.requested_symbols!r}, "
            f"enabled_symbols={selection.enabled_symbols!r}, "
            f"skipped_disabled_symbols={selection.skipped_disabled_symbols!r}"
        )
    if len(enabled) > 1:
        raise RuntimeError(
            "Multiple enabled symbols are not supported by single-child "
            "supervisor. "
            f"requested_symbols={selection.requested_symbols!r}, "
            f"enabled_symbols={selection.enabled_symbols!r}, "
            f"skipped_disabled_symbols={selection.skipped_disabled_symbols!r}"
        )

    return enabled[0]
