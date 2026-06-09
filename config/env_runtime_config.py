#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment runtime configuration module.

Defines ``EnvRuntimeConfig`` — a dataclass that holds startup-orchestration,
global-path, and privacy-related configuration sourced from environment
variables (``.env`` or ``os.environ``).

This module follows the **Config Object / Loader** pattern:

* ``EnvRuntimeConfig`` is a pure DTO — no I/O, no env reads, no side effects.
* ``load_env_runtime_config()`` is the single entry point that reads env vars.
* Importing this module does **not** read ``os.environ``.

Design rules:
  - No file I/O.
  - No network I/O.
  - No logging / print.
  - No threads or async tasks.
  - No live-runtime coupling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

# Currently only ETH-USDT-SWAP is supported by the validator, Trader and
# supervisor.  Expanding this set must be done deliberately.
_ALLOWED_SYMBOLS: frozenset[str] = frozenset({"ETH-USDT-SWAP"})

_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off", ""})


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvRuntimeConfig:
    """Global runtime / orchestration / privacy configuration.

    This dataclass is intentionally **not** responsible for per-symbol
    strategy parameters — those live in ``SymbolConfig``.
    """

    run_mode: str
    """Startup run-mode identifier (e.g. ``"live"``, ``"dry-run"``)."""

    symbols: tuple[str, ...]
    """Ordered list of symbol instrument IDs to run (e.g. ``("ETH-USDT-SWAP",)``)."""

    symbol_config_dir: Path
    """Directory where per-symbol TOML configuration files live."""

    runtime_dir: Path
    """Directory for runtime artifacts (logs, state, telemetry snapshots)."""

    use_symbol_toml: bool
    """Whether the runtime should load symbol config from TOML files."""

    # -- OKX credentials (optional — may be absent in dry-run / test) ---------

    okx_api_key: str | None
    okx_secret_key: str | None
    okx_passphase: str | None

    # -- Email alerting -------------------------------------------------------

    email_enabled: bool
    smtp_host: str | None
    smtp_user: str | None
    smtp_password: str | None
    alert_email_to: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    """Retrieve a string value from *env*, returning *default* if missing."""
    return env.get(key, default)


def _parse_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    """Parse a boolean-like env value.

    Accepted true values:  ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Accepted false values: ``0``, ``false``, ``no``, ``off``, ``""`` (case-insensitive).

    Raises:
        ValueError: If the value is set but not recognised.
    """
    raw = _get(env, key)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean value for {key!r}: {raw!r}. "
        f"Accepted (case-insensitive): true={sorted(_TRUE_VALUES)!r}, "
        f"false={sorted(_FALSE_VALUES)!r}"
    )


def _parse_symbols(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated symbol list into a deduplicated ordered tuple.

    Raises:
        ValueError: If the list is empty, contains duplicates, or includes a
                    symbol that is not yet allowed by the runtime validator.
    """
    parts = [s.strip() for s in raw.split(",")]
    parts = [s for s in parts if s]  # drop empty entries

    if not parts:
        raise ValueError(
            f"RECLAIM_SYMBOLS must contain at least one symbol, got: {raw!r}"
        )

    seen: set[str] = set()
    unique: list[str] = []
    for sym in parts:
        if sym in seen:
            raise ValueError(
                f"RECLAIM_SYMBOLS contains duplicate symbol {sym!r}: {raw!r}"
            )
        seen.add(sym)
        unique.append(sym)

    result = tuple(unique)

    # -- gate: only allow the currently-supported symbol(s) -------------------
    for sym in result:
        if sym not in _ALLOWED_SYMBOLS:
            raise ValueError(
                f"RECLAIM_SYMBOLS contains unsupported symbol {sym!r}. "
                f"Currently allowed: {sorted(_ALLOWED_SYMBOLS)!r}"
            )

    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_env_runtime_config(
    env: Mapping[str, str] | None = None,
) -> EnvRuntimeConfig:
    """Load global runtime configuration from environment variables.

    Args:
        env: Optional explicit mapping of environment variables (e.g. for
             testing).  When ``None``, ``os.environ`` is read.

    Returns:
        A fully-populated ``EnvRuntimeConfig`` instance.

    .. note::
        This function does **not** read ``.env`` files, create directories,
        or perform filesystem existence checks.
    """
    if env is None:
        env = os.environ

    # -- startup orchestration -------------------------------------------------
    run_mode = _get(env, "RECLAIM_RUN_MODE", "live")
    if not run_mode or not run_mode.strip():
        raise ValueError("RECLAIM_RUN_MODE must be a non-empty string")

    raw_symbols = _get(env, "RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
    symbols = _parse_symbols(raw_symbols)

    symbol_config_dir_raw = _get(env, "RECLAIM_SYMBOL_CONFIG_DIR", "config/symbols")
    symbol_config_dir = Path(symbol_config_dir_raw)

    runtime_dir_raw = _get(env, "RECLAIM_RUNTIME_DIR", "runtime")
    runtime_dir = Path(runtime_dir_raw)

    use_symbol_toml = _parse_bool(env, "RECLAIM_USE_SYMBOL_TOML", default=False)

    # -- OKX credentials -------------------------------------------------------
    okx_api_key = _get(env, "OKX_API_KEY")
    okx_secret_key = _get(env, "OKX_SECRET_KEY")
    okx_passphase = _get(env, "OKX_PASSPHASE")  # deliberate: matches project usage

    # -- email alerting --------------------------------------------------------
    email_enabled = _parse_bool(env, "EMAIL_ENABLED", default=False)
    smtp_host = _get(env, "SMTP_HOST")
    smtp_user = _get(env, "SMTP_USER")
    smtp_password = _get(env, "SMTP_PASSWORD")
    alert_email_to = _get(env, "ALERT_EMAIL_TO")

    return EnvRuntimeConfig(
        run_mode=run_mode,
        symbols=symbols,
        symbol_config_dir=symbol_config_dir,
        runtime_dir=runtime_dir,
        use_symbol_toml=use_symbol_toml,
        okx_api_key=okx_api_key,
        okx_secret_key=okx_secret_key,
        okx_passphase=okx_passphase,
        email_enabled=email_enabled,
        smtp_host=smtp_host,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        alert_email_to=alert_email_to,
    )
