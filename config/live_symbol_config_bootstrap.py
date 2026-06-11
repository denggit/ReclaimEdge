#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live entrypoint symbol-config bootstrapper (A07 / A08).

Provides a single pure-startup helper that builds runtime config objects
for the live entrypoint.  As of A08 the **default** path is the TOML-based
symbol config; the legacy ``.env``-based path is retained as an explicit
opt-out (``RECLAIM_USE_SYMBOL_TOML=false``).

Design rules
------------
* Startup-only – never call from tick / worker / strategy loop.
* No file I/O outside the single TOML load (and only when the TOML path
  is active).
* No network I/O.
* No logging / print.
* No threads / async tasks.
* Never instantiates monitor, CVD tracker, position sizer, strategy or
  Trader objects — only their *config* dataclasses.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping

from config.env_runtime_config import EnvRuntimeConfig, load_env_runtime_config
from config.symbol_config import SymbolConfig
from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_mapper import map_symbol_config
from config.symbol_config_validator import validate_symbol_config
from src.indicators.cvd_tracker import CvdTrackerConfig
from src.monitors.boll_band_breakout_monitor import BollBandBreakoutMonitorConfig
from src.risk.simple_position_sizer import SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig

# ---------------------------------------------------------------------------
# Supported live TOML symbols (worker-level, single-symbol only)
# ---------------------------------------------------------------------------

SUPPORTED_LIVE_SYMBOL_TOML_SYMBOLS: frozenset[str] = frozenset({
    "ETH-USDT-SWAP",
    "BTC-USDT-SWAP",
})

# ---------------------------------------------------------------------------
# Internal: temporary environ patch (test-safe, scope-guaranteed restore)
# ---------------------------------------------------------------------------


@contextmanager
def _temporary_environ(env: Mapping[str, str] | None) -> Iterator[None]:
    """Temporarily replace ``os.environ`` with *env* for ``from_env()`` calls.

    When *env* is ``None`` this is a no-op — real ``os.environ`` is used.
    When *env* is a mapping the current ``os.environ`` is backed up,
    replaced, and then restored on exit (including after exceptions).

    .. warning::
       This context manager exists **only** to let the bootstrap helper
       invoke ``from_env()`` class-methods during startup.  It must never
       be used on a tick / worker / strategy path.
    """
    if env is None:
        yield
        return

    backup = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(backup)


# ---------------------------------------------------------------------------
# Single-symbol guard (worker-level)
# ---------------------------------------------------------------------------


def require_single_supported_live_symbol(symbols: Sequence[str]) -> str:
    """Validate that *symbols* contains exactly one supported live TOML symbol.

    Returns the normalised (stripped) symbol string on success.

    Raises ``ValueError`` with a descriptive message when:
    * *symbols* is empty
    * *symbols* contains more than one entry (worker must be single-symbol)
    * the symbol is not in ``SUPPORTED_LIVE_SYMBOL_TOML_SYMBOLS``
    """
    if not symbols:
        raise ValueError(
            "RECLAIM_SYMBOLS must contain exactly one symbol for worker "
            "TOML bootstrap, but the symbol list is empty. "
            "Supported symbols: "
            f"{sorted(SUPPORTED_LIVE_SYMBOL_TOML_SYMBOLS)!r}"
        )

    if len(symbols) > 1:
        raise ValueError(
            "Worker TOML bootstrap requires exactly one symbol per worker. "
            f"Got {len(symbols)} symbols: {list(symbols)!r}. "
            "Split multi-symbol configs across separate workers "
            "(e.g. via the supervisor). "
            "Supported symbols: "
            f"{sorted(SUPPORTED_LIVE_SYMBOL_TOML_SYMBOLS)!r}"
        )

    symbol = symbols[0].strip()

    if symbol not in SUPPORTED_LIVE_SYMBOL_TOML_SYMBOLS:
        raise ValueError(
            f"Unsupported live TOML symbol {symbol!r}. "
            "Worker TOML bootstrap currently supports: "
            f"{sorted(SUPPORTED_LIVE_SYMBOL_TOML_SYMBOLS)!r}. "
            f"Got: {list(symbols)!r}"
        )

    return symbol


# ---------------------------------------------------------------------------
# Public DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSymbolRuntimeConfigs:
    """Bundle of all runtime config objects needed by the live entrypoint.

    This is a pure DTO — no I/O, no side-effects.
    """

    env_runtime: EnvRuntimeConfig
    """Global runtime / orchestration config sourced from env vars."""

    symbol_config: SymbolConfig | None
    """The loaded ``SymbolConfig`` when ``use_symbol_toml`` is ``True``;
    ``None`` when running on the legacy ``.env`` path."""

    monitor: BollBandBreakoutMonitorConfig
    """Bollinger-band breakout monitor config."""

    cvd: CvdTrackerConfig
    """CVD (cumulative volume delta) tracker config."""

    strategy: BollCvdReclaimStrategyConfig
    """Strategy config for ``BollCvdShockReclaimStrategy``."""

    position_sizer: SimplePositionSizerConfig
    """Position sizer config with optional account-equity override."""


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_live_symbol_runtime_configs(
    *,
    env: Mapping[str, str] | None = None,
    account_equity_usdt: float | None = None,
) -> LiveSymbolRuntimeConfigs:
    """Build the full set of live runtime configs.

    Behaviour is controlled by ``RECLAIM_USE_SYMBOL_TOML`` (read via
    *env* or ``os.environ``):

    * **TOML path** (default, ``RECLAIM_USE_SYMBOL_TOML`` unset or
      ``true``): the worker TOML file for the single symbol in
      ``RECLAIM_SYMBOLS`` is loaded, validated and mapped.
      Supported symbols: ``ETH-USDT-SWAP``, ``BTC-USDT-SWAP``.
      Each worker must run with exactly one symbol.
      ``account_equity_usdt`` overrides ``dry_run_equity_usdt`` when
      provided (preserving current live-startup account-equity semantics).

    * **Legacy path** (``RECLAIM_USE_SYMBOL_TOML=false``): all config
      objects are created via their respective ``.from_env()``
      class-methods — exactly as the live entrypoint did before A08.
      Legacy path continues to support ETH-USDT-SWAP only; BTC must
      use the TOML path.

    Parameters
    ----------
    env : Mapping[str, str] | None
        Optional explicit env-variable mapping (for testing).  When
        ``None``, ``os.environ`` is used.
    account_equity_usdt : float | None
        If set, overrides ``dry_run_equity_usdt`` in the position sizer
        config (both legacy and TOML paths).

    Returns
    -------
    LiveSymbolRuntimeConfigs

    Raises
    ------
    ValueError
        If ``RECLAIM_USE_SYMBOL_TOML=true`` but the symbol list does not
        contain exactly one supported symbol, or the TOML fails validation.
    FileNotFoundError
        If ``RECLAIM_USE_SYMBOL_TOML=true`` but the TOML file for the
        requested symbol is missing.
    """
    # -- 1. Load env-runtime config (this reads *env* OR os.environ) -----------
    env_runtime = load_env_runtime_config(env)

    # -- 2. TOML path (default as of A08) -----------------------------------
    if env_runtime.use_symbol_toml:
        # Gate: worker TOML bootstrap requires exactly one supported symbol.
        worker_symbol = require_single_supported_live_symbol(env_runtime.symbols)

        symbol_config = load_symbol_config_from_dir(
            env_runtime.symbol_config_dir,
            worker_symbol,
        )
        validate_symbol_config(symbol_config)
        mapped = map_symbol_config(symbol_config)

        position_sizer = mapped.position_sizer
        if account_equity_usdt is not None:
            position_sizer = replace(
                mapped.position_sizer,
                dry_run_equity_usdt=account_equity_usdt,
            )

        return LiveSymbolRuntimeConfigs(
            env_runtime=env_runtime,
            symbol_config=symbol_config,
            monitor=mapped.monitor,
            cvd=mapped.cvd,
            strategy=mapped.strategy,
            position_sizer=position_sizer,
        )

    # -- 3. Legacy path (explicit opt-out: RECLAIM_USE_SYMBOL_TOML=false) ---
    # Legacy env-only path continues to support ETH-USDT-SWAP only.
    # BTC live must go through the TOML path (use_symbol_toml=true).
    if env_runtime.symbols != ("ETH-USDT-SWAP",):
        raise ValueError(
            "RECLAIM_USE_SYMBOL_TOML=false (legacy env-only path) only supports "
            'RECLAIM_SYMBOLS="ETH-USDT-SWAP". '
            "BTC-USDT-SWAP must use the TOML path (RECLAIM_USE_SYMBOL_TOML=true). "
            f"Got: {env_runtime.symbols!r}"
        )

    with _temporary_environ(env):
        monitor = BollBandBreakoutMonitorConfig.from_env()
        cvd = CvdTrackerConfig.from_env()
        strategy = BollCvdReclaimStrategyConfig.from_env()
        if account_equity_usdt is not None:
            position_sizer = SimplePositionSizerConfig.from_account_equity(
                account_equity_usdt
            )
        else:
            position_sizer = SimplePositionSizerConfig.from_env()

    return LiveSymbolRuntimeConfigs(
        env_runtime=env_runtime,
        symbol_config=None,
        monitor=monitor,
        cvd=cvd,
        strategy=strategy,
        position_sizer=position_sizer,
    )
