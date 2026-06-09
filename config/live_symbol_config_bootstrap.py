#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live entrypoint symbol-config bootstrapper (A07).

Provides a single pure-startup helper that chooses between the legacy
``.env``-based config path and the new TOML-based symbol config path,
controlled by ``RECLAIM_USE_SYMBOL_TOML``.

Design rules
------------
* Startup-only – never call from tick / worker / strategy loop.
* No file I/O outside the single TOML load (and only when the feature flag
  is enabled).
* No network I/O.
* No logging / print.
* No threads / async tasks.
* Never instantiates monitor, CVD tracker, position sizer, strategy or
  Trader objects — only their *config* dataclasses.
"""

from __future__ import annotations

import os
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

    * **Legacy path** (default, ``RECLAIM_USE_SYMBOL_TOML`` unset or
      ``false``): all config objects are created via their respective
      ``.from_env()`` class-methods — exactly as the current live
      entrypoint does today.

    * **TOML path** (``RECLAIM_USE_SYMBOL_TOML=true``): the ETH TOML
      file is loaded, validated and mapped.  ``account_equity_usdt``
      overrides ``dry_run_equity_usdt`` when provided (preserving current
      live-startup account-equity semantics).

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
        If ``RECLAIM_USE_SYMBOL_TOML=true`` but the symbol list is not
        exactly ``("ETH-USDT-SWAP",)``, or the TOML fails validation.
    FileNotFoundError
        If ``RECLAIM_USE_SYMBOL_TOML=true`` but the ETH TOML file is
        missing.
    """
    # -- 1. Load env-runtime config (this reads *env* OR os.environ) -----------
    env_runtime = load_env_runtime_config(env)

    # -- 2. Legacy path --------------------------------------------------------
    if not env_runtime.use_symbol_toml:
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

    # -- 3. TOML path ----------------------------------------------------------
    # Gate: only ETH-USDT-SWAP is allowed for now.
    if env_runtime.symbols != ("ETH-USDT-SWAP",):
        raise ValueError(
            "RECLAIM_USE_SYMBOL_TOML=true currently only supports "
            'RECLAIM_SYMBOLS="ETH-USDT-SWAP". '
            f"Got: {env_runtime.symbols!r}"
        )

    symbol_config = load_symbol_config_from_dir(
        env_runtime.symbol_config_dir,
        "ETH-USDT-SWAP",
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
