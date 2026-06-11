#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-symbol live startup preflight (G09d).

Provides a read-only safety check that verifies ETH/BTC multi-worker live
configuration is internally consistent **before** the supervisor launches any
real Trader or connects to OKX.

Design rules
------------
* Read-only — never starts a Trader, never connects to OKX, never places orders.
* Only imports pure functions from config and startup layers.
* Never instantiates Trader, never calls OKX private/public client, never opens
  websocket connections.
* Each worker is checked in isolation with its own single-symbol env.
* Does not modify config files, env vars, or runtime state.
* Does not change strategy, tick path, portfolio allocator, or ledger.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from config.env_runtime_config import EnvRuntimeConfig, load_env_runtime_config
from config.live_symbol_config_bootstrap import (
    LiveSymbolRuntimeConfigs,
    build_live_symbol_runtime_configs,
)
from config.symbol_config import SymbolConfig
from src.execution.trader import parse_allowed_live_symbols
from src.execution.trader_types import TraderInstrumentMetadata, TraderMarketSettings
from src.live.runtime_paths import RuntimePaths, build_runtime_paths
from src.live.supervisor.symbol_selection import (
    SupervisorSymbolSelection,
    select_enabled_supervisor_symbols,
)
from src.live.supervisor.symbol_worker_plan import (
    VALID_WORKER_MODES,
    SymbolWorkerPlan,
    build_symbol_worker_plans,
    parse_worker_modes,
    validate_supported_supervisor_symbol,
)
from src.live.symbol_trader_config import (
    build_trader_instrument_metadata,
    build_trader_market_settings,
)
from src.live.worker_logging import sanitize_symbol_for_log_dir

_SUPPORTED_SUPERVISOR_SYMBOLS = frozenset({
    "ETH-USDT-SWAP",
    "BTC-USDT-SWAP",
})


# ---------------------------------------------------------------------------
# Result DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolPreflightResult:
    """Per-symbol preflight check result.

    All fields are immutable — this is a pure data-transfer object.
    """

    symbol: str
    """Instrument ID, e.g. ``"ETH-USDT-SWAP"``."""

    worker_mode: str
    """Worker mode: ``"live"`` or ``"paper"``."""

    enabled: bool
    """Whether the symbol TOML has ``enabled = true``."""

    live_trading: bool | None
    """The ``live_trading`` flag from the TOML ``[symbol]`` block (or None)."""

    child_env_symbol: str
    """The value of ``RECLAIM_SYMBOLS`` in the worker child env (must be single symbol)."""

    okx_inst_id: str
    """The value of ``OKX_INST_ID`` in the worker child env."""

    runtime_dir: Path
    """Parent runtime directory for this worker."""

    heartbeat_path: Path
    """Heartbeat file path for this worker."""

    event_outbox_path: Path | None
    """Worker event outbox path (or None if not set)."""

    log_dir: Path | None
    """Worker log directory (or None if not computed)."""

    metadata_ok: bool
    """``True`` if ``build_trader_instrument_metadata`` succeeded."""

    market_settings_ok: bool
    """``True`` if ``build_trader_market_settings`` succeeded."""

    sidecar_enabled: bool | None
    """Whether the symbol TOML sidecar is enabled (or None if TOML not loaded)."""


@dataclass(frozen=True)
class MultiSymbolLivePreflightResult:
    """Overall multi-symbol live preflight result.

    ``ok`` is ``True`` only when every check passes with no errors.
    Warnings do not make the result fail, but they are always surfaced.
    """

    ok: bool
    requested_symbols: tuple[str, ...]
    enabled_symbols: tuple[str, ...]
    skipped_disabled_symbols: tuple[str, ...]
    worker_results: tuple[SymbolPreflightResult, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _symbol_worker_log_dir(base_log_dir: str, symbol: str) -> Path:
    """Compute the per-symbol worker log directory.

    Mirrors ``configure_symbol_worker_logging_env`` without creating
    directories or mutating the environment.
    """
    safe = sanitize_symbol_for_log_dir(symbol)
    return Path(base_log_dir) / safe


def _check_child_env_single_symbol(
    plan: SymbolWorkerPlan,
) -> list[str]:
    """Verify the child env has exactly one symbol and no comma-separated values."""
    errors: list[str] = []

    symbols = plan.child_env.get("RECLAIM_SYMBOLS", "")
    if not symbols or symbols.strip() == "":
        errors.append(
            f"{plan.symbol}: RECLAIM_SYMBOLS is empty in child env"
        )
    elif "," in symbols:
        errors.append(
            f"{plan.symbol}: RECLAIM_SYMBOLS contains comma in child env: {symbols!r}"
        )
    elif symbols.strip() != plan.symbol:
        errors.append(
            f"{plan.symbol}: RECLAIM_SYMBOLS mismatch in child env: "
            f"got {symbols.strip()!r}, expected {plan.symbol!r}"
        )

    rec_symbol = plan.child_env.get("RECLAIM_SYMBOL", "")
    if rec_symbol.strip() != plan.symbol:
        errors.append(
            f"{plan.symbol}: RECLAIM_SYMBOL mismatch in child env: "
            f"got {rec_symbol.strip()!r}, expected {plan.symbol!r}"
        )

    okx = plan.child_env.get("OKX_INST_ID", "")
    if okx.strip() != plan.symbol:
        errors.append(
            f"{plan.symbol}: OKX_INST_ID mismatch in child env: "
            f"got {okx.strip()!r}, expected {plan.symbol!r}"
        )

    mode = plan.child_env.get("RECLAIM_WORKER_MODE", "")
    if mode.strip().lower() not in VALID_WORKER_MODES:
        errors.append(
            f"{plan.symbol}: RECLAIM_WORKER_MODE invalid in child env: {mode!r}"
        )

    return errors


def _check_path_uniqueness(
    plans: list[SymbolWorkerPlan],
    runtime_dir: Path,
) -> list[str]:
    """Verify runtime paths (heartbeat, event outbox, child name, etc.) do not collide."""
    errors: list[str] = []

    heartbeats: dict[Path, str] = {}
    outboxes: dict[Path, str] = {}
    names: dict[str, str] = {}
    state_files: dict[Path, str] = {}
    journal_files: dict[Path, str] = {}

    for plan in plans:
        # heartbeat
        if plan.heartbeat_path in heartbeats:
            other = heartbeats[plan.heartbeat_path]
            errors.append(
                f"heartbeat_path collision: {plan.symbol} and {other} "
                f"both use {plan.heartbeat_path}"
            )
        else:
            heartbeats[plan.heartbeat_path] = plan.symbol

        # event outbox
        if plan.event_outbox_path is not None:
            if plan.event_outbox_path in outboxes:
                other = outboxes[plan.event_outbox_path]
                errors.append(
                    f"event_outbox_path collision: {plan.symbol} and {other} "
                    f"both use {plan.event_outbox_path}"
                )
            else:
                outboxes[plan.event_outbox_path] = plan.symbol

        # child name
        if plan.child_name in names:
            other = names[plan.child_name]
            errors.append(
                f"child_name collision: {plan.symbol} and {other} "
                f"both use {plan.child_name!r}"
            )
        else:
            names[plan.child_name] = plan.symbol

        # RuntimePaths-derived files
        rp = build_runtime_paths(runtime_dir, plan.symbol)

        if rp.state_file in state_files:
            other = state_files[rp.state_file]
            errors.append(
                f"state_file collision: {plan.symbol} and {other} "
                f"both use {rp.state_file}"
            )
        else:
            state_files[rp.state_file] = plan.symbol

        if rp.journal_file in journal_files:
            other = journal_files[rp.journal_file]
            errors.append(
                f"journal_file collision: {plan.symbol} and {other} "
                f"both use {rp.journal_file}"
            )
        else:
            journal_files[rp.journal_file] = plan.symbol

    return errors


def _check_log_path_uniqueness(
    plans: list[SymbolWorkerPlan],
    base_log_dir: str,
) -> list[str]:
    """Verify per-symbol worker log directories do not collide."""
    errors: list[str] = []
    log_dirs: dict[Path, str] = {}

    for plan in plans:
        log_dir = _symbol_worker_log_dir(base_log_dir, plan.symbol)
        if log_dir in log_dirs:
            other = log_dirs[log_dir]
            errors.append(
                f"log_dir collision: {plan.symbol} and {other} "
                f"both use {log_dir}"
            )
        else:
            log_dirs[log_dir] = plan.symbol

    return errors


def _try_build_metadata_and_settings(
    symbol_config: SymbolConfig,
    symbol: str,
) -> tuple[TraderInstrumentMetadata | None, TraderMarketSettings | None, list[str]]:
    """Try to build Trader metadata and market settings from a SymbolConfig.

    Returns ``(metadata, settings, errors)``.  This function never raises —
    all failures are captured as error strings.
    """
    errors: list[str] = []
    metadata: TraderInstrumentMetadata | None = None
    market_settings: TraderMarketSettings | None = None

    try:
        metadata = build_trader_instrument_metadata(symbol_config)
    except Exception as exc:
        errors.append(
            f"{symbol}: build_trader_instrument_metadata failed: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        market_settings = build_trader_market_settings(symbol_config)
    except Exception as exc:
        errors.append(
            f"{symbol}: build_trader_market_settings failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # Cross-check inst_id
    if metadata is not None and metadata.inst_id != symbol:
        errors.append(
            f"{symbol}: metadata.inst_id mismatch: "
            f"got {metadata.inst_id!r}, expected {symbol!r}"
        )
    if market_settings is not None and market_settings.inst_id != symbol:
        errors.append(
            f"{symbol}: market_settings.inst_id mismatch: "
            f"got {market_settings.inst_id!r}, expected {symbol!r}"
        )

    return metadata, market_settings, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_multi_symbol_live_preflight(
    *,
    env: Mapping[str, str] | None = None,
    strict_requested_symbols: bool = True,
) -> MultiSymbolLivePreflightResult:
    """Run a read-only multi-symbol live startup preflight.

    This function verifies that the environment and per-symbol TOML
    configuration are internally consistent for a multi-worker live launch.
    It **never** instantiates a Trader, connects to OKX, places orders, or
    modifies any state.

    Parameters
    ----------
    env : Mapping[str, str] | None
        Optional explicit environment mapping (for testing).  When ``None``,
        ``os.environ`` is read.
    strict_requested_symbols : bool
        When ``True`` (default), every requested symbol must be enabled in
        its TOML config.  When ``False``, disabled symbols are recorded as
        warnings instead of errors.

    Returns
    -------
    MultiSymbolLivePreflightResult
        Immutable result with ``ok``, per-symbol results, errors, and warnings.
    """
    if env is None:
        env = os.environ

    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Load env-runtime config ──────────────────────────────────────────
    try:
        env_runtime: EnvRuntimeConfig = load_env_runtime_config(env)
    except Exception as exc:
        return MultiSymbolLivePreflightResult(
            ok=False,
            requested_symbols=(),
            enabled_symbols=(),
            skipped_disabled_symbols=(),
            worker_results=(),
            errors=(f"load_env_runtime_config failed: {type(exc).__name__}: {exc}",),
            warnings=(),
        )

    requested_symbols = env_runtime.symbols

    # ── 2. Check LIVE_TRADING ───────────────────────────────────────────────
    raw_live_trading = str(env.get("LIVE_TRADING", "")).strip().lower()
    live_trading_enabled = raw_live_trading in ("1", "true", "yes", "on")

    try:
        worker_modes = parse_worker_modes(env)
    except Exception as exc:
        errors.append(
            f"RECLAIM_WORKER_MODES parse failed: {type(exc).__name__}: {exc}"
        )
        worker_modes = {}
    has_live_worker = any(
        worker_modes.get(sym, "live") == "live"
        for sym in requested_symbols
    ) or (
        not worker_modes and True  # fallback default is "live"
    )

    if has_live_worker and not live_trading_enabled:
        errors.append(
            "LIVE_TRADING is not true but live worker mode is requested"
        )

    # ── 3. Parse RECLAIM_ALLOWED_LIVE_SYMBOLS ───────────────────────────────
    try:
        allowed_live_symbols = parse_allowed_live_symbols(
            env.get("RECLAIM_ALLOWED_LIVE_SYMBOLS")
        )
    except Exception as exc:
        errors.append(
            f"RECLAIM_ALLOWED_LIVE_SYMBOLS parse failed: "
            f"{type(exc).__name__}: {exc}"
        )
        allowed_live_symbols = ()

    for sym in allowed_live_symbols:
        if sym not in _SUPPORTED_SUPERVISOR_SYMBOLS:
            errors.append(
                f"RECLAIM_ALLOWED_LIVE_SYMBOLS contains unsupported "
                f"symbol {sym!r}; supported={sorted(_SUPPORTED_SUPERVISOR_SYMBOLS)!r}"
            )

    # ── 4. Select enabled supervisor symbols ────────────────────────────────
    try:
        selection: SupervisorSymbolSelection = select_enabled_supervisor_symbols(
            symbols=requested_symbols,
            symbol_config_dir=env_runtime.symbol_config_dir,
        )
    except Exception as exc:
        errors.append(
            f"select_enabled_supervisor_symbols failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return MultiSymbolLivePreflightResult(
            ok=False,
            requested_symbols=requested_symbols,
            enabled_symbols=(),
            skipped_disabled_symbols=(),
            worker_results=(),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    enabled_symbols = selection.enabled_symbols
    skipped_disabled = selection.skipped_disabled_symbols

    if not enabled_symbols:
        errors.append(
            "No enabled symbols selected — all requested symbols are disabled "
            "or missing TOML configs"
        )

    if strict_requested_symbols and skipped_disabled:
        errors.append(
            f"strict_requested_symbols=True but some requested symbols are "
            f"disabled: {skipped_disabled!r}"
        )
    elif not strict_requested_symbols and skipped_disabled:
        warnings.append(
            f"Skipped disabled symbols (strict_requested_symbols=False): "
            f"{skipped_disabled!r}"
        )

    # ── 5. Build worker plans ───────────────────────────────────────────────
    runtime_dir = env_runtime.runtime_dir
    heartbeat_dir = runtime_dir / "heartbeats"
    event_dir = runtime_dir / "events"

    try:
        plans = build_symbol_worker_plans(
            list(enabled_symbols),
            base_env=env,
            runtime_dir=runtime_dir,
            heartbeat_dir=heartbeat_dir,
            event_dir=event_dir,
        )
    except Exception as exc:
        errors.append(
            f"build_symbol_worker_plans failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return MultiSymbolLivePreflightResult(
            ok=False,
            requested_symbols=requested_symbols,
            enabled_symbols=enabled_symbols,
            skipped_disabled_symbols=skipped_disabled,
            worker_results=(),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    # ── 6. Check child env single-symbol ────────────────────────────────────
    for plan in plans:
        child_errors = _check_child_env_single_symbol(plan)
        errors.extend(child_errors)

    # ── 7. Check path uniqueness ────────────────────────────────────────────
    path_errors = _check_path_uniqueness(plans, runtime_dir)
    errors.extend(path_errors)

    # ── 8. Check log path uniqueness ────────────────────────────────────────
    base_log_dir = str(env.get("WORKER_LOG_BASE_DIR", "logs/workers"))
    log_errors = _check_log_path_uniqueness(plans, base_log_dir)
    errors.extend(log_errors)

    # ── 9. Per-worker checks ────────────────────────────────────────────────
    worker_results: list[SymbolPreflightResult] = []
    for plan in plans:
        symbol = plan.symbol
        mode = plan.worker_mode

        # ── 9a. Allowlist check (live mode only) ────────────────────────────
        if mode == "live":
            if allowed_live_symbols and symbol not in allowed_live_symbols:
                errors.append(
                    f"{symbol}: worker_mode=live but symbol not in "
                    f"RECLAIM_ALLOWED_LIVE_SYMBOLS={allowed_live_symbols!r}"
                )

        # ── 9b. Bootstrap runtime configs with child env ────────────────────
        runtime_configs: LiveSymbolRuntimeConfigs | None = None
        try:
            runtime_configs = build_live_symbol_runtime_configs(
                env=plan.child_env,
                account_equity_usdt=None,
            )
        except Exception as exc:
            errors.append(
                f"{symbol}: build_live_symbol_runtime_configs failed: "
                f"{type(exc).__name__}: {exc}"
            )

        symbol_config = runtime_configs.symbol_config if runtime_configs is not None else None
        enabled = symbol_config.symbol.enabled if symbol_config is not None else False
        live_trading = symbol_config.symbol.live_trading if symbol_config is not None else None
        sidecar_enabled = symbol_config.sidecar.enabled if symbol_config is not None else None

        # ── 9c. inst_id match ───────────────────────────────────────────────
        if symbol_config is not None and symbol_config.inst_id != symbol:
            errors.append(
                f"{symbol}: symbol_config.inst_id mismatch: "
                f"got {symbol_config.inst_id!r}, expected {symbol!r}"
            )

        # ── 9d. enabled check for live mode ─────────────────────────────────
        if mode == "live" and symbol_config is not None and not symbol_config.symbol.enabled:
            errors.append(
                f"{symbol}: worker_mode=live but TOML symbol.enabled is false"
            )

        # ── 9e. live_trading warning ────────────────────────────────────────
        if mode == "live" and symbol_config is not None and symbol_config.symbol.live_trading is False:
            warnings.append(
                f"{symbol}: worker_mode=live but TOML "
                f"symbol.live_trading is false (live gate is primarily "
                f"LIVE_TRADING + RECLAIM_ALLOWED_LIVE_SYMBOLS)"
            )

        # ── 9f. Build metadata and market settings ──────────────────────────
        metadata_ok = False
        market_settings_ok = False
        if symbol_config is not None:
            metadata, market_settings, meta_errors = _try_build_metadata_and_settings(
                symbol_config, symbol
            )
            metadata_ok = metadata is not None
            market_settings_ok = market_settings is not None
            errors.extend(meta_errors)
        else:
            # No TOML loaded — metadata/settings are N/A
            metadata_ok = False
            market_settings_ok = False

        worker_results.append(
            SymbolPreflightResult(
                symbol=symbol,
                worker_mode=mode,
                enabled=enabled,
                live_trading=live_trading,
                child_env_symbol=plan.child_env.get("RECLAIM_SYMBOLS", ""),
                okx_inst_id=plan.child_env.get("OKX_INST_ID", ""),
                runtime_dir=runtime_dir,
                heartbeat_path=plan.heartbeat_path,
                event_outbox_path=plan.event_outbox_path,
                log_dir=_symbol_worker_log_dir(base_log_dir, symbol),
                metadata_ok=metadata_ok,
                market_settings_ok=market_settings_ok,
                sidecar_enabled=sidecar_enabled,
            )
        )

    ok = len(errors) == 0

    return MultiSymbolLivePreflightResult(
        ok=ok,
        requested_symbols=requested_symbols,
        enabled_symbols=enabled_symbols,
        skipped_disabled_symbols=skipped_disabled,
        worker_results=tuple(worker_results),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
