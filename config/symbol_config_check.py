#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pure config-check module for per-symbol TOML configuration (F05).

Loads, validates, and maps a ``SymbolConfig`` from a single TOML file
and returns a ``SymbolConfigCheckResult`` — a lightweight preview DTO.

Design rules
------------
* Read-only: loads a TOML file, calls validator, calls mapper.
* No Trader, no SymbolWorkerApp, no ReclaimSupervisor, no OKX client.
* No EmailSender, no network I/O, no file writes, no .env reads.
* No ``os.environ`` mutations.
* No asyncio, no httpx, no requests, no websocket imports.
* This is a **config-check / dry-run preview** only — it does not
  start a worker, does not enable a symbol, and does not create
  live trading infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from config.symbol_config import SymbolConfig
from config.symbol_config_loader import load_symbol_config_from_dir
from config.symbol_config_mapper import MappedSymbolConfigs, map_symbol_config
from config.symbol_config_validator import validate_symbol_config


@dataclass(frozen=True)
class SymbolConfigCheckResult:
    """Result of a dry-run config-check for a single symbol.

    This is a pure data-transfer object — it does not instantiate any
    live component and must never be wired into a ``Trader`` or worker.
    """

    inst_id: str
    enabled: bool
    live_trading: bool
    contract_value: Decimal
    min_contracts: Decimal
    contract_precision: Decimal
    price_precision: Decimal
    sidecar_enabled: bool
    middle_bucket_split_enabled: bool
    mapped: MappedSymbolConfigs

    @property
    def safe_for_config_check_only(self) -> bool:
        """Return ``True`` when live_trading is ``False``.

        A config-check result is **always** safe to inspect — it does
        not start a worker or connect to OKX.  This property exists so
        callers can assert the invariant explicitly without depending on
        knowledge of the internal implementation.
        """
        return self.live_trading is False

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary dict for CLI / test output."""
        tp = self.mapped.trader_preview
        return {
            "inst_id": self.inst_id,
            "enabled": self.enabled,
            "live_trading": self.live_trading,
            "safe_for_config_check_only": self.safe_for_config_check_only,
            "contract_value": str(self.contract_value),
            "min_contracts": str(self.min_contracts),
            "contract_precision": str(self.contract_precision),
            "price_precision": str(self.price_precision),
            "sidecar_enabled": self.sidecar_enabled,
            "middle_bucket_split_enabled": self.middle_bucket_split_enabled,
            "trader_preview": {
                "inst_id": tp.inst_id,
                "td_mode": tp.td_mode,
                "pos_side_mode": tp.pos_side_mode,
                "leverage": tp.leverage,
                "contract_value": str(tp.contract_value),
                "contract_precision": str(tp.contract_precision),
                "min_contracts": str(tp.min_contracts),
                "live_trading": tp.live_trading,
            },
        }


def _load_config(*, symbol_config_dir: Path, inst_id: str) -> SymbolConfig:
    """Load a single symbol config from disk."""
    if not inst_id or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
    return load_symbol_config_from_dir(symbol_config_dir, inst_id.strip())


def check_symbol_config(
    *,
    symbol_config_dir: Path,
    inst_id: str,
) -> SymbolConfigCheckResult:
    """Load, validate, and map a single symbol TOML config.

    This function is a **config-check / dry-run preview** only:
    * It reads the TOML file named ``<inst_id>.toml`` from *symbol_config_dir*.
    * It runs the pure ``validate_symbol_config`` validator.
    * It runs the pure ``map_symbol_config`` mapper.
    * It returns a frozen ``SymbolConfigCheckResult`` DTO.

    It does **not**:
    * Start a worker or supervisor child process.
    * Instantiate ``SymbolWorkerApp``, ``Trader``, or any live component.
    * Connect to OKX (public or private).
    * Place orders.
    * Send email.
    * Write runtime state.
    * Enable a symbol or modify its TOML file.
    * Read ``.env`` or modify ``os.environ``.

    Parameters
    ----------
    symbol_config_dir : Path
        Directory containing ``.toml`` files named by instrument ID.
    inst_id : str
        Instrument ID to check (e.g. ``"BTC-USDT-SWAP"``).

    Returns
    -------
    SymbolConfigCheckResult

    Raises
    ------
    FileNotFoundError
        If the TOML file does not exist.
    ValueError
        If *inst_id* is empty or the config fails validation.
    TypeError
        If a TOML field has the wrong type.
    """
    inst_id = inst_id.strip()
    if not inst_id:
        raise ValueError("inst_id must be a non-empty string")

    config = load_symbol_config_from_dir(symbol_config_dir, inst_id)
    validate_symbol_config(config)
    mapped = map_symbol_config(config)

    return SymbolConfigCheckResult(
        inst_id=config.inst_id,
        enabled=config.symbol.enabled,
        live_trading=config.symbol.live_trading,
        contract_value=config.market.contract_value,
        min_contracts=config.market.min_contracts,
        contract_precision=config.market.contract_precision,
        price_precision=config.market.price_precision,
        sidecar_enabled=config.sidecar.enabled,
        middle_bucket_split_enabled=config.middle_bucket_split.enabled,
        mapped=mapped,
    )
