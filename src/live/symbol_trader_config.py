#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Trader configuration objects from ``SymbolConfig``.

This module provides pure functions that map a validated ``SymbolConfig``
(typically loaded from a per-symbol TOML file) into the types that
``Trader`` consumes at construction time:

* :class:`~src.execution.trader_types.TraderInstrumentMetadata`
* :class:`~src.execution.trader_types.TraderMarketSettings`

----
Design
----

* **Pure functions** — no I/O, no env, no network, no ``Trader`` import.
* **No fallback** — every required field must be present on the config;
  if a field is missing or invalid the caller gets a clear ``ValueError``.
* **Single symbol** — each call handles exactly one ``SymbolConfig`` and
  produces metadata / settings for *that* symbol only.
"""

from __future__ import annotations

from config.symbol_config import SymbolConfig
from src.execution.trader_types import TraderInstrumentMetadata, TraderMarketSettings


def build_trader_instrument_metadata(
    symbol_config: SymbolConfig,
) -> TraderInstrumentMetadata:
    """Build a :class:`TraderInstrumentMetadata` from a TOML-backed
    :class:`~config.symbol_config.SymbolConfig`.

    Mapping
    -------
    =====================================  ===================================
    ``TraderInstrumentMetadata`` field      ``SymbolConfig`` source
    =====================================  ===================================
    ``inst_id``                            ``symbol_config.symbol.inst_id``
    ``contract_multiplier``                ``symbol_config.market.contract_value``
    ``contract_precision``                 ``symbol_config.market.contract_precision``
    ``min_contracts``                      ``symbol_config.market.min_contracts``
    =====================================  ===================================
    """
    return TraderInstrumentMetadata(
        inst_id=symbol_config.symbol.inst_id,
        contract_multiplier=symbol_config.market.contract_value,
        contract_precision=symbol_config.market.contract_precision,
        min_contracts=symbol_config.market.min_contracts,
    )


def build_trader_market_settings(
    symbol_config: SymbolConfig,
) -> TraderMarketSettings:
    """Build a :class:`TraderMarketSettings` from a TOML-backed
    :class:`~config.symbol_config.SymbolConfig`.

    Mapping
    -------
    =====================================  ===================================
    ``TraderMarketSettings`` field          ``SymbolConfig`` source
    =====================================  ===================================
    ``inst_id``                            ``symbol_config.symbol.inst_id``
    ``td_mode``                            ``symbol_config.market.td_mode``
    ``pos_side_mode``                      ``symbol_config.market.pos_side_mode``
    ``leverage``                           ``symbol_config.capital.leverage``
    =====================================  ===================================
    """
    return TraderMarketSettings(
        inst_id=symbol_config.symbol.inst_id,
        td_mode=symbol_config.market.td_mode,
        pos_side_mode=symbol_config.market.pos_side_mode,
        leverage=symbol_config.capital.leverage,
    )
