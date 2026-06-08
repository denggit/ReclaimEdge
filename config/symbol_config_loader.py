#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOML loader for ``SymbolConfig`` (A02).

Reads a per-symbol TOML file and constructs a frozen ``SymbolConfig``
dataclass.  This module is intentionally kept free of business-logic
validation — the validator layer is reserved for A04.

Design rules
------------
* No file / network / env-var I/O on import.
* No logging, no printing, no side-effects.
* Fail-fast: unknown sections/keys and type mismatches raise immediately.
* Single Responsibility: only reads TOML and constructs SymbolConfig.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, get_type_hints

# ---------------------------------------------------------------------------
# TOML library – prefer stdlib tomllib (3.11+), fall back to tomli.
# tomli is declared as a conditional dependency in requirements.txt:
#     tomli>=2.0.0; python_version < "3.11"
# ---------------------------------------------------------------------------
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover – Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from config.symbol_config import (
    SymbolCapitalConfig,
    SymbolConfig,
    SymbolCvdConfig,
    SymbolEntryConfig,
    SymbolExecutionConfig,
    SymbolIdentityConfig,
    SymbolMarketConfig,
    SymbolMiddleBucketSplitConfig,
    SymbolRiskConfig,
    SymbolRuntimeConfig,
    SymbolSidecarConfig,
    SymbolTpConfig,
    decimal_from_any,
)

# ---------------------------------------------------------------------------
# Section → sub-dataclass mapping
# ---------------------------------------------------------------------------

_SECTION_CLASS_MAP: dict[str, type] = {
    "symbol": SymbolIdentityConfig,
    "market": SymbolMarketConfig,
    "capital": SymbolCapitalConfig,
    "entry": SymbolEntryConfig,
    "cvd": SymbolCvdConfig,
    "tp": SymbolTpConfig,
    "middle_bucket_split": SymbolMiddleBucketSplitConfig,
    "sidecar": SymbolSidecarConfig,
    "risk": SymbolRiskConfig,
    "execution": SymbolExecutionConfig,
    "runtime": SymbolRuntimeConfig,
}

_ALLOWED_SECTIONS: frozenset[str] = frozenset(_SECTION_CLASS_MAP.keys())


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------

def _convert_value(
    field_name: str,
    value: Any,
    target_type: type,
    *,
    section: str,
) -> Any:
    """Convert *value* to *target_type*, raising on mismatch.

    Supported target types: ``Decimal``, ``int``, ``bool``, ``str``.

    Raises
    ------
    TypeError / ValueError
        With a message that includes the section and field name for
        easy debugging of TOML files.
    """
    # -- Decimal ----------------------------------------------------------
    if target_type is Decimal:
        try:
            return decimal_from_any(value)
        except (ValueError, TypeError) as exc:
            raise type(exc)(
                f"[{section}].{field_name}: {exc}"
            ) from exc

    # -- int --------------------------------------------------------------
    if target_type is int:
        if isinstance(value, bool):
            raise TypeError(
                f"[{section}].{field_name}: expected int, got bool ({value!r})"
            )
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise TypeError(
                    f"[{section}].{field_name}: cannot convert string "
                    f"{value!r} to int"
                ) from None
        if isinstance(value, float):
            raise TypeError(
                f"[{section}].{field_name}: expected int, got float "
                f"({value!r}); refusing to silently truncate"
            )
        raise TypeError(
            f"[{section}].{field_name}: cannot convert "
            f"{type(value).__name__} ({value!r}) to int"
        )

    # -- bool -------------------------------------------------------------
    if target_type is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"[{section}].{field_name}: expected bool, got "
                f"{type(value).__name__} ({value!r})"
            )
        return value

    # -- str --------------------------------------------------------------
    if target_type is str:
        if not isinstance(value, str):
            raise TypeError(
                f"[{section}].{field_name}: expected str, got "
                f"{type(value).__name__} ({value!r})"
            )
        return value

    # -- unsupported ------------------------------------------------------
    raise TypeError(
        f"[{section}].{field_name}: unsupported target type "
        f"{target_type.__name__}"
    )


# ---------------------------------------------------------------------------
# Dataclass construction helpers
# ---------------------------------------------------------------------------


def _build_dataclass_from_mapping(
    cls: type,
    values: Mapping[str, Any],
    *,
    section: str,
) -> Any:
    """Build an instance of *cls* from a string-keyed mapping.

    Only keys that match declared fields of *cls* are consumed.  Unknown
    keys must be rejected by the caller before this helper is invoked.

    Missing fields fall back to the field's declared default (or
    default_factory result).
    """
    if not is_dataclass(cls):
        raise TypeError(
            f"[{section}]: expected dataclass type, got {cls!r}"
        )

    # Resolve type hints (needed because the project uses
    # ``from __future__ import annotations``, which stringifies all
    # annotations).
    resolved_types = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        field_type = resolved_types[f.name]
        if f.name in values:
            raw = values[f.name]
            kwargs[f.name] = _convert_value(
                f.name, raw, field_type, section=section
            )
        else:
            # Use the field's default or default_factory.
            if f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                kwargs[f.name] = f.default_factory()
            else:
                # Every field in our schema has a default, so this branch
                # should never be reached in practice.
                raise ValueError(
                    f"[{section}].{f.name}: field has no default and "
                    f"no value was provided"
                )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_symbol_config_from_mapping(
    data: Mapping[str, Any],
) -> SymbolConfig:
    """Construct a ``SymbolConfig`` from a TOML-parsed ``dict``.

    Parameters
    ----------
    data : Mapping[str, Any]
        The parsed TOML content.  Top-level keys are section names; each
        section value must itself be a ``Mapping``.

    Returns
    -------
    SymbolConfig

    Raises
    ------
    ValueError
        If an unknown section or unknown section-level key is encountered.
    TypeError
        If a section value is not a mapping, or a field value cannot be
        converted to the expected type.
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"TOML top-level must be a mapping (dict), "
            f"got {type(data).__name__}"
        )

    # -- reject unknown sections -----------------------------------------
    unknown_sections = set(data.keys()) - _ALLOWED_SECTIONS
    if unknown_sections:
        raise ValueError(
            f"Unknown TOML section(s): {sorted(unknown_sections)}. "
            f"Allowed sections: {sorted(_ALLOWED_SECTIONS)}"
        )

    # -- build each sub-dataclass ----------------------------------------
    section_kwargs: dict[str, Any] = {}
    for section_name, section_cls in _SECTION_CLASS_MAP.items():
        section_data = data.get(section_name, {})

        if not isinstance(section_data, Mapping):
            raise TypeError(
                f"TOML section [{section_name}] must be a mapping "
                f"(dict/table), got {type(section_data).__name__}"
            )

        # Reject unknown keys within the section.
        field_names = {f.name for f in fields(section_cls)}
        unknown_keys = set(section_data.keys()) - field_names
        if unknown_keys:
            raise ValueError(
                f"Unknown key(s) in section [{section_name}]: "
                f"{sorted(unknown_keys)}. "
                f"Allowed keys: {sorted(field_names)}"
            )

        section_kwargs[section_name] = _build_dataclass_from_mapping(
            section_cls, section_data, section=section_name
        )

    return SymbolConfig(**section_kwargs)


def load_symbol_config(path: str | Path) -> SymbolConfig:
    """Load a ``SymbolConfig`` from a single TOML file.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the ``.toml`` file.

    Returns
    -------
    SymbolConfig

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the TOML is malformed or contains unknown sections/keys.
    TypeError
        If a field value has the wrong type.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TOML config file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return build_symbol_config_from_mapping(data)


def load_symbol_config_from_dir(
    config_dir: str | Path,
    inst_id: str,
) -> SymbolConfig:
    """Load a symbol config from ``<config_dir>/<inst_id>.toml``.

    After loading, verifies that ``config.inst_id`` matches *inst_id*.

    Parameters
    ----------
    config_dir : str | Path
        Directory containing ``.toml`` files named by instrument ID.
    inst_id : str
        Expected instrument ID (e.g. ``"ETH-USDT-SWAP"``).

    Returns
    -------
    SymbolConfig

    Raises
    ------
    FileNotFoundError
        If the TOML file does not exist.
    ValueError
        If ``config.inst_id`` does not match *inst_id*.
    """
    config_dir = Path(config_dir)
    path = config_dir / f"{inst_id}.toml"
    config = load_symbol_config(path)
    if config.inst_id != inst_id:
        raise ValueError(
            f"symbol.inst_id mismatch: expected {inst_id!r}, "
            f"got {config.inst_id!r}"
        )
    return config
