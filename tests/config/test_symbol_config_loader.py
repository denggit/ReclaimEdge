#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``config.symbol_config_loader`` — TOML loader (A02)."""

from __future__ import annotations

import textwrap
from decimal import Decimal
from pathlib import Path

import pytest

from config.symbol_config import SymbolConfig
from config.symbol_config_loader import (
    build_symbol_config_from_mapping,
    load_symbol_config,
    load_symbol_config_from_dir,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_toml(dir_path: Path, filename: str, content: str) -> Path:
    """Write dedented *content* to *dir_path* / *filename*, return the path."""
    path = dir_path / filename
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ===================================================================
# load_symbol_config — full TOML file
# ===================================================================


class TestLoadSymbolConfigFromTomlFile:
    """Load a complete-ish TOML file and spot-check field values."""

    @staticmethod
    def config(tmp_path: Path) -> SymbolConfig:
        toml = """\
            [symbol]
            inst_id = "ETH-USDT-SWAP"
            enabled = true
            live_trading = false

            [capital]
            layer_margin_pct = "0.04"
            leverage = "20"
            max_layers = 4

            [tp]
            three_stage_tp2_use_structure_boll = true
            """
        path = _write_toml(tmp_path, "ETH-USDT-SWAP.toml", toml)
        return load_symbol_config(path)

    def test_inst_id(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).inst_id == "ETH-USDT-SWAP"

    def test_capital_layer_margin_pct(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).capital.layer_margin_pct == Decimal("0.04")

    def test_capital_leverage(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).capital.leverage == Decimal("20")

    def test_capital_max_layers(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).capital.max_layers == 4

    def test_tp_three_stage_tp2_use_structure_boll(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).tp.three_stage_tp2_use_structure_boll is True


# ===================================================================
# Partial TOML — defaults are preserved
# ===================================================================


class TestPartialTomlUsesDefaults:
    """When only ``[symbol]`` is provided, all other fields are defaults."""

    @staticmethod
    def config(tmp_path: Path) -> SymbolConfig:
        toml = """\
            [symbol]
            inst_id = "ETH-USDT-SWAP"
            """
        path = _write_toml(tmp_path, "ETH-USDT-SWAP.toml", toml)
        return load_symbol_config(path)

    def test_market_bar_default(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).market.bar == "15m"

    def test_capital_max_layers_default(self, tmp_path: Path) -> None:
        assert self.config(tmp_path).capital.max_layers == 3

    def test_risk_order_failure_market_exit_delay(self, tmp_path: Path) -> None:
        c = self.config(tmp_path)
        assert c.risk.order_failure_market_exit_delay_seconds == 1800


# ===================================================================
# load_symbol_config_from_dir
# ===================================================================


class TestLoadSymbolConfigFromDir:
    """load_symbol_config_from_dir happy-path and mismatch rejection."""

    def test_checks_inst_id(self, tmp_path: Path) -> None:
        toml = """\
            [symbol]
            inst_id = "ETH-USDT-SWAP"
            """
        _write_toml(tmp_path, "ETH-USDT-SWAP.toml", toml)
        config = load_symbol_config_from_dir(tmp_path, "ETH-USDT-SWAP")
        assert config.inst_id == "ETH-USDT-SWAP"

    def test_rejects_mismatch(self, tmp_path: Path) -> None:
        # File name says ETH but content says BTC.
        toml = """\
            [symbol]
            inst_id = "BTC-USDT-SWAP"
            """
        _write_toml(tmp_path, "ETH-USDT-SWAP.toml", toml)
        with pytest.raises(ValueError, match="mismatch"):
            load_symbol_config_from_dir(tmp_path, "ETH-USDT-SWAP")

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.toml"
        with pytest.raises(FileNotFoundError):
            load_symbol_config(str(missing))


# ===================================================================
# Unknown section / key rejection
# ===================================================================


class TestUnknownSectionRejected:
    """Unknown top-level sections must raise ValueError."""

    def test_unknown_section(self, tmp_path: Path) -> None:
        toml = """\
            [unknown]
            x = 1
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises(ValueError, match="unknown.*section"):
            load_symbol_config(path)


class TestUnknownKeyRejected:
    """Unknown keys within a known section must raise ValueError."""

    def test_unknown_key_in_capital(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            unknown_key = 1
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises(ValueError, match="capital"):
            load_symbol_config(path)

    def test_unknown_key_error_message_contains_key(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            unknown_key = 1
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises(ValueError, match="unknown_key"):
            load_symbol_config(path)


# ===================================================================
# Type-rejection cases
# ===================================================================


class TestBoolStringRejected:
    """Bool fields must be actual booleans, not strings."""

    def test_bool_string_rejected(self, tmp_path: Path) -> None:
        toml = """\
            [symbol]
            enabled = "true"
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises((TypeError, ValueError), match="enabled"):
            load_symbol_config(path)


class TestBoolRejectedForInt:
    """Int fields must reject ``true`` / ``false`` (bool is a subclass of int)."""

    def test_bool_rejected_for_int(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            max_layers = true
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises((TypeError, ValueError), match="max_layers"):
            load_symbol_config(path)


class TestFloatRejectedForInt:
    """Float values must be rejected for int fields (no silent truncation)."""

    def test_float_rejected_for_int(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            max_layers = 3.5
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises((TypeError, ValueError), match="max_layers"):
            load_symbol_config(path)


# ===================================================================
# Decimal from float — must round-trip via str
# ===================================================================


class TestDecimalFloatConvertedViaStr:
    """TOML floats for Decimal fields go through ``str(value)`` first."""

    def test_decimal_float_is_converted_via_str(self, tmp_path: Path) -> None:
        toml = """\
            [execution]
            private_write_min_interval_seconds = 0.6
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        config = load_symbol_config(path)
        assert config.execution.private_write_min_interval_seconds == Decimal("0.6")


# ===================================================================
# Section must be a mapping
# ===================================================================


class TestSectionMustBeMapping:
    """A section value that is not a dict/table must raise TypeError."""

    def test_section_must_be_mapping(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            build_symbol_config_from_mapping({"capital": 1})  # type: ignore[dict-item]


# ===================================================================
# Edge-case: int field from string "3"
# ===================================================================


class TestIntFromString:
    """Int fields accept string ``"3"`` (as allowed by the spec)."""

    def test_int_from_string(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            max_layers = "3"
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        config = load_symbol_config(path)
        assert config.capital.max_layers == 3

    def test_int_from_float_string_rejected(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            max_layers = "3.5"
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        with pytest.raises((TypeError, ValueError), match="max_layers"):
            load_symbol_config(path)


# ===================================================================
# Edge-case: TOML top-level is not a dict
# ===================================================================


class TestTopLevelNotADict:
    """If the TOML parses as a non-dict (e.g. a bare value), raise."""

    def test_top_level_not_a_dict(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            build_symbol_config_from_mapping("not a dict")  # type: ignore[arg-type]


# ===================================================================
# Edge-case: Decimal from int and from Decimal
# ===================================================================


class TestDecimalFromInt:
    """Decimal fields accept plain ints in TOML."""

    def test_decimal_from_int(self, tmp_path: Path) -> None:
        toml = """\
            [capital]
            leverage = 20
            """
        path = _write_toml(tmp_path, "test.toml", toml)
        config = load_symbol_config(path)
        assert config.capital.leverage == Decimal("20")
