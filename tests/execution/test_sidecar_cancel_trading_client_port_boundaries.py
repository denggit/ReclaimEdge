#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_sidecar_cancel_trading_client_port_boundaries.py
@Description: Boundary tests — verify cancel_sidecar_take_profit is routed
              through TradingClientPort (20C-CLEAN-PORTS-11A).
              Method-level scan of cancel_sidecar_take_profit only.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ======================================================================
# Source file path
# ======================================================================

_SIDECAR_PATH = Path(__file__).resolve().parents[2] / "src" / "execution" / "tp_sl_sidecar_manager.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_method(source: str, method_name: str) -> str:
    """Extract a single method body from source text."""
    for marker in (f"async def {method_name}", f"def {method_name}"):
        idx = source.find(marker)
        if idx != -1:
            break
    else:
        raise AssertionError(f"Method {method_name!r} not found in source")
    remaining = source[idx:]

    for delim in ("\n    async def ", "\n    def "):
        parts = remaining.split(delim, 1)
        if len(parts) > 1:
            return parts[0]
    return remaining


# ======================================================================
# cancel_sidecar_take_profit boundary scans
# ======================================================================


class TestCancelSidecarTpMethod:
    """Method-level scan of SidecarTpManager.cancel_sidecar_take_profit."""

    REQUIRED = [
        "self.trading_client.cancel_order(",
    ]

    FORBIDDEN = [
        '"/api/v5/trade/cancel-order"',
        "'/api/v5/trade/cancel-order'",
        "self.trader.request(",
        "t.request(",
        "build_cancel_order_body",
        "order_specs",
    ]

    def test_has_required_port_call(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "cancel_sidecar_take_profit")

        for required in self.REQUIRED:
            assert required in method_text, (
                f"cancel_sidecar_take_profit must contain {required!r}"
            )

    def test_no_forbidden_direct_rest(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "cancel_sidecar_take_profit")

        for forbidden in self.FORBIDDEN:
            lines = method_text.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in line:
                    pytest.fail(
                        f"cancel_sidecar_take_profit:{i} must not contain {forbidden!r}"
                    )


# ======================================================================
# Semantic branch remains untouched
# ======================================================================


class TestCancelSidecarTpSemanticBranch:
    """Semantic branch in cancel_sidecar_take_profit must be unchanged."""

    def test_semantic_branch_still_exists(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "cancel_sidecar_take_profit")

        assert "_broker_semantic_sidecar_tp_cancel_enabled" in method_text
        assert "_cancel_sidecar_take_profit_semantic" in method_text
        assert "semantic=true" in method_text

    def test_semantic_branch_not_altered(self):
        """The semantic branch control flow must be identical."""
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "cancel_sidecar_take_profit")

        # Guard clause
        assert "if not order_id:" in method_text
        assert "return True" in method_text  # first return for None guard

        # Semantic enabled path
        assert "if self._broker_semantic_sidecar_tp_cancel_enabled():" in method_text
        assert "ok = await self._cancel_sidecar_take_profit_semantic(order_id)" in method_text


# ======================================================================
# No new forbidden abstractions in cancel path
# ======================================================================


class TestNoForbiddenInSidecarCancel:
    """SidecarTpManager.cancel_sidecar_take_profit must not introduce banned patterns."""

    FORBIDDEN_TOKENS = [
        "SidecarAdapter",
        "SidecarTradingClient",
        "place_sidecar_limit_order",
        "cancel_sidecar_order",
        "fetch_order_status",
        "fetch_sidecar_order_status",
    ]

    def test_no_forbidden_tokens_in_cancel_method(self):
        text = _read_source(_SIDECAR_PATH)
        method_text = _extract_method(text, "cancel_sidecar_take_profit")

        for token in self.FORBIDDEN_TOKENS:
            assert token not in method_text, (
                f"cancel_sidecar_take_profit must not contain {token!r}"
            )


# ======================================================================
# Compilation check
# ======================================================================


class TestSidecarFileCompiles:
    def test_file_compiles(self) -> None:
        text = _read_source(_SIDECAR_PATH)
        compile(text, str(_SIDECAR_PATH), "exec")
