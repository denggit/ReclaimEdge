#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OKX execution REST boundary audit.

Scans OKX execution files for direct REST endpoint usage and order body
builder calls, then enforces that every occurrence is explicitly classified
in an allow-list.  Migrated methods must NOT contain forbidden direct REST
endpoints and MUST call the TradingClientPort primitives.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ── Project-absolute root ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ── Target files ─────────────────────────────────────────────────────────
EXECUTION_FILES: list[str] = [
    "src/execution/trader.py",
    "src/execution/okx_trading_client.py",
    "src/execution/tp_sl_execution_manager.py",
    "src/execution/tp_sl_core_tp_manager.py",
    "src/execution/tp_sl_protective_stop_manager.py",
    "src/execution/tp_sl_market_exit_manager.py",
    "src/execution/tp_sl_near_tp_manager.py",
    "src/execution/tp_sl_sidecar_manager.py",
]

# ── REST endpoints to scan ───────────────────────────────────────────────
REST_ENDPOINTS: list[str] = [
    "/api/v5/trade/order",
    "/api/v5/trade/order-algo",
    "/api/v5/trade/cancel-order",
    "/api/v5/trade/cancel-algos",
    "/api/v5/trade/orders-pending",
    "/api/v5/trade/orders-algo-pending",
    "/api/v5/account/balance",
    "/api/v5/account/positions",
    "/api/v5/account/set-leverage",
]

# ── Order body builders to scan ──────────────────────────────────────────
ORDER_BODY_BUILDERS: list[str] = [
    "build_market_entry_order_body",
    "build_reduce_only_market_order_body",
    "build_reduce_only_tp_order_body",
    "build_conditional_protective_sl_algo_body",
    "_reduce_only_market_order_body",
    "_near_tp_protective_sl_algo_body",
    "_near_tp_fallback_conditional_close_body",
]

# ── Allowed direct REST whitelist ────────────────────────────────────────
# key: (path, method, endpoint) -> reason
ALLOWED_DIRECT_REST: dict[tuple[str, str, str], str] = {
    # ── OkxTradingClient adapter bridge ──────────────────────────────────
    ("src/execution/okx_trading_client.py", "place_market_order", "/api/v5/trade/order"):
        "adapter bridge",
    ("src/execution/okx_trading_client.py", "place_limit_order", "/api/v5/trade/order"):
        "adapter bridge",
    ("src/execution/okx_trading_client.py", "place_stop_market_order", "/api/v5/trade/order-algo"):
        "adapter bridge",
    ("src/execution/okx_trading_client.py", "cancel_order", "/api/v5/trade/cancel-order"):
        "adapter bridge",
    ("src/execution/okx_trading_client.py", "cancel_order", "/api/v5/trade/cancel-algos"):
        "adapter bridge (fallback)",
    ("src/execution/okx_trading_client.py", "cancel_algo_order", "/api/v5/trade/cancel-algos"):
        "adapter bridge — direct algo cancel API",
    ("src/execution/okx_trading_client.py", "fetch_balance", "/api/v5/account/balance"):
        "adapter bridge — direct REST (no recursion via Trader)",
    ("src/execution/okx_trading_client.py", "fetch_position", "/api/v5/account/positions"):
        "adapter bridge — direct REST (no recursion via Trader)",
    ("src/execution/okx_trading_client.py", "fetch_open_algo_orders", "/api/v5/trade/orders-algo-pending"):
        "adapter bridge — direct REST (no recursion via Trader)",
    ("src/execution/okx_trading_client.py", "configure_instrument", "/api/v5/account/set-leverage"):
        "adapter bridge — direct REST (no recursion via Trader)",

    # ── OkxTradingClient extended reads ──────────────────────────────────
    ("src/execution/okx_trading_client.py", "fetch_order_status", "/api/v5/trade/order"):
        "adapter bridge",

    # ── OkxTradingClient fetch_open_orders (direct REST, no broker) ─────
    ("src/execution/okx_trading_client.py", "fetch_open_orders", "/api/v5/trade/orders-pending"):
        "adapter bridge — direct REST (no broker, no Trader recursion)",
}

# ── Allowed order body builder whitelist ─────────────────────────────────
# key: (path, method, builder) -> reason
ALLOWED_ORDER_BODY_BUILDERS: dict[tuple[str, str, str], str] = {
    # ── OkxTradingClient adapter bridge ──────────────────────────────────
    ("src/execution/okx_trading_client.py", "place_market_order", "build_market_entry_order_body"):
        "adapter bridge builds OKX body",
    ("src/execution/okx_trading_client.py", "place_market_order", "build_reduce_only_market_order_body"):
        "adapter bridge builds OKX body",
    ("src/execution/okx_trading_client.py", "place_limit_order", "build_reduce_only_tp_order_body"):
        "adapter bridge builds OKX body",
    ("src/execution/okx_trading_client.py", "place_stop_market_order", "build_conditional_protective_sl_algo_body"):
        "adapter bridge builds OKX body",

    # ── Trader legacy body helpers ───────────────────────────────────────
    ("src/execution/trader.py", "_reduce_only_market_order_body", "_reduce_only_market_order_body"):
        "legacy helper pending deletion after all call sites are migrated",
    ("src/execution/trader.py", "_near_tp_protective_sl_algo_body", "_near_tp_protective_sl_algo_body"):
        "legacy helper pending deletion after all call sites are migrated",
    ("src/execution/trader.py", "_near_tp_fallback_conditional_close_body", "_near_tp_fallback_conditional_close_body"):
        "legacy helper pending deletion after all call sites are migrated",

    # ── Trader legacy helpers calling order_specs builders ──────────────
    ("src/execution/trader.py", "_reduce_only_tp_order_body", "build_reduce_only_tp_order_body"):
        "legacy helper pending deletion after all call sites are migrated",
    ("src/execution/trader.py", "_reduce_only_market_order_body", "build_reduce_only_market_order_body"):
        "legacy helper pending deletion after all call sites are migrated",
    ("src/execution/trader.py", "_near_tp_protective_sl_algo_body", "build_conditional_protective_sl_algo_body"):
        "legacy helper pending deletion after all call sites are migrated",
    ("src/execution/trader.py", "_near_tp_fallback_conditional_close_body", "build_conditional_protective_sl_algo_body"):
        "legacy helper pending deletion after all call sites are migrated",
}

# ── Migrated methods that must NOT contain direct REST ───────────────────
FORBIDDEN_DIRECT_REST_ENDPOINTS: frozenset[str] = frozenset({
    "/api/v5/trade/order",
    "/api/v5/trade/order-algo",
    "/api/v5/trade/cancel-order",
    "/api/v5/trade/cancel-algos",
    "/api/v5/trade/orders-pending",
    "/api/v5/account/balance",
    "/api/v5/account/positions",
})

MIGRATED_METHODS: set[tuple[str, str]] = {
    ("src/execution/trader.py", "execute_intent"),
    ("src/execution/trader.py", "place_sidecar_market_order"),
    ("src/execution/tp_sl_execution_manager.py", "cancel_existing_reduce_only_orders"),
    ("src/execution/tp_sl_core_tp_manager.py", "replace_take_profit"),
    ("src/execution/tp_sl_core_tp_manager.py", "_place_reduce_only_take_profit_orders"),
    ("src/execution/tp_sl_protective_stop_manager.py", "place_near_tp_protective_stop_with_retries"),
    ("src/execution/tp_sl_market_exit_manager.py", "market_exit_remaining_position_with_retries"),
    ("src/execution/tp_sl_near_tp_manager.py", "execute_near_tp_reduce"),
    ("src/execution/tp_sl_sidecar_manager.py", "place_sidecar_fixed_take_profit"),
    ("src/execution/tp_sl_sidecar_manager.py", "cancel_sidecar_take_profit"),
    ("src/execution/trader.py", "initialize"),
    ("src/execution/trader.py", "fetch_sidecar_order_status"),
    ("src/execution/tp_sl_sidecar_manager.py", "fetch_sidecar_order_status"),
    ("src/execution/tp_sl_protective_stop_manager.py", "verify_near_tp_protective_stop"),
}

# ── Migrated methods must call these port methods ────────────────────────
MIGRATED_METHOD_REQUIRED_PORT_CALLS: dict[tuple[str, str], list[str]] = {
    ("src/execution/trader.py", "execute_intent"): [".place_market_order("],
    ("src/execution/trader.py", "place_sidecar_market_order"): [".place_market_order("],
    ("src/execution/tp_sl_execution_manager.py", "cancel_existing_reduce_only_orders"): [
        ".fetch_open_orders(", ".cancel_order(",
    ],
    ("src/execution/tp_sl_core_tp_manager.py", "replace_take_profit"): [".fetch_position("],
    ("src/execution/tp_sl_core_tp_manager.py", "_place_reduce_only_take_profit_orders"): [
        ".place_limit_order(",
    ],
    ("src/execution/tp_sl_protective_stop_manager.py", "place_near_tp_protective_stop_with_retries"): [
        ".place_stop_market_order(",
    ],
    ("src/execution/tp_sl_market_exit_manager.py", "market_exit_remaining_position_with_retries"): [
        ".fetch_position(", ".place_market_order(",
    ],
    ("src/execution/tp_sl_near_tp_manager.py", "execute_near_tp_reduce"): [
        ".fetch_position(", ".place_market_order(",
    ],
    ("src/execution/tp_sl_sidecar_manager.py", "place_sidecar_fixed_take_profit"): [
        ".place_limit_order(",
    ],
    ("src/execution/tp_sl_sidecar_manager.py", "cancel_sidecar_take_profit"): [
        ".cancel_order(",
    ],
    ("src/execution/trader.py", "initialize"): [
        ".fetch_balance(",
        ".configure_instrument(",
    ],
    ("src/execution/trader.py", "fetch_sidecar_order_status"): [
        "._require_tp_sl_manager().fetch_sidecar_order_status(",
    ],
    ("src/execution/tp_sl_sidecar_manager.py", "fetch_sidecar_order_status"): [
        ".fetch_order_status(",
    ],
    ("src/execution/tp_sl_protective_stop_manager.py", "verify_near_tp_protective_stop"): [
        ".fetch_open_algo_orders(",
    ],
}

# ── Banned re-introduced types ───────────────────────────────────────────
BANNED_TOKENS: list[str] = [
    "ThreeStageAdapter",
    "MiddleRunnerAdapter",
    "SidecarAdapter",
    "ExchangeRuntimeBundle",
    "BinanceBrokerSemanticExecutor",
]


# ── Helper types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Occurrence:
    path: str
    method: str  # "<module>" when outside any function
    token: str
    line_no: int
    line: str


@dataclass
class AuditReport:
    endpoint_occurrences: list[Occurrence] = field(default_factory=list)
    builder_occurrences: list[Occurrence] = field(default_factory=list)
    banned_occurrences: list[Occurrence] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _method_spans(source: str) -> dict[str, tuple[int, int]]:
    """Return {method_name: (start_line, end_line)} for every function def.

    Module-level code that is not inside any function is not represented
    here — we treat those as ``"<module>"``.
    """
    tree = ast.parse(source)
    spans: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans[node.name] = (node.lineno, node.end_lineno or node.lineno)
    return spans


def _method_for_line(line_no: int, spans: dict[str, tuple[int, int]]) -> str:
    """Return the method name that contains *line_no*, or ``"<module>"``."""
    for name, (start, end) in spans.items():
        if start <= line_no <= end:
            return name
    return "<module>"


# ── Occurrence collectors ────────────────────────────────────────────────


def _find_endpoint_occurrences(path: str, source: str) -> list[Occurrence]:
    """Find REST endpoint occurrences in *source*.

    To avoid false positives (e.g. ``/api/v5/trade/order`` matching inside
    ``/api/v5/trade/order-algo``), we check that the character immediately
    after the match is NOT ``-`` (path-continuation) or an alphanumeric.
    """
    spans = _method_spans(source)
    hits: list[Occurrence] = []
    seen: set[tuple[int, str]] = set()  # dedupe by (line_no, endpoint)

    for endpoint in REST_ENDPOINTS:
        # Sort endpoints by length descending so longer paths are matched
        # first; this prevents /api/v5/trade/order from matching inside
        # /api/v5/trade/order-algo when the longer endpoint is also being
        # scanned.
        idx = 0
        while True:
            idx = source.find(endpoint, idx)
            if idx == -1:
                break
            # Verify boundary — the character after the endpoint must NOT be
            # a path-continuation character.
            after_idx = idx + len(endpoint)
            if after_idx < len(source):
                after_char = source[after_idx]
                if after_char.isalnum() or after_char == "-":
                    idx = after_idx
                    continue
            line_no = source[:idx].count("\n") + 1
            method = _method_for_line(line_no, spans)
            line = source.splitlines()[line_no - 1].strip()
            key = (line_no, endpoint)
            if key not in seen:
                seen.add(key)
                hits.append(Occurrence(path=path, method=method, token=endpoint, line_no=line_no, line=line))
            idx = after_idx

    return hits


def _find_builder_occurrences(path: str, source: str) -> list[Occurrence]:
    """Find order body builder occurrences in *source*.

    Uses word-boundary matching so ``_reduce_only_market_order_body`` does
    NOT match inside ``build_reduce_only_market_order_body``.
    """
    spans = _method_spans(source)
    hits: list[Occurrence] = []
    seen: set[tuple[int, str]] = set()  # dedupe by (line_no, builder)

    for builder in ORDER_BODY_BUILDERS:
        pattern = re.compile(r"(?<!\w)" + re.escape(builder) + r"(?!\w)")
        for match in pattern.finditer(source):
            line_no = source[: match.start()].count("\n") + 1
            method = _method_for_line(line_no, spans)
            line = source.splitlines()[line_no - 1].strip()
            key = (line_no, builder)
            if key not in seen:
                seen.add(key)
                hits.append(Occurrence(path=path, method=method, token=builder, line_no=line_no, line=line))

    return hits


def _find_banned_occurrences(path: str, source: str) -> list[Occurrence]:
    """Find banned type names in *source*."""
    spans = _method_spans(source)
    hits: list[Occurrence] = []
    seen: set[tuple[int, str]] = set()

    for token in BANNED_TOKENS:
        pattern = re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)")
        for match in pattern.finditer(source):
            line_no = source[: match.start()].count("\n") + 1
            method = _method_for_line(line_no, spans)
            line = source.splitlines()[line_no - 1].strip()
            key = (line_no, token)
            if key not in seen:
                seen.add(key)
                hits.append(Occurrence(path=path, method=method, token=token, line_no=line_no, line=line))

    return hits


# ── Collect all occurrences ──────────────────────────────────────────────


def _collect_all() -> AuditReport:
    report = AuditReport()
    for rel_path in EXECUTION_FILES:
        source = _read(rel_path)
        report.endpoint_occurrences.extend(_find_endpoint_occurrences(rel_path, source))
        report.builder_occurrences.extend(_find_builder_occurrences(rel_path, source))
        report.banned_occurrences.extend(_find_banned_occurrences(rel_path, source))
    return report


# ── Helper: extract method source ────────────────────────────────────────


def _method_source(source: str, method_name: str) -> str | None:
    """Return the source segment for *method_name*, or None."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node)
    return None


# ══════════════════════════════════════════════════════════════════════════
# Test cases
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def report() -> AuditReport:
    return _collect_all()


# ── Test 1: All direct REST endpoints are classified ─────────────────────


def test_all_direct_rest_endpoints_are_classified(report: AuditReport) -> None:
    """Every REST endpoint occurrence must exist in ALLOWED_DIRECT_REST."""
    unclassified: list[str] = []
    for occ in report.endpoint_occurrences:
        key = (occ.path, occ.method, occ.token)
        if key not in ALLOWED_DIRECT_REST:
            unclassified.append(
                f"- {occ.path}::{occ.method} line {occ.line_no} token={occ.token}\n"
                f"    {occ.line}"
            )

    if unclassified:
        msg = (
            f"Unclassified direct REST endpoint(s) ({len(unclassified)}):\n\n"
            + "\n".join(unclassified)
            + "\n\nEach occurrence must be listed in ALLOWED_DIRECT_REST with a "
            "concrete (path, method, endpoint) → reason entry."
        )
        pytest.fail(msg)


# ── Test 2: All order body builders are classified ───────────────────────


def test_all_okx_order_body_builders_are_classified(report: AuditReport) -> None:
    """Every order body builder occurrence must exist in ALLOWED_ORDER_BODY_BUILDERS."""
    unclassified: list[str] = []
    for occ in report.builder_occurrences:
        key = (occ.path, occ.method, occ.token)
        if key not in ALLOWED_ORDER_BODY_BUILDERS:
            unclassified.append(
                f"- {occ.path}::{occ.method} line {occ.line_no} token={occ.token}\n"
                f"    {occ.line}"
            )

    if unclassified:
        msg = (
            f"Unclassified order body builder(s) ({len(unclassified)}):\n\n"
            + "\n".join(unclassified)
            + "\n\nEach occurrence must be listed in ALLOWED_ORDER_BODY_BUILDERS with a "
            "concrete (path, method, builder) → reason entry."
        )
        pytest.fail(msg)


# ── Test 3: Migrated methods must NOT contain forbidden direct REST ──────


def test_migrated_methods_no_forbidden_direct_rest(report: AuditReport) -> None:
    """Migrated methods must not contain forbidden direct REST endpoints."""
    violations: list[str] = []
    for occ in report.endpoint_occurrences:
        key = (occ.path, occ.method)
        if key in MIGRATED_METHODS and occ.token in FORBIDDEN_DIRECT_REST_ENDPOINTS:
            violations.append(
                f"- {occ.path}::{occ.method} line {occ.line_no} token={occ.token}\n"
                f"    {occ.line}"
            )

    if violations:
        msg = (
            f"Forbidden direct REST in migrated method(s) ({len(violations)}):\n\n"
            + "\n".join(violations)
            + "\n\nMigrated methods must use TradingClientPort primitives instead "
            "of direct REST calls."
        )
        pytest.fail(msg)


# ── Test 4: Migrated methods must call port primitives ───────────────────


def test_migrated_methods_call_port_primitives() -> None:
    """Migrated methods must call their required TradingClientPort methods."""
    violations: list[str] = []

    for (path, method_name), required_calls in MIGRATED_METHOD_REQUIRED_PORT_CALLS.items():
        source = _read(path)
        method_src = _method_source(source, method_name)
        if method_src is None:
            violations.append(f"- {path}::{method_name}: method not found in source")
            continue
        for required in required_calls:
            if required not in method_src:
                violations.append(
                    f"- {path}::{method_name}: missing required port call {required!r}"
                )

    if violations:
        msg = (
            f"Migrated method(s) missing required port call(s) ({len(violations)}):\n\n"
            + "\n".join(violations)
            + "\n\nMigrated methods must call TradingClientPort primitives instead "
            "of direct REST."
        )
        pytest.fail(msg)


# ── Test 5: All allowed REST endpoint reasons are non-empty ──────────────


def test_allowed_direct_rest_reasons_are_non_empty() -> None:
    """Every entry in ALLOWED_DIRECT_REST must have a non-empty reason."""
    violations: list[str] = []
    for key, reason in ALLOWED_DIRECT_REST.items():
        if not reason or not reason.strip():
            violations.append(f"- {key}: empty reason")
    if violations:
        pytest.fail(
            "ALLOWED_DIRECT_REST entries with empty reason:\n" + "\n".join(violations)
        )


# ── Test 6: All allowed builder reasons are non-empty ────────────────────


def test_allowed_order_body_builder_reasons_are_non_empty() -> None:
    """Every entry in ALLOWED_ORDER_BODY_BUILDERS must have a non-empty reason."""
    violations: list[str] = []
    for key, reason in ALLOWED_ORDER_BODY_BUILDERS.items():
        if not reason or not reason.strip():
            violations.append(f"- {key}: empty reason")
    if violations:
        pytest.fail(
            "ALLOWED_ORDER_BODY_BUILDERS entries with empty reason:\n"
            + "\n".join(violations)
        )


# ── Test 7: No re-introduced banned types ────────────────────────────────


def test_no_banned_types_reintroduced(report: AuditReport) -> None:
    """None of the banned type names must appear in the target files."""
    if report.banned_occurrences:
        lines: list[str] = []
        for occ in report.banned_occurrences:
            lines.append(
                f"- {occ.path}::{occ.method} line {occ.line_no} token={occ.token}\n"
                f"    {occ.line}"
            )
        pytest.fail(
            f"Banned type(s) found in target files ({len(report.banned_occurrences)}):\n\n"
            + "\n".join(lines)
        )


# ── Test 8: No wildcard / file-level allowlist entries ───────────────────


def test_no_file_level_allowlist_wildcards() -> None:
    """ALLOWED_DIRECT_REST and ALLOWED_ORDER_BODY_BUILDERS must be method-level."""
    violations: list[str] = []
    for key in ALLOWED_DIRECT_REST:
        if key[1] == "*":
            violations.append(f"ALLOWED_DIRECT_REST {key}: method must not be '*'")
    for key in ALLOWED_ORDER_BODY_BUILDERS:
        if key[1] == "*":
            violations.append(f"ALLOWED_ORDER_BODY_BUILDERS {key}: method must not be '*'")
    if violations:
        pytest.fail(
            "File-level wildcard entries are forbidden:\n" + "\n".join(violations)
        )


# ── Test 9: TP/SL managers must not use OKX body builders ────────────────


def test_tp_sl_managers_no_okx_body_builders(report: AuditReport) -> None:
    """TP/SL manager files must not contain order body builder usage.

    The files covered:
    - src/execution/tp_sl_execution_manager.py
    - src/execution/tp_sl_core_tp_manager.py
    - src/execution/tp_sl_protective_stop_manager.py
    - src/execution/tp_sl_market_exit_manager.py
    - src/execution/tp_sl_near_tp_manager.py
    - src/execution/tp_sl_sidecar_manager.py
    """
    tp_sl_files: frozenset[str] = frozenset({
        "src/execution/tp_sl_execution_manager.py",
        "src/execution/tp_sl_core_tp_manager.py",
        "src/execution/tp_sl_protective_stop_manager.py",
        "src/execution/tp_sl_market_exit_manager.py",
        "src/execution/tp_sl_near_tp_manager.py",
        "src/execution/tp_sl_sidecar_manager.py",
    })

    violations: list[str] = []
    for occ in report.builder_occurrences:
        if occ.path in tp_sl_files:
            violations.append(
                f"- {occ.path}::{occ.method} line {occ.line_no} token={occ.token}\n"
                f"    {occ.line}"
            )

    if violations:
        msg = (
            f"Order body builder(s) found in TP/SL manager file(s) ({len(violations)}):\n\n"
            + "\n".join(violations)
            + "\n\nTP/SL managers must not use OKX body builders directly."
        )
        pytest.fail(msg)
