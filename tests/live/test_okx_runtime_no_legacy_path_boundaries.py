#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_runtime_no_legacy_path_boundaries.py
@Description: Boundary tests verifying that business/live/strategy layers
              do NOT use OKX legacy env vars or /api/v5 directly.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# Files where /api/v5 is FORBIDDEN (business/live layer — no direct OKX REST)
FORBIDDEN_API_V5_FILES = [
    "src/execution/trader.py",
    "scripts/run_boll_cvd_live.py",
    "src/live/runtime_bundle.py",
    "src/live/runtime_factory.py",
    "src/strategies/boll_cvd_reclaim_strategy.py",
    "src/strategies/boll_cvd_shock_reclaim_strategy.py",
]

# Files where OKX legacy env vars are FORBIDDEN (live/strategy layer)
FORBIDDEN_OKX_ENV_FILES = [
    "scripts/run_boll_cvd_live.py",
]

OKX_LEGACY_ENV_VARS = ["OKX_INST_ID", "OKX_BAR", "OKX_TD_MODE", "OKX_POS_SIDE_MODE"]

# Files where /api/v5 is ALLOWED
ALLOWED_API_V5_FILES = [
    "src/execution/okx_private_client.py",
    "src/execution/okx_trading_client.py",
    "src/data_feed/okx_loader.py",
    "src/data_feed/okx_stream.py",
    "src/data_feed/okx_books_stream.py",
    "src/data_feed/okx_market_data_client.py",
]


class TestApiV5OnlyInAdapterLayer:
    """/api/v5 must ONLY appear in OKX adapter/client files."""

    def test_api_v5_not_in_forbidden_files(self) -> None:
        violations = []
        for rel_path in FORBIDDEN_API_V5_FILES:
            filepath = ROOT / rel_path
            if not filepath.exists():
                continue
            text = filepath.read_text(encoding="utf-8")
            if "/api/v5" in text:
                # Check which line
                for i, line in enumerate(text.split("\n"), 1):
                    if "/api/v5" in line:
                        violations.append(f"{rel_path}:{i}: {line.strip()}")
        assert not violations, (
            f"/api/v5 found in forbidden files:\n" + "\n".join(violations)
        )

    def test_api_v5_is_in_allowed_files(self) -> None:
        """Confirm /api/v5 appears in at least some allowed files."""
        found = []
        for rel_path in ALLOWED_API_V5_FILES:
            filepath = ROOT / rel_path
            if filepath.exists() and "/api/v5" in filepath.read_text(encoding="utf-8"):
                found.append(rel_path)
        assert len(found) > 0, "/api/v5 should appear in allowed adapter/client files"


class TestNoLegacyOkxEnvInBusinessLayer:
    """OKX legacy env vars must NOT be read in business/live layers."""

    def test_okx_inst_id_not_in_forbidden_files(self) -> None:
        violations = []
        for rel_path in FORBIDDEN_OKX_ENV_FILES:
            filepath = ROOT / rel_path
            if not filepath.exists():
                continue
            text = filepath.read_text(encoding="utf-8")
            for var in OKX_LEGACY_ENV_VARS:
                if var in text:
                    for i, line in enumerate(text.split("\n"), 1):
                        if var in line:
                            violations.append(f"{rel_path}:{i}: {line.strip()}")
        assert not violations, (
            f"OKX legacy env vars found in forbidden files:\n" + "\n".join(violations)
        )
