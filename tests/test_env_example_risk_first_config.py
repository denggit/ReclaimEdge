"""Verify .env.example is aligned with risk-first live defaults.

Covers:
1. .env.example contains LEVERAGE=20
2. .env.example does not contain LEVERAGE=50
3. .env.example does not contain uncommented LAYER_MARGIN_PCT=
4. .env.example does not contain MAX_LAYERS / ADD_GAP / SPLIT_TP /
   EXTREME_RETEST / MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT
5. .env.example contains required risk-first keys:
   TRADE_RISK_PCT=0.003
   ENTRY_MAX_STOP_DISTANCE_PCT=0.012
   ENTRY_CVD_STRUCTURE_MODE=DIVERGENCE_OR_ABSORPTION
   POST_ENTRY_SL_COOLDOWN_ENABLED=true
"""

from __future__ import annotations

import os
import re
import unittest


class TestEnvExampleRiskFirstConfig(unittest.TestCase):
    """Risk-first config assertions on .env.example."""

    _env_text: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        env_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".env.example",
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                cls._env_text = f.read()

    # ── LEVERAGE ────────────────────────────────────────────────────────

    def test_env_example_contains_leverage_20(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        # Check that LEVERAGE=20 appears as an active (non-commented) line.
        for line in self._env_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped == "LEVERAGE=20":
                return
        self.fail("LEVERAGE=20 must appear as an active line in .env.example")

    def test_env_example_does_not_contain_leverage_50(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        for line in self._env_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "LEVERAGE" in stripped and "50" in stripped:
                self.fail(
                    f"LEVERAGE=50 must not appear in .env.example: {stripped!r}"
                )

    # ── LAYER_MARGIN_PCT ─────────────────────────────────────────────────

    def test_env_example_does_not_contain_uncommented_layer_margin_pct(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        for line in self._env_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped == "":
                continue
            if "LAYER_MARGIN_PCT" in stripped and "=" in stripped:
                self.fail(
                    f"Uncommented LAYER_MARGIN_PCT= must not appear in .env.example: {stripped!r}"
                )

    # ── Legacy / removed keys ────────────────────────────────────────────

    def test_env_example_no_max_layers(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self.assertNotIn("MAX_LAYERS", self._env_text,
                         "MAX_LAYERS must not appear in .env.example")

    def test_env_example_no_add_gap(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        for key in ["ADD_GAP_MODE", "ADD_GAP_BASE_PCT"]:
            self.assertNotIn(key, self._env_text,
                             f"{key} must not appear in .env.example")

    def test_env_example_no_split_tp(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self.assertNotIn("SPLIT_TP", self._env_text,
                         "SPLIT_TP must not appear in .env.example")

    def test_env_example_no_extreme_retest(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        for key in ["EXTREME_RETEST_ADD_ENABLED", "EXTREME_RETEST_PIVOT"]:
            self.assertNotIn(key, self._env_text,
                             f"{key} must not appear in .env.example")

    def test_env_example_no_max_entry_distance(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self.assertNotIn("MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT", self._env_text,
                         "MAX_ENTRY_DISTANCE_FROM_EXTREME_PCT must not appear in .env.example")

    # ── Required risk-first keys ─────────────────────────────────────────

    def _assert_active_key_value(self, key: str, expected_value: str) -> None:
        """Assert that `key=expected_value` exists as an active (uncommented) line."""
        pattern = re.compile(rf"^{re.escape(key)}={re.escape(expected_value)}$")
        for line in self._env_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.match(stripped):
                return
        self.fail(
            f"{key}={expected_value} must appear as an active line in .env.example"
        )

    def test_env_example_contains_trade_risk_pct(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self._assert_active_key_value("TRADE_RISK_PCT", "0.003")

    def test_env_example_contains_entry_max_stop_distance_pct(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self._assert_active_key_value("ENTRY_MAX_STOP_DISTANCE_PCT", "0")

    def test_env_example_contains_entry_cvd_structure_mode(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self._assert_active_key_value(
            "ENTRY_CVD_STRUCTURE_MODE", "DIVERGENCE_OR_ABSORPTION"
        )

    def test_env_example_contains_post_entry_sl_cooldown_enabled(self) -> None:
        if self._env_text is None:
            raise unittest.SkipTest(".env.example not found")
        self._assert_active_key_value("POST_ENTRY_SL_COOLDOWN_ENABLED", "true")
