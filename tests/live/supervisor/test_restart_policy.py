#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D07b unit tests for RestartPolicy — covers config validation, evaluate
decisions (disabled, cooldown, max, window prune), record_restart, and
dataclass frozen semantics.
"""

from __future__ import annotations

import pytest

from src.live.supervisor.restart_policy import (
    RestartDecision,
    RestartPolicy,
    RestartPolicyConfig,
)


# ============================================================================
# 1. Config defaults and validation
# ============================================================================


def test_default_config_valid() -> None:
    config = RestartPolicyConfig()
    assert config.enabled is True
    assert config.cooldown_seconds == 10.0
    assert config.max_restarts == 3
    assert config.window_seconds == 600.0


def test_invalid_cooldown_raises() -> None:
    with pytest.raises(ValueError, match="cooldown_seconds must be >= 0"):
        RestartPolicyConfig(cooldown_seconds=-1)


def test_invalid_max_restarts_raises() -> None:
    with pytest.raises(ValueError, match="max_restarts must be >= 0"):
        RestartPolicyConfig(max_restarts=-1)


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window_seconds must be > 0"):
        RestartPolicyConfig(window_seconds=0)

    with pytest.raises(ValueError, match="window_seconds must be > 0"):
        RestartPolicyConfig(window_seconds=-5)


# ============================================================================
# 2. Dataclasses are frozen
# ============================================================================


def test_config_is_frozen() -> None:
    config = RestartPolicyConfig()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        config.enabled = False  # type: ignore[misc]


def test_decision_is_frozen() -> None:
    decision = RestartDecision(True, "allowed", 0)
    with pytest.raises(Exception):
        decision.allowed = False  # type: ignore[misc]


# ============================================================================
# 3. disabled → always not allowed
# ============================================================================


def test_disabled_not_allowed() -> None:
    config = RestartPolicyConfig(enabled=False)
    policy = RestartPolicy(config)
    decision = policy.evaluate(now_monotonic=100.0)
    assert decision.allowed is False
    assert decision.reason == "disabled"
    assert decision.restart_count_in_window == 0


# ============================================================================
# 4. max_restarts zero → not allowed
# ============================================================================


def test_max_restarts_zero_not_allowed() -> None:
    config = RestartPolicyConfig(max_restarts=0)
    policy = RestartPolicy(config)
    decision = policy.evaluate(now_monotonic=100.0)
    assert decision.allowed is False
    assert decision.reason == "max_restarts_zero"
    assert decision.restart_count_in_window == 0


# ============================================================================
# 5. allowed when under max and no cooldown
# ============================================================================


def test_allowed_when_under_max() -> None:
    config = RestartPolicyConfig(max_restarts=3, cooldown_seconds=0, window_seconds=600.0)
    policy = RestartPolicy(config)
    decision = policy.evaluate(now_monotonic=100.0)
    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert decision.restart_count_in_window == 0


# ============================================================================
# 6. record_restart increments count
# ============================================================================


def test_record_restart_increments_count() -> None:
    config = RestartPolicyConfig(max_restarts=5, window_seconds=600.0)
    policy = RestartPolicy(config)
    assert policy.restart_count_in_window == 0
    assert policy.last_restart_monotonic is None

    count = policy.record_restart(now_monotonic=100.0)
    assert count == 1
    assert policy.restart_count_in_window == 1
    assert policy.last_restart_monotonic == 100.0

    count = policy.record_restart(now_monotonic=200.0)
    assert count == 2
    assert policy.restart_count_in_window == 2
    assert policy.last_restart_monotonic == 200.0


# ============================================================================
# 7. cooldown blocks before next_allowed
# ============================================================================


def test_cooldown_blocks_restart() -> None:
    config = RestartPolicyConfig(cooldown_seconds=10.0, max_restarts=5)
    policy = RestartPolicy(config)

    # First restart at t=100
    policy.record_restart(now_monotonic=100.0)

    # At t=105 (< cooldown 10s): blocked
    decision = policy.evaluate(now_monotonic=105.0)
    assert decision.allowed is False
    assert decision.reason == "cooldown"
    assert decision.restart_count_in_window == 1
    assert decision.next_allowed_monotonic == 110.0


# ============================================================================
# 8. after cooldown allowed
# ============================================================================


def test_after_cooldown_allowed() -> None:
    config = RestartPolicyConfig(cooldown_seconds=10.0, max_restarts=5)
    policy = RestartPolicy(config)

    policy.record_restart(now_monotonic=100.0)

    # At t=110 exactly (cooldown elapsed): allowed
    decision = policy.evaluate(now_monotonic=110.0)
    assert decision.allowed is True
    assert decision.reason == "allowed"


# ============================================================================
# 9. max_restarts_exceeded within window
# ============================================================================


def test_max_restarts_exceeded_in_window() -> None:
    config = RestartPolicyConfig(max_restarts=3, window_seconds=600.0, cooldown_seconds=0)
    policy = RestartPolicy(config)

    policy.record_restart(now_monotonic=100.0)
    policy.record_restart(now_monotonic=200.0)
    policy.record_restart(now_monotonic=300.0)

    decision = policy.evaluate(now_monotonic=350.0)
    assert decision.allowed is False
    assert decision.reason == "max_restarts_exceeded"
    assert decision.restart_count_in_window == 3


# ============================================================================
# 10. window prune allows restart after old entries expire
# ============================================================================


def test_window_prune_allows_restart_after_expiry() -> None:
    config = RestartPolicyConfig(max_restarts=3, window_seconds=600.0, cooldown_seconds=0)
    policy = RestartPolicy(config)

    policy.record_restart(now_monotonic=100.0)
    policy.record_restart(now_monotonic=200.0)
    policy.record_restart(now_monotonic=300.0)

    # At t=350 all three still in window → exceeded
    decision = policy.evaluate(now_monotonic=350.0)
    assert decision.allowed is False
    assert decision.reason == "max_restarts_exceeded"

    # At t=950: all three are older than 950-600=350 → all pruned
    decision = policy.evaluate(now_monotonic=950.0)
    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert decision.restart_count_in_window == 0


# ============================================================================
# 11. partial window prune
# ============================================================================


def test_partial_window_prune() -> None:
    config = RestartPolicyConfig(max_restarts=3, window_seconds=600.0, cooldown_seconds=0)
    policy = RestartPolicy(config)

    policy.record_restart(now_monotonic=100.0)  # expires at 700
    policy.record_restart(now_monotonic=500.0)  # expires at 1100
    policy.record_restart(now_monotonic=650.0)  # expires at 1250

    # At t=750: first one (100) expired (100 < 750-600=150), two remain
    decision = policy.evaluate(now_monotonic=750.0)
    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert decision.restart_count_in_window == 2


# ============================================================================
# 12. cooldown AND max both in effect
# ============================================================================


def test_cooldown_and_max_interaction() -> None:
    config = RestartPolicyConfig(cooldown_seconds=10.0, max_restarts=2, window_seconds=600.0)
    policy = RestartPolicy(config)

    policy.record_restart(now_monotonic=100.0)
    policy.record_restart(now_monotonic=150.0)

    # At t=155: cooldown ok (150+10=160 > 155? No, 155 < 160, still cooldown)
    # Actually, cooldown is 10s from last restart at 150 → next allowed at 160
    decision = policy.evaluate(now_monotonic=155.0)
    assert decision.allowed is False
    assert decision.reason == "cooldown"

    # At t=165: cooldown ok, but max_restarts == 2, count == 2 → exceeded
    decision = policy.evaluate(now_monotonic=165.0)
    assert decision.allowed is False
    assert decision.reason == "max_restarts_exceeded"


# ============================================================================
# 13. disabled overrides all
# ============================================================================


def test_disabled_overrides() -> None:
    config = RestartPolicyConfig(enabled=False, max_restarts=5)
    policy = RestartPolicy(config)
    policy.record_restart(now_monotonic=100.0)
    decision = policy.evaluate(now_monotonic=200.0)
    assert decision.allowed is False
    assert decision.reason == "disabled"


# ============================================================================
# 14. record_restart returns correct count after prune
# ============================================================================


def test_record_restart_prunes_before_counting() -> None:
    config = RestartPolicyConfig(max_restarts=5, window_seconds=600.0)
    policy = RestartPolicy(config)

    policy.record_restart(now_monotonic=100.0)
    policy.record_restart(now_monotonic=200.0)

    # At t=900: both entries expired (100 < 300, 200 < 300)
    count = policy.record_restart(now_monotonic=900.0)
    assert count == 1  # only the new one at 900
    assert policy.restart_count_in_window == 1


# ============================================================================
# 15. RestartDecision fields
# ============================================================================


def test_restart_decision_fields() -> None:
    allowed = RestartDecision(True, "allowed", 2)
    assert allowed.allowed is True
    assert allowed.reason == "allowed"
    assert allowed.restart_count_in_window == 2
    assert allowed.next_allowed_monotonic is None

    blocked = RestartDecision(False, "cooldown", 3, next_allowed_monotonic=110.0)
    assert blocked.allowed is False
    assert blocked.reason == "cooldown"
    assert blocked.restart_count_in_window == 3
    assert blocked.next_allowed_monotonic == 110.0
