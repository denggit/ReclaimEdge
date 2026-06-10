from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.live.supervisor.alert_policy import (
    DEFAULT_CRITICAL_RUNTIME_EVENT_TYPES,
    DEFAULT_SEVERITY_GATED_RUNTIME_EVENT_TYPES,
    DEFAULT_SUPPRESSED_RUNTIME_EVENT_TYPES,
    AlertPolicy,
    AlertPolicyDecision,
)

# ============================================================================
# Source path for guards
# ============================================================================

_ALERT_POLICY_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "live"
    / "supervisor"
    / "alert_policy.py"
)


# ============================================================================
# 1. Default allows WORKER_STARTUP_RECOVERY_FAILED
# ============================================================================


class TestDefaultAllowsCriticalEvents:
    def test_startup_recovery_failed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_STARTUP_RECOVERY_FAILED", severity="ERROR"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "critical_event_type"

    def test_trading_halted(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_TRADING_HALTED", severity="CRITICAL"
        )
        assert decision.allowed is True

    def test_heartbeat_write_failed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_HEARTBEAT_WRITE_FAILED", severity="ERROR"
        )
        assert decision.allowed is True

    def test_drain_timeout(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_DRAIN_TIMEOUT", severity="ERROR"
        )
        assert decision.allowed is True

    def test_child_event_read_failed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="CHILD_EVENT_READ_FAILED", severity="ERROR"
        )
        assert decision.allowed is True


# ============================================================================
# 6-11. Default suppresses normal lifecycle events
# ============================================================================


class TestDefaultSuppressesNormalLifecycle:
    def test_worker_started(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_STARTED", severity="INFO"
        )
        assert decision.allowed is False
        assert decision.policy_reason == "suppressed_normal_lifecycle"

    def test_worker_stopping(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_STOPPING", severity="INFO"
        )
        assert decision.allowed is False

    def test_worker_stopped(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_STOPPED", severity="INFO"
        )
        assert decision.allowed is False

    def test_startup_recovery_completed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_STARTUP_RECOVERY_COMPLETED", severity="INFO"
        )
        assert decision.allowed is False

    def test_drain_started(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_DRAIN_STARTED", severity="INFO"
        )
        assert decision.allowed is False

    def test_drain_completed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_DRAIN_COMPLETED", severity="INFO"
        )
        assert decision.allowed is False


# ============================================================================
# E06: Severity-gated event types (WORKER_ROLLING_LOSS_GUARD)
# ============================================================================


class TestSeverityGatedRollingLossGuard:
    def test_warning_allowed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_ROLLING_LOSS_GUARD", severity="WARNING"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "severity_gated_allowed"

    def test_error_allowed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_ROLLING_LOSS_GUARD", severity="ERROR"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "severity_gated_allowed"

    def test_critical_allowed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_ROLLING_LOSS_GUARD", severity="CRITICAL"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "severity_gated_allowed"

    def test_info_suppressed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_ROLLING_LOSS_GUARD", severity="INFO"
        )
        assert decision.allowed is False
        assert decision.policy_reason == "severity_gated_info_suppressed"

    def test_worker_started_still_suppressed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_STARTED", severity="INFO"
        )
        assert decision.allowed is False
        assert decision.policy_reason == "suppressed_normal_lifecycle"

    def test_worker_drain_timeout_still_allowed(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_DRAIN_TIMEOUT", severity="ERROR"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "critical_event_type"


# ============================================================================
# 12-14. Unknown event handling
# ============================================================================


class TestUnknownEvent:
    def test_unknown_error_suppressed_by_default(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="SOME_UNKNOWN_EVENT", severity="ERROR"
        )
        assert decision.allowed is False
        assert decision.policy_reason == "unknown_event_type"

    def test_unknown_error_allowed_when_enabled(self) -> None:
        policy = AlertPolicy(allow_unknown_error_severity=True)
        decision = policy.should_publish(
            event_type="SOME_UNKNOWN_EVENT", severity="ERROR"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "unknown_error_severity"

    def test_unknown_warning_still_suppressed_when_enabled(self) -> None:
        policy = AlertPolicy(allow_unknown_error_severity=True)
        decision = policy.should_publish(
            event_type="SOME_UNKNOWN_EVENT", severity="WARNING"
        )
        assert decision.allowed is False


# ============================================================================
# 15-18. Normalisation
# ============================================================================


class TestNormalisation:
    def test_event_type_stripped(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type=" WORKER_TRADING_HALTED ", severity="ERROR"
        )
        assert decision.event_type == "WORKER_TRADING_HALTED"

    def test_severity_uppercased(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_TRADING_HALTED", severity="critical"
        )
        assert decision.severity == "CRITICAL"

    def test_reason_stripped(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_TRADING_HALTED",
            severity="ERROR",
            reason="  abc  ",
        )
        assert decision.reason == "abc"

    def test_blank_reason_becomes_none(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish(
            event_type="WORKER_TRADING_HALTED",
            severity="ERROR",
            reason="   ",
        )
        assert decision.reason is None


# ============================================================================
# 19. should_publish invalid args raise ValueError
# ============================================================================


class TestShouldPublishInvalidArgs:
    def test_empty_event_type(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(event_type="", severity="INFO")

    def test_whitespace_event_type(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(event_type="   ", severity="INFO")

    def test_none_event_type(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(event_type=None, severity="INFO")

    def test_empty_severity(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(event_type="WORKER_STARTED", severity="")

    def test_whitespace_severity(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(event_type="WORKER_STARTED", severity="   ")

    def test_none_severity(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(event_type="WORKER_STARTED", severity=None)

    def test_reason_not_str(self) -> None:
        policy = AlertPolicy()
        with pytest.raises(ValueError):
            policy.should_publish(
                event_type="WORKER_STARTED", severity="INFO", reason=123
            )


# ============================================================================
# 20. Custom critical event types (replace semantics)
# ============================================================================


class TestCustomCriticalEventTypes:
    def test_custom_critical_allowed(self) -> None:
        policy = AlertPolicy(critical_event_types={"CUSTOM_CRITICAL"})
        decision = policy.should_publish(
            event_type="CUSTOM_CRITICAL", severity="ERROR"
        )
        assert decision.allowed is True
        assert decision.policy_reason == "critical_event_type"

    def test_default_critical_not_allowed_when_replaced(self) -> None:
        policy = AlertPolicy(critical_event_types={"CUSTOM_CRITICAL"})
        decision = policy.should_publish(
            event_type="WORKER_TRADING_HALTED", severity="CRITICAL"
        )
        assert decision.allowed is False
        assert decision.policy_reason == "unknown_event_type"


# ============================================================================
# 21. Custom suppressed event types (replace semantics)
# ============================================================================


class TestCustomSuppressedEventTypes:
    def test_custom_suppressed(self) -> None:
        policy = AlertPolicy(suppressed_event_types={"CUSTOM_NORMAL"})
        decision = policy.should_publish(
            event_type="CUSTOM_NORMAL", severity="INFO"
        )
        assert decision.allowed is False
        assert decision.policy_reason == "suppressed_normal_lifecycle"


# ============================================================================
# 22-23. Constructor rejects invalid event type config
# ============================================================================


class TestConstructorRejectsInvalidConfig:
    def test_critical_not_iterable(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(critical_event_types=123)

    def test_critical_is_str(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(critical_event_types="ABC")

    def test_suppressed_is_str(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(suppressed_event_types="ABC")

    def test_critical_bad_item_empty(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(critical_event_types={"   "})

    def test_critical_bad_item_not_str(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(critical_event_types={123})


# ============================================================================
# 24. Constructor rejects overlap
# ============================================================================


class TestConstructorRejectsOverlap:
    def test_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(
                critical_event_types={"A"},
                suppressed_event_types={"A"},
            )


# ============================================================================
# 25. Constructor requires bool flag
# ============================================================================


class TestConstructorRequiresBoolFlag:
    def test_flag_not_bool_int(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(allow_unknown_error_severity=1)

    def test_flag_not_bool_str(self) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(allow_unknown_error_severity="true")


# ============================================================================
# 26-28. should_publish_alert duck typing
# ============================================================================


@dataclass
class FakeAlert:
    event_type: str
    severity: str
    reason: str | None = None


class TestShouldPublishAlert:
    def test_duck_typed_alert_allowed(self) -> None:
        policy = AlertPolicy()
        alert = FakeAlert("WORKER_TRADING_HALTED", "CRITICAL", "halt")
        decision = policy.should_publish_alert(alert)
        assert decision.allowed is True
        assert decision.policy_reason == "critical_event_type"

    def test_invalid_object_does_not_raise(self) -> None:
        policy = AlertPolicy()
        decision = policy.should_publish_alert(object())
        assert decision.allowed is False
        assert decision.policy_reason == "invalid_alert_object"

    def test_missing_event_type(self) -> None:
        policy = AlertPolicy()

        @dataclass
        class BadAlert:
            severity: str = "INFO"

        decision = policy.should_publish_alert(BadAlert())
        assert decision.allowed is False
        assert decision.policy_reason == "invalid_alert_object"

    def test_invalid_reason_type_silently_dropped(self) -> None:
        policy = AlertPolicy()
        alert = FakeAlert("WORKER_TRADING_HALTED", "CRITICAL", reason=123)  # type: ignore[arg-type]
        decision = policy.should_publish_alert(alert)
        assert decision.allowed is True
        assert decision.reason is None


# ============================================================================
# 29. Source guard
# ============================================================================


class TestSourceGuard:
    def test_no_forbidden_imports(self) -> None:
        forbidden = [
            "Trader",
            "Strategy",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "EmailSender",
            "os.getenv",
            "load_dotenv",
            "asyncio",
            "ChildEventReader",
            "AlertDeduper",
            "SupervisorEventPipeline",
            "SupervisorAlert",
            "JsonlOutbox",
            "write_json_atomic",
            "read_json_or_none",
            "src.live.workers",
            "src.trader",
            "src.strategies",
            "src.live.symbol_worker_app",
            "src.live.symbol_worker_factory",
        ]
        source_text = _ALERT_POLICY_SOURCE.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source_text, (
                f"Forbidden import/usage '{token}' found in alert_policy.py"
            )


# ============================================================================
# 30. No IO / no state / no background task guard
# ============================================================================


class TestNoIOGuard:
    def test_no_forbidden_io_patterns(self) -> None:
        forbidden = [
            "open(",
            "Path",
            "write_",
            "read_",
            "create_task",
            "sleep(",
            "json",
            "yaml",
            "toml",
        ]
        source_text = _ALERT_POLICY_SOURCE.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source_text, (
                f"Forbidden IO pattern '{token}' found in alert_policy.py"
            )


# ============================================================================
# 31. AlertPolicyDecision is frozen
# ============================================================================


class TestAlertPolicyDecisionFrozen:
    def test_frozen(self) -> None:
        decision = AlertPolicyDecision(
            allowed=True,
            event_type="TEST",
            severity="INFO",
            reason=None,
            policy_reason="critical_event_type",
        )
        with pytest.raises(Exception):
            decision.allowed = False  # type: ignore[misc]


# ============================================================================
# 32. Default constants have expected values
# ============================================================================


class TestDefaultConstants:
    def test_critical_set_contains_expected(self) -> None:
        expected = {
            "WORKER_STARTUP_RECOVERY_FAILED",
            "WORKER_TRADING_HALTED",
            "WORKER_HEARTBEAT_WRITE_FAILED",
            "WORKER_DRAIN_TIMEOUT",
            "CHILD_EVENT_READ_FAILED",
            "SUPERVISOR_CHILD_EXITED_UNEXPECTED",
            "SUPERVISOR_RESTART_FAILED",
            "SUPERVISOR_HEARTBEAT_MISSING",
        }
        assert DEFAULT_CRITICAL_RUNTIME_EVENT_TYPES == expected

    def test_suppressed_set_contains_expected(self) -> None:
        expected = {
            "WORKER_STARTED",
            "WORKER_STOPPING",
            "WORKER_STOPPED",
            "WORKER_STARTUP_RECOVERY_COMPLETED",
            "WORKER_DRAIN_STARTED",
            "WORKER_DRAIN_COMPLETED",
        }
        assert DEFAULT_SUPPRESSED_RUNTIME_EVENT_TYPES == expected

    def test_severity_gated_set_contains_expected(self) -> None:
        expected = {"WORKER_ROLLING_LOSS_GUARD"}
        assert DEFAULT_SEVERITY_GATED_RUNTIME_EVENT_TYPES == expected

    def test_no_overlap_between_defaults(self) -> None:
        overlap_cs = (
            DEFAULT_CRITICAL_RUNTIME_EVENT_TYPES
            & DEFAULT_SUPPRESSED_RUNTIME_EVENT_TYPES
        )
        assert overlap_cs == set()
        overlap_cg = (
            DEFAULT_CRITICAL_RUNTIME_EVENT_TYPES
            & DEFAULT_SEVERITY_GATED_RUNTIME_EVENT_TYPES
        )
        assert overlap_cg == set()
        overlap_sg = (
            DEFAULT_SUPPRESSED_RUNTIME_EVENT_TYPES
            & DEFAULT_SEVERITY_GATED_RUNTIME_EVENT_TYPES
        )
        assert overlap_sg == set()
