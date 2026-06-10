from __future__ import annotations

from pathlib import Path

import pytest

from src.live.supervisor.supervisor_email_publisher import SupervisorEmailPublisher
from src.live.supervisor.supervisor_event_pipeline import SupervisorAlert

# ============================================================================
# Source path for guards
# ============================================================================

_PUBLISHER_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "live"
    / "supervisor"
    / "supervisor_email_publisher.py"
)


# ============================================================================
# Fake email sender
# ============================================================================


class FakeEmailSender:
    """A fake async email sender for testing.

    Records every call and returns a configurable result.  Can be configured
    to raise on every invocation to exercise the publisher's exception
    handling path.
    """

    def __init__(self, result: bool = True, raises: bool = False) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []

    async def send_email_async(
        self,
        subject: str,
        content: str,
        content_type: str = "plain",
    ) -> bool:
        self.calls.append(
            {
                "subject": subject,
                "content": content,
                "content_type": content_type,
            }
        )
        if self.raises:
            raise RuntimeError("send failed")
        return self.result


# ============================================================================
# Helper: build a valid alert
# ============================================================================


def _make_alert(**overrides) -> SupervisorAlert:
    """Build a valid SupervisorAlert with optional field overrides."""
    defaults: dict = {
        "symbol": "ETH-USDT-SWAP",
        "event_type": "WORKER_TRADING_HALTED",
        "severity": "CRITICAL",
        "reason": "halt",
        "subject": "[ReclaimEdge][CRITICAL] ETH-USDT-SWAP WORKER_TRADING_HALTED",
        "body": "<html><body>halt</body></html>",
        "content_type": "html",
        "source_path": "runtime/events/x.jsonl",
        "ts_ms": 1000,
    }
    defaults.update(overrides)
    return SupervisorAlert(**defaults)


# ============================================================================
# Happy-path tests
# ============================================================================


class TestPublishAlertSends:
    @pytest.mark.asyncio
    async def test_sends_html_alert(self):
        """publish_alert delegates html alert fields to email sender."""
        fake = FakeEmailSender(result=True)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert()

        ok = await publisher.publish_alert(alert)

        assert ok is True
        assert len(fake.calls) == 1
        assert fake.calls[0]["subject"] == alert.subject
        assert fake.calls[0]["content"] == alert.body
        assert fake.calls[0]["content_type"] == "html"

    @pytest.mark.asyncio
    async def test_sends_plain_alert(self):
        """publish_alert delegates plain-text alert to email sender."""
        fake = FakeEmailSender(result=True)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert(content_type="plain")

        ok = await publisher.publish_alert(alert)

        assert ok is True
        assert len(fake.calls) == 1
        assert fake.calls[0]["content_type"] == "plain"

    @pytest.mark.asyncio
    async def test_content_type_normalized(self):
        """Whitespace and case in content_type are normalised before calling sender."""
        fake = FakeEmailSender(result=True)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert(content_type=" HTML ")

        ok = await publisher.publish_alert(alert)

        assert ok is True
        assert fake.calls[0]["content_type"] == "html"

    @pytest.mark.asyncio
    async def test_subject_stripped(self):
        """Leading / trailing whitespace is stripped from subject before sending."""
        fake = FakeEmailSender(result=True)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert(subject="  hello  ")

        ok = await publisher.publish_alert(alert)

        assert ok is True
        assert fake.calls[0]["subject"] == "hello"


# ============================================================================
# Sender failure paths
# ============================================================================


class TestPublishAlertSenderFailure:
    @pytest.mark.asyncio
    async def test_sender_returns_false(self):
        """When email sender returns False, publish_alert returns False."""
        fake = FakeEmailSender(result=False)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert()

        ok = await publisher.publish_alert(alert)

        assert ok is False
        assert len(fake.calls) == 1

    @pytest.mark.asyncio
    async def test_sender_raises_exception(self):
        """When email sender raises, publish_alert catches and returns False."""
        fake = FakeEmailSender(raises=True)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert()

        # must not raise
        ok = await publisher.publish_alert(alert)

        assert ok is False
        assert len(fake.calls) == 1


# ============================================================================
# Constructor validation
# ============================================================================


class TestConstructorValidation:
    def test_rejects_email_sender_without_send_email_async(self):
        """Constructor raises ValueError when email_sender lacks the required method."""

        class BadSender:
            pass

        with pytest.raises(ValueError):
            SupervisorEmailPublisher(email_sender=BadSender())


# ============================================================================
# Alert validation: structural checks
# ============================================================================


class TestPublishAlertInvalidAlert:
    @pytest.mark.asyncio
    async def test_missing_subject_attribute_returns_false(self):
        """Alert without a 'subject' attribute returns False, sender not called."""

        class BadAlert:
            body = "x"
            content_type = "html"

        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        ok = await publisher.publish_alert(BadAlert())

        assert ok is False
        assert len(fake.calls) == 0

    @pytest.mark.asyncio
    async def test_missing_body_attribute_returns_false(self):
        """Alert without a 'body' attribute returns False, sender not called."""

        class BadAlert:
            subject = "x"
            content_type = "html"

        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        ok = await publisher.publish_alert(BadAlert())

        assert ok is False
        assert len(fake.calls) == 0

    @pytest.mark.asyncio
    async def test_missing_content_type_attribute_returns_false(self):
        """Alert without a 'content_type' attribute returns False, sender not called."""

        class BadAlert:
            subject = "x"
            body = "x"

        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        ok = await publisher.publish_alert(BadAlert())

        assert ok is False
        assert len(fake.calls) == 0

    @pytest.mark.asyncio
    async def test_empty_subject_returns_false(self):
        """Empty subject (after strip) returns False, sender not called."""
        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        for subject in ("", "   "):
            alert = _make_alert(subject=subject)
            ok = await publisher.publish_alert(alert)
            assert ok is False
            assert len(fake.calls) == 0
            fake.calls.clear()

    @pytest.mark.asyncio
    async def test_non_str_body_returns_false(self):
        """Non-string body returns False, sender not called."""
        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert(body=123)

        ok = await publisher.publish_alert(alert)

        assert ok is False
        assert len(fake.calls) == 0

    @pytest.mark.asyncio
    async def test_empty_content_type_returns_false(self):
        """Empty content_type (after strip) returns False, sender not called."""
        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        for ct in ("", "   "):
            alert = _make_alert(content_type=ct)
            ok = await publisher.publish_alert(alert)
            assert ok is False
            assert len(fake.calls) == 0
            fake.calls.clear()

    @pytest.mark.asyncio
    async def test_unsupported_content_type_returns_false(self):
        """Unsupported content_type values return False, sender not called."""
        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        for ct in ("markdown", "json"):
            alert = _make_alert(content_type=ct)
            ok = await publisher.publish_alert(alert)
            assert ok is False, f"content_type={ct!r} should be rejected"
            assert len(fake.calls) == 0
            fake.calls.clear()


# ============================================================================
# Non-mutation
# ============================================================================


class TestDoesNotMutateAlert:
    @pytest.mark.asyncio
    async def test_alert_fields_unchanged_after_publish(self):
        """publish_alert must not mutate the alert object."""
        fake = FakeEmailSender(result=True)
        publisher = SupervisorEmailPublisher(email_sender=fake)
        alert = _make_alert()

        orig_subject = alert.subject
        orig_body = alert.body
        orig_ct = alert.content_type

        await publisher.publish_alert(alert)

        assert alert.subject == orig_subject
        assert alert.body == orig_body
        assert alert.content_type == orig_ct


# ============================================================================
# Source guard
# ============================================================================


class TestSourceGuard:
    _FORBIDDEN = [
        "EmailSender",
        "src.utils.email_sender",
        "os.getenv",
        "load_dotenv",
        "smtplib",
        "requests",
        "httpx",
        "websocket",
        "okx",
        "Trader",
        "Strategy",
        "src.trader",
        "src.strategies",
        "src.live.workers",
        "src.live.symbol_worker_app",
        "src.live.symbol_worker_factory",
        "JsonlOutbox",
        "write_json_atomic",
        "read_json_or_none",
        "ChildEventReader",
        "AlertDeduper",
        "AlertPolicy",
        "ReclaimSupervisor",
        "create_task",
        "sleep(",
        "open(",
        "Path",
        "json",
        "yaml",
        "toml",
    ]

    def test_source_does_not_contain_forbidden_symbols(self):
        """Source file must not import or reference forbidden symbols."""
        text = _PUBLISHER_SOURCE.read_text()

        for word in self._FORBIDDEN:
            assert word not in text, (
                f"Forbidden symbol {word!r} found in "
                f"supervisor_email_publisher.py"
            )


# ============================================================================
# Integration-shape test: publisher interface
# ============================================================================


class TestPublisherInterface:
    @pytest.mark.asyncio
    async def test_has_publish_alert_and_is_awaitable(self):
        """SupervisorEmailPublisher satisfies the SupervisorAlertPublisher protocol."""
        fake = FakeEmailSender()
        publisher = SupervisorEmailPublisher(email_sender=fake)

        assert hasattr(publisher, "publish_alert")

        # publish_alert must be async-callable and return a coroutine
        coro = publisher.publish_alert(_make_alert())
        assert coro is not None
        ok = await coro
        assert ok is True
