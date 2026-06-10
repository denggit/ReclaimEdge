from __future__ import annotations

from typing import Protocol

from src.live.supervisor.supervisor_event_pipeline import SupervisorAlert

# ---------------------------------------------------------------------------
# Protocol: what we require from an injected email sender
# ---------------------------------------------------------------------------


class AsyncMailSender(Protocol):
    """Protocol for an async email sender.

    Any object with an ``async def send_email_async(subject, content,
    content_type) -> bool`` method satisfies this protocol.
    """

    async def send_email_async(
        self,
        subject: str,
        content: str,
        content_type: str = "plain",
    ) -> bool: ...


# ---------------------------------------------------------------------------
# Allowed content types
# ---------------------------------------------------------------------------

_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({"plain", "html"})


# ---------------------------------------------------------------------------
# SupervisorEmailPublisher
# ---------------------------------------------------------------------------


class SupervisorEmailPublisher:
    """Publishes a :class:`SupervisorAlert` via an injected email sender.

    This adapter implements the :class:`SupervisorAlertPublisher` protocol.
    It validates the alert and delegates to ``email_sender.send_email_async()``.

    The adapter does **not** create a real email sender, read environment
    variables, or connect to an SMTP server.  The caller is responsible for
    providing a fully configured ``email_sender`` that satisfies the
    :class:`AsyncMailSender` protocol.

    Parameters
    ----------
    email_sender : AsyncMailSender
        An object with an ``async def send_email_async(subject, content,
        content_type) -> bool`` method.
    """

    def __init__(self, *, email_sender: AsyncMailSender) -> None:
        if not hasattr(email_sender, "send_email_async"):
            raise ValueError(
                f"email_sender must have 'send_email_async' attribute, "
                f"got {type(email_sender).__name__}"
            )
        self._email_sender = email_sender

    # ------------------------------------------------------------------
    # publish_alert
    # ------------------------------------------------------------------

    async def publish_alert(self, alert: SupervisorAlert) -> bool:
        """Publish a supervisor alert via email.

        Validates the alert fields, normalises ``content_type`` and
        ``subject``, and delegates to the injected email sender.

        Returns ``True`` when the email sender indicates success.
        Returns ``False`` when:

        * the alert fails validation (malformed or missing fields)
        * the email sender returns ``False``
        * the email sender raises an exception
        """
        # -- validate alert shape --------------------------------------------
        if not hasattr(alert, "subject"):
            return False
        if not hasattr(alert, "body"):
            return False
        if not hasattr(alert, "content_type"):
            return False

        # -- validate subject -------------------------------------------------
        subject_raw = alert.subject
        if not isinstance(subject_raw, str):
            return False
        subject = subject_raw.strip()
        if not subject:
            return False

        # -- validate body ----------------------------------------------------
        body = alert.body
        if not isinstance(body, str):
            return False

        # -- validate & normalise content_type --------------------------------
        ct_raw = alert.content_type
        if not isinstance(ct_raw, str):
            return False
        ct = ct_raw.strip().lower()
        if not ct:
            return False
        if ct not in _ALLOWED_CONTENT_TYPES:
            return False

        # -- delegate to email sender -----------------------------------------
        try:
            ok = await self._email_sender.send_email_async(subject, body, ct)
        except Exception:
            return False
        return bool(ok)
