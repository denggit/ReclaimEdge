from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

from src.monitors.boll_band_breakout_monitor import BreakoutSignal


@dataclass(frozen=True)
class EmailNotifierConfig:
    sender: str
    password: str
    receiver: str
    smtp_host: str
    smtp_port: int = 465

    @classmethod
    def from_env(cls) -> "EmailNotifierConfig":
        sender = os.getenv("EMAIL_SENDER", "")
        password = os.getenv("EMAIL_PASSWORD", "")
        receiver = os.getenv("EMAIL_RECEIVER", "")
        if not sender or not password or not receiver:
            raise ValueError("EMAIL_SENDER, EMAIL_PASSWORD and EMAIL_RECEIVER are required")
        return cls(
            sender=sender,
            password=password,
            receiver=receiver,
            smtp_host=os.getenv("EMAIL_SMTP_HOST") or infer_smtp_host(sender),
            smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "465")),
        )


def infer_smtp_host(sender: str) -> str:
    domain = sender.split("@")[-1].lower()
    mapping = {
        "qq.com": "smtp.qq.com",
        "163.com": "smtp.163.com",
        "126.com": "smtp.126.com",
        "gmail.com": "smtp.gmail.com",
        "outlook.com": "smtp.office365.com",
        "hotmail.com": "smtp.office365.com",
    }
    if domain not in mapping:
        raise ValueError(f"Cannot infer SMTP host for {domain}. Please set EMAIL_SMTP_HOST")
    return mapping[domain]


class EmailNotifier:
    def __init__(self, config: EmailNotifierConfig):
        self.config = config

    def send_signal(self, signal: BreakoutSignal) -> None:
        direction_text = {
            "BREAK_UPPER": "price moved from inside BOLL band to above upper band",
            "BREAK_LOWER": "price moved from inside BOLL band to below lower band",
        }[signal.direction]

        subject = f"[ReclaimEdge] {signal.inst_id} BOLL breakout: {signal.direction}"
        body = (
            "ReclaimEdge BOLL Breakout Alert\n\n"
            f"instrument: {signal.inst_id}\n"
            f"direction: {signal.direction}\n"
            f"description: {direction_text}\n\n"
            f"price: {signal.price}\n"
            f"previous_price: {signal.previous_price}\n\n"
            f"boll_upper: {signal.upper}\n"
            f"boll_middle: {signal.middle}\n"
            f"boll_lower: {signal.lower}\n\n"
            f"upper_distance_pct: {signal.upper_distance_pct * 100:.4f}%\n"
            f"lower_distance_pct: {signal.lower_distance_pct * 100:.4f}%\n\n"
            f"candle_ts_ms: {signal.candle_ts_ms}\n"
            f"tick_ts_ms: {signal.tick_ts_ms}\n"
            f"freeze_seconds: {signal.freeze_seconds}\n"
        )

        msg = EmailMessage()
        msg["From"] = self.config.sender
        msg["To"] = self.config.receiver
        msg["Subject"] = subject
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port, context=context) as server:
            server.login(self.config.sender, self.config.password)
            server.send_message(msg)
