from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)


OKX_CONFIG: dict[str, Any] = {
    "api_key": os.getenv("OKX_API_KEY", ""),
    "secret_key": os.getenv("OKX_SECRET_KEY", ""),
    "passphrase": os.getenv("OKX_PASSPHASE", ""),
}

EMAIL_CONFIG: dict[str, Any] = {
    "sender": os.getenv("EMAIL_SENDER", ""),
    "password": os.getenv("EMAIL_PASSWORD", ""),
    "receiver": os.getenv("EMAIL_RECEIVER", ""),
}
