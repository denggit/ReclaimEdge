from __future__ import annotations

import base64
import datetime
import hmac
import json
from dataclasses import dataclass
from typing import Any, Callable

import aiohttp


@dataclass(frozen=True)
class OkxPrivateClientConfig:
    base_url: str
    api_key: str
    secret_key: str
    passphrase: str
    timeout_seconds: float = 10.0


class OkxPrivateClient:
    """OKX private REST client with HMAC-SHA256 signing.

    Handles aiohttp session lifecycle, request signing, and response
    validation.  Does NOT know about trading strategies, position
    management, order specs, or any ReclaimEdge business logic.
    """

    def __init__(
            self,
            config: OkxPrivateClientConfig,
            *,
            timestamp_factory: Callable[[], str] | None = None,
    ) -> None:
        self._config = config
        self._timestamp_factory = timestamp_factory
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            return
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session is None:
            return
        await self._session.close()
        self._session = None

    # ------------------------------------------------------------------
    # request
    # ------------------------------------------------------------------

    async def request(self, method: str, endpoint: str, payload: Any | None = None) -> dict[str, Any]:
        await self.start()
        if self._session is None:
            raise RuntimeError("OKX private REST session is not initialized")
        method = method.upper()
        body = "" if method == "GET" else json.dumps(payload or {}, separators=(",", ":"))
        headers = self.headers(method, endpoint, body)
        if method == "GET":
            async with self._session.get(self._config.base_url + endpoint, headers=headers,
                                         timeout=self._config.timeout_seconds) as resp:
                res = await resp.json()
        else:
            async with self._session.post(self._config.base_url + endpoint, headers=headers, data=body,
                                          timeout=self._config.timeout_seconds) as resp:
                res = await resp.json()
        if res.get("code") != "0":
            raise RuntimeError(f"OKX API error: method={method} endpoint={endpoint} response={res}")
        return res

    # ------------------------------------------------------------------
    # headers / signature
    # ------------------------------------------------------------------

    def headers(self, method: str, endpoint: str, body: str) -> dict[str, str]:
        if self._timestamp_factory is not None:
            timestamp = self._timestamp_factory()
        else:
            timestamp = (
                datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
        message = timestamp + method + endpoint + body
        digest = hmac.new(
            self._config.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            digestmod="sha256",
        ).digest()
        signature = base64.b64encode(digest).decode("utf-8")
        return {
            "OK-ACCESS-KEY": self._config.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self._config.passphrase,
            "Content-Type": "application/json",
        }
