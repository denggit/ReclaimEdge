#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_aiohttp_transport.py
@Description: Unit tests for AiohttpBinanceTransport using fake session / response.

No real network requests.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest

from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
from src.exchanges.binance.signing import BinanceSignedRequest
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResponse:
    """Simulates aiohttp.ClientResponse for testing without real network."""

    def __init__(
        self,
        *,
        status: int = 200,
        text: str = '{"ok": true}',
        headers: dict | None = None,
        json_exc: Exception | None = None,
    ):
        self.status = status
        self._text = text
        self.headers = headers or {"x-test": "1"}
        self._json_exc = json_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return self._text

    async def json(self, content_type=None):
        if self._json_exc:
            raise self._json_exc
        return json.loads(self._text)


class FakeSession:
    """Records requests and returns a canned FakeResponse."""

    def __init__(self, response=None, exc=None):
        self.response = response or FakeResponse()
        self.exc = exc
        self.calls: list[dict] = []

    def request(self, method, url, headers=None):
        self.calls.append({"method": method, "url": url, "headers": headers})
        if self.exc:
            raise self.exc
        return self.response


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _signed_request(**overrides):
    """Return a default BinanceSignedRequest for test convenience."""
    kwargs = {
        "method": "GET",
        "path": "/fapi/v1/openOrders",
        "base_url": "https://fapi.binance.com",
        "params": {"symbol": "ETHUSDT"},
        "headers": {"X-MBX-APIKEY": "test-key"},
        "query_string": "symbol=ETHUSDT&timestamp=1&signature=abc",
    }
    kwargs.update(overrides)
    return BinanceSignedRequest(**kwargs)


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_builds_url_from_base_url_and_path() -> None:
    session = FakeSession()
    transport = AiohttpBinanceTransport(session=session)

    await transport.send(_signed_request(query_string=""))

    assert len(session.calls) == 1
    assert session.calls[0]["url"] == "https://fapi.binance.com/fapi/v1/openOrders"


@pytest.mark.asyncio
async def test_send_appends_query_string_to_url() -> None:
    session = FakeSession()
    transport = AiohttpBinanceTransport(session=session)

    await transport.send(_signed_request())

    assert len(session.calls) == 1
    assert session.calls[0]["url"] == (
        "https://fapi.binance.com/fapi/v1/openOrders"
        "?symbol=ETHUSDT&timestamp=1&signature=abc"
    )


# ---------------------------------------------------------------------------
# Method and headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_passes_method() -> None:
    session = FakeSession()
    transport = AiohttpBinanceTransport(session=session)

    await transport.send(_signed_request(method="POST"))

    assert len(session.calls) == 1
    assert session.calls[0]["method"] == "POST"


@pytest.mark.asyncio
async def test_send_passes_headers() -> None:
    session = FakeSession()
    transport = AiohttpBinanceTransport(session=session)

    await transport.send(_signed_request(headers={"X-MBX-APIKEY": "my-key", "X-Extra": "val"}))

    assert len(session.calls) == 1
    assert session.calls[0]["headers"]["X-MBX-APIKEY"] == "my-key"
    assert session.calls[0]["headers"]["X-Extra"] == "val"


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_returns_binance_transport_response_status_code() -> None:
    session = FakeSession(FakeResponse(status=201, text='{"created": true}'))
    transport = AiohttpBinanceTransport(session=session)

    result = await transport.send(_signed_request())

    assert isinstance(result, BinanceTransportResponse)
    assert result.status_code == 201


@pytest.mark.asyncio
async def test_send_returns_json_payload() -> None:
    session = FakeSession(FakeResponse(text='{"symbol": "ETHUSDT", "orderId": 42}'))
    transport = AiohttpBinanceTransport(session=session)

    result = await transport.send(_signed_request())

    assert result.payload == {"symbol": "ETHUSDT", "orderId": 42}


@pytest.mark.asyncio
async def test_send_returns_response_headers() -> None:
    session = FakeSession(FakeResponse(status=200, headers={"x-rate-limit": "100", "content-type": "application/json"}))
    transport = AiohttpBinanceTransport(session=session)

    result = await transport.send(_signed_request())

    assert result.headers["x-rate-limit"] == "100"
    assert result.headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# Empty and non-JSON payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_response_text_returns_empty_dict() -> None:
    session = FakeSession(FakeResponse(status=200, text=""))
    transport = AiohttpBinanceTransport(session=session)

    result = await transport.send(_signed_request())

    assert result.payload == {}


@pytest.mark.asyncio
async def test_invalid_json_text_returns_message_dict() -> None:
    session = FakeSession(FakeResponse(status=500, text="Internal Server Error"))
    transport = AiohttpBinanceTransport(session=session)

    result = await transport.send(_signed_request())

    assert result.payload == {"message": "Internal Server Error"}


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_error_raises_network_error() -> None:
    session = FakeSession(exc=aiohttp.ClientError("connection lost"))
    transport = AiohttpBinanceTransport(session=session)

    with pytest.raises(ExchangeError) as exc_info:
        await transport.send(_signed_request())

    err = exc_info.value
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.NETWORK_ERROR
    assert "connection lost" in err.message
    assert err.raw["method"] == "GET"
    assert err.raw["path"] == "/fapi/v1/openOrders"


@pytest.mark.asyncio
async def test_timeout_error_raises_network_error() -> None:
    session = FakeSession(exc=asyncio.TimeoutError("timed out"))
    transport = AiohttpBinanceTransport(session=session)

    with pytest.raises(ExchangeError) as exc_info:
        await transport.send(_signed_request())

    err = exc_info.value
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.NETWORK_ERROR
    assert "timed out" in err.message


# ---------------------------------------------------------------------------
# External session support
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_session_is_used_when_provided() -> None:
    session = FakeSession(FakeResponse(text='{"via": "external"}'))
    transport = AiohttpBinanceTransport(session=session)

    result = await transport.send(_signed_request())

    assert result.payload == {"via": "external"}
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_without_session_does_not_raise() -> None:
    """Transport can be constructed without an external session."""
    transport = AiohttpBinanceTransport(timeout_seconds=10.0)
    assert transport._session is None
    assert transport._timeout_seconds == 10.0


def test_construction_with_session_stores_it() -> None:
    """Transport stores an externally provided session."""
    session = FakeSession()
    transport = AiohttpBinanceTransport(session=session)
    assert transport._session is session
