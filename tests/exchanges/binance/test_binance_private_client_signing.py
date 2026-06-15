#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_private_client_signing.py
@Description: Tests for BinancePrivateClient signing and transport integration.
              All tests use fake transports — no real Binance access.
"""

from __future__ import annotations

import pytest

from src.exchanges.binance.private_client import BinancePrivateClient
from src.exchanges.binance.signing import BinanceSignedRequest
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeError, ExchangeErrorKind


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeTransport:
    """Records requests and returns canned responses."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.requests: list[BinanceSignedRequest] = []

    async def send(self, request: BinanceSignedRequest) -> BinanceTransportResponse:
        self.requests.append(request)
        return BinanceTransportResponse(
            status_code=self.status_code,
            payload=self.payload,
            headers={},
        )

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    transport=None,
    *,
    api_key="test-api-key",
    api_secret="test-api-secret",
    timestamp_ms=1700000000000,
):
    return BinancePrivateClient(
        api_key=api_key,
        api_secret=api_secret,
        transport=transport,
        timestamp_factory=lambda: timestamp_ms,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        BinancePrivateClient(api_key="", api_secret="secret")


def test_constructor_rejects_empty_api_secret() -> None:
    with pytest.raises(ValueError, match="api_secret"):
        BinancePrivateClient(api_key="key", api_secret="")


# ---------------------------------------------------------------------------
# signing — query string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_includes_timestamp_and_signature_in_query() -> None:
    fake = FakeTransport({"ok": True})
    client = _make_client(transport=fake, timestamp_ms=1700000000000)

    await client.get("/fapi/v1/test", {"symbol": "ETHUSDT"})

    assert len(fake.requests) == 1
    signed: BinanceSignedRequest = fake.requests[0]
    assert "timestamp=1700000000000" in signed.query_string
    assert "recvWindow=5000" in signed.query_string
    assert "signature=" in signed.query_string
    assert signed.method == "GET"


@pytest.mark.asyncio
async def test_post_includes_timestamp_and_signature() -> None:
    fake = FakeTransport({"orderId": 123})
    client = _make_client(transport=fake, timestamp_ms=1700000000000)

    await client.post("/fapi/v1/order", {"symbol": "ETHUSDT", "side": "BUY"})

    assert len(fake.requests) == 1
    signed = fake.requests[0]
    assert signed.method == "POST"
    assert "timestamp=1700000000000" in signed.query_string
    assert "symbol=ETHUSDT" in signed.query_string
    assert "side=BUY" in signed.query_string


@pytest.mark.asyncio
async def test_delete_includes_params() -> None:
    fake = FakeTransport({"orderId": 456, "status": "CANCELED"})
    client = _make_client(transport=fake, timestamp_ms=1700000000000)

    await client.delete("/fapi/v1/order", {"symbol": "ETHUSDT", "orderId": "456"})

    assert len(fake.requests) == 1
    signed = fake.requests[0]
    assert signed.method == "DELETE"
    assert "orderId=456" in signed.query_string


# ---------------------------------------------------------------------------
# X-MBX-APIKEY header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_in_header_not_query() -> None:
    fake = FakeTransport({"ok": True})
    client = _make_client(transport=fake, timestamp_ms=1700000000000)

    await client.get("/fapi/v1/test", {"symbol": "ETHUSDT"})

    signed = fake.requests[0]
    assert "X-MBX-APIKEY" in signed.headers
    assert signed.headers["X-MBX-APIKEY"] == "test-api-key"
    # API key must NOT be in the query string
    assert "test-api-key" not in signed.query_string


# ---------------------------------------------------------------------------
# secret not leaked
# ---------------------------------------------------------------------------


def test_secret_not_in_repr() -> None:
    client = BinancePrivateClient(
        api_key="my-key",
        api_secret="super-secret-123",
        transport=FakeTransport({}),
    )
    r = repr(client)
    assert "super-secret-123" not in r


def test_secret_not_in_request_repr() -> None:
    from src.exchanges.binance.signing import build_signed_request
    req = build_signed_request(
        method="GET",
        path="/fapi/v1/test",
        params={"symbol": "ETHUSDT"},
        api_key="my-key",
        api_secret="my-secret",
        timestamp_ms=1700000000000,
    )
    r = repr(req)
    assert "my-secret" not in r
    assert "signature" not in r.lower()


# ---------------------------------------------------------------------------
# signature stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signature_stable_with_fixed_timestamp() -> None:
    fake = FakeTransport({"ok": True})
    client1 = _make_client(transport=fake, timestamp_ms=1700000000000)
    fake2 = FakeTransport({"ok": True})
    client2 = _make_client(transport=fake2, timestamp_ms=1700000000000)

    await client1.get("/fapi/v1/test", {"symbol": "ETHUSDT"})
    await client2.get("/fapi/v1/test", {"symbol": "ETHUSDT"})

    sig1 = fake.requests[0].query_string
    sig2 = fake2.requests[0].query_string
    assert sig1 == sig2


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_401_raises_auth_error() -> None:
    fake = FakeTransport(
        {"code": -2015, "msg": "Invalid API-key"},
        status_code=401,
    )
    client = _make_client(transport=fake)
    with pytest.raises(ExchangeError) as exc_info:
        await client.get("/fapi/v1/test", {})
    assert exc_info.value.kind == ExchangeErrorKind.AUTH_ERROR


@pytest.mark.asyncio
async def test_binance_error_code_raises() -> None:
    fake = FakeTransport(
        {"code": -2019, "msg": "Margin is insufficient."},
        status_code=200,
    )
    client = _make_client(transport=fake)
    with pytest.raises(ExchangeError) as exc_info:
        await client.get("/fapi/v1/test", {})
    assert exc_info.value.kind == ExchangeErrorKind.INSUFFICIENT_BALANCE


@pytest.mark.asyncio
async def test_success_response_returns_payload() -> None:
    expected = {"orderId": 12345, "status": "NEW"}
    fake = FakeTransport(expected)
    client = _make_client(transport=fake)
    result = await client.post("/fapi/v1/order", {})
    assert result == expected


@pytest.mark.asyncio
async def test_transport_not_started_raises() -> None:
    client = BinancePrivateClient(
        api_key="key",
        api_secret="secret",
    )
    with pytest.raises(RuntimeError, match="transport"):
        await client.get("/fapi/v1/test", {})


# ---------------------------------------------------------------------------
# vanilla GET/POST interface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_payload() -> None:
    expected = [{"asset": "USDT", "balance": "1000"}]
    fake = FakeTransport(expected)
    client = _make_client(transport=fake)
    result = await client.get("/fapi/v2/balance")
    assert result == expected


@pytest.mark.asyncio
async def test_post_returns_payload() -> None:
    expected = {"orderId": 999, "clientOrderId": "cid-1"}
    fake = FakeTransport(expected)
    client = _make_client(transport=fake)
    result = await client.post("/fapi/v1/order", {"symbol": "ETHUSDT"})
    assert result == expected


@pytest.mark.asyncio
async def test_delete_returns_payload() -> None:
    expected = {"orderId": 1, "status": "CANCELED"}
    fake = FakeTransport(expected)
    client = _make_client(transport=fake)
    result = await client.delete("/fapi/v1/order", {"symbol": "ETHUSDT", "orderId": "1"})
    assert result == expected


# ---------------------------------------------------------------------------
# start / close lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_with_existing_transport_is_noop() -> None:
    fake = FakeTransport({"ok": True})
    client = _make_client(transport=fake)
    await client.start()
    # Should not replace the existing transport
    assert client._transport is fake


@pytest.mark.asyncio
async def test_close_clears_transport() -> None:
    fake = FakeTransport({"ok": True})
    client = _make_client(transport=fake)
    await client.close()
    assert client._transport is None
