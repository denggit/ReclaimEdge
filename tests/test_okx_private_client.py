from __future__ import annotations

import base64
import hmac
import json
import unittest
from unittest.mock import MagicMock, patch

import src.execution.okx_private_client as client_module
from src.execution.okx_private_client import OkxPrivateClient, OkxPrivateClientConfig


class FakeResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        pass

    def __init__(self, code: str = "0", data: list[dict[str, object]] | None = None) -> None:
        self._code = code
        self._data = data or []

    async def json(self) -> dict[str, object]:
        return {"code": self._code, "data": self._data}


class FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.get_calls: list[dict[str, object]] = []
        self.post_calls: list[dict[str, object]] = []
        self._next_response = FakeResponse()

    def get(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        self.get_calls.append({"url": url, **kwargs})
        return self._next_response

    def post(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        self.post_calls.append({"url": url, **kwargs})
        return self._next_response

    async def close(self) -> None:
        self.closed = True


FIXED_TIMESTAMP = "2025-06-07T12:00:00.000Z"


def make_config(**overrides) -> OkxPrivateClientConfig:
    kwargs = dict(
        base_url="https://www.okx.test",
        api_key="test-api-key",
        secret_key="test-secret-key",
        passphrase="test-passphrase",
        timeout_seconds=5.0,
    )
    kwargs.update(overrides)
    return OkxPrivateClientConfig(**kwargs)


def fixed_timestamp_factory() -> str:
    return FIXED_TIMESTAMP


def expected_signature(timestamp: str, method: str, endpoint: str, body: str, secret_key: str) -> str:
    message = timestamp + method + endpoint + body
    digest = hmac.new(secret_key.encode("utf-8"), message.encode("utf-8"), digestmod="sha256").digest()
    return base64.b64encode(digest).decode("utf-8")


# ============================================================
# headers
# ============================================================


class OkxPrivateClientHeadersTest(unittest.TestCase):
    def test_headers_deterministic_with_timestamp_factory(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        headers = client.headers("POST", "/api/v5/trade/order", '{"a":1}')

        self.assertEqual(headers["OK-ACCESS-KEY"], "test-api-key")
        self.assertEqual(headers["OK-ACCESS-TIMESTAMP"], FIXED_TIMESTAMP)
        self.assertEqual(headers["OK-ACCESS-PASSPHRASE"], "test-passphrase")
        self.assertEqual(headers["Content-Type"], "application/json")

        expected_sig = expected_signature(
            FIXED_TIMESTAMP, "POST", "/api/v5/trade/order", '{"a":1}', "test-secret-key"
        )
        self.assertEqual(headers["OK-ACCESS-SIGN"], expected_sig)

    def test_headers_get_method_empty_body(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        headers = client.headers("GET", "/api/v5/account/balance?ccy=USDT", "")

        expected_sig = expected_signature(
            FIXED_TIMESTAMP, "GET", "/api/v5/account/balance?ccy=USDT", "", "test-secret-key"
        )
        self.assertEqual(headers["OK-ACCESS-SIGN"], expected_sig)

    def test_headers_uses_real_timestamp_when_no_factory(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)
        headers = client.headers("GET", "/api/v5/test", "")
        ts = headers["OK-ACCESS-TIMESTAMP"]
        # Must end with Z and contain T (ISO 8601)
        self.assertTrue(ts.endswith("Z"), f"timestamp must end with Z, got {ts}")
        self.assertIn("T", ts)


# ============================================================
# request
# ============================================================


class OkxPrivateClientRequestTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_request_body_empty_string_in_signature(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        session = FakeSession()
        session._next_response = FakeResponse(code="0", data=[{"result": "ok"}])
        client._session = session

        res = await client.request("GET", "/api/v5/account/balance?ccy=USDT")

        self.assertEqual(res, {"code": "0", "data": [{"result": "ok"}]})
        self.assertEqual(len(session.get_calls), 1)
        call = session.get_calls[0]
        self.assertEqual(call["url"], "https://www.okx.test/api/v5/account/balance?ccy=USDT")
        self.assertEqual(call["headers"]["OK-ACCESS-KEY"], "test-api-key")
        self.assertEqual(call["timeout"], 5.0)

    async def test_post_request_serializes_payload(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        session = FakeSession()
        session._next_response = FakeResponse(code="0", data=[{"ordId": "123"}])
        client._session = session

        res = await client.request("POST", "/api/v5/trade/order", {"a": 1})

        self.assertEqual(res, {"code": "0", "data": [{"ordId": "123"}]})
        self.assertEqual(len(session.post_calls), 1)
        call = session.post_calls[0]
        self.assertEqual(call["url"], "https://www.okx.test/api/v5/trade/order")
        # separators=(",", ":") means no spaces after colons/commas
        self.assertEqual(call["data"], '{"a":1}')
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["timeout"], 5.0)

    async def test_post_request_none_payload_sends_empty_object(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        session = FakeSession()
        session._next_response = FakeResponse(code="0", data=[])
        client._session = session

        await client.request("POST", "/api/v5/trade/order")

        self.assertEqual(session.post_calls[0]["data"], "{}")

    async def test_request_error_code_not_zero_raises_runtime_error(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        session = FakeSession()
        session._next_response = FakeResponse(code="1", data=[{"sMsg": "Invalid request"}])
        client._session = session

        with self.assertRaises(RuntimeError) as ctx:
            await client.request("POST", "/api/v5/trade/order", {"a": 1})
        msg = str(ctx.exception)
        self.assertIn("OKX API error:", msg)
        self.assertIn("method=POST", msg)
        self.assertIn("endpoint=/api/v5/trade/order", msg)
        self.assertIn("response=", msg)

    async def test_request_method_lowercase_is_uppercased(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config, timestamp_factory=fixed_timestamp_factory)
        session = FakeSession()
        session._next_response = FakeResponse(code="0", data=[])
        client._session = session

        await client.request("get", "/api/v5/test")

        self.assertEqual(len(session.get_calls), 1)
        # Verify signature was computed with uppercased method
        headers = session.get_calls[0]["headers"]
        expected_sig = expected_signature(
            FIXED_TIMESTAMP, "GET", "/api/v5/test", "", "test-secret-key"
        )
        self.assertEqual(headers["OK-ACCESS-SIGN"], expected_sig)


# ============================================================
# session lifecycle
# ============================================================


class OkxPrivateClientSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_idempotent(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)
        self.assertIsNone(client._session)

        with patch.object(client_module.aiohttp, "ClientSession", FakeSession, create=True):
            await client.start()
            first_session = client._session
            self.assertIsNotNone(first_session)

            await client.start()
            self.assertIs(client._session, first_session, "start should reuse existing session")

    async def test_start_creates_new_session_when_closed(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)

        with patch.object(client_module.aiohttp, "ClientSession", FakeSession, create=True):
            await client.start()
            first_session = client._session
            await client.close()
            self.assertIsNone(client._session)

            await client.start()
            self.assertIsNotNone(client._session)
            self.assertIsNot(client._session, first_session, "start should create new session after close")

    async def test_close_when_none_is_noop(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)
        # Should not raise
        await client.close()
        self.assertIsNone(client._session)

    async def test_close_sets_session_to_none(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)

        with patch.object(client_module.aiohttp, "ClientSession", FakeSession, create=True):
            await client.start()
            session = client._session
            self.assertIsNotNone(session)

            await client.close()
            self.assertIsNone(client._session)
            self.assertTrue(session.closed)


# ============================================================
# request with start
# ============================================================


class OkxPrivateClientRequestStartTest(unittest.IsolatedAsyncioTestCase):
    async def test_request_calls_start_automatically(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)

        with patch.object(client_module.aiohttp, "ClientSession", FakeSession, create=True):
            self.assertIsNone(client._session)
            await client.request("GET", "/api/v5/test")
            self.assertIsNotNone(client._session)
            self.assertFalse(client._session.closed)

    async def test_request_session_none_after_start_raises(self) -> None:
        config = make_config()
        client = OkxPrivateClient(config)
        # Simulate broken start that leaves session as None
        client._session = None

        async def fake_start() -> None:
            pass  # does NOT create a session

        client.start = fake_start  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            await client.request("GET", "/api/v5/test")
        self.assertIn("OKX private REST session is not initialized", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
