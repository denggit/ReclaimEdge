#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_signing.py
@Description: Tests for the Binance signed REST request builder.
"""

from __future__ import annotations

from src.exchanges.binance.signing import (
    BINANCE_USDM_BASE_URL,
    BINANCE_USDM_TESTNET_BASE_URL,
    BINANCE_USDM_ORDER_PATH,
    BINANCE_USDM_OPEN_ORDERS_PATH,
    BINANCE_USDM_POSITION_RISK_PATH,
    BinanceSignedRequest,
    binance_api_key_headers,
    build_query_string,
    build_signed_params,
    build_signed_request,
    sign_query_string,
)


# ---------------------------------------------------------------------------
# build_query_string
# ---------------------------------------------------------------------------


def test_build_query_string_skips_none() -> None:
    result = build_query_string({"symbol": "ETHUSDT", "side": None, "quantity": "1"})
    assert "side" not in result
    assert "None" not in result
    assert "symbol=ETHUSDT" in result
    assert "quantity=1" in result


def test_build_query_string_preserves_insertion_order() -> None:
    """Verify that urlencode respects the insertion order of the dict."""
    params = {"timestamp": "1", "symbol": "ETHUSDT", "recvWindow": "5000"}
    result = build_query_string(params)
    # insertion order: timestamp → symbol → recvWindow
    assert result.index("timestamp") < result.index("symbol") < result.index("recvWindow")


# ---------------------------------------------------------------------------
# sign_query_string
# ---------------------------------------------------------------------------


def test_sign_query_string_matches_official_hmac_example() -> None:
    query = (
        "symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1&price=9000"
        "&timeInForce=GTC&recvWindow=5000&timestamp=1591702613943"
    )
    secret = "2b5eb11e18796d12d88f13dc27dbbd02c2cc51ff7059765ed9821957d82bb4d9"
    expected_signature = "3c661234138461fcc7a7d8746c6558c9842d4e10870d2ecbedf7777cad694af9"

    assert sign_query_string(query, secret) == expected_signature


# ---------------------------------------------------------------------------
# build_signed_params
# ---------------------------------------------------------------------------


def test_build_signed_params_adds_recv_window_timestamp_signature(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.exchanges.binance.signing.current_timestamp_ms",
        lambda: 1591702613943,
    )

    result = build_signed_params(
        {"symbol": "ETHUSDT", "side": "BUY"},
        api_secret="test-secret",
        recv_window=5000,
    )

    assert result["recvWindow"] == 5000
    assert result["timestamp"] == 1591702613943
    assert "signature" in result
    # signature must be a 64-character hex string
    assert len(result["signature"]) == 64
    assert all(c in "0123456789abcdef" for c in result["signature"])


def test_build_signed_params_does_not_mutate_original_params() -> None:
    original = {"symbol": "ETHUSDT"}
    _result = build_signed_params(original, api_secret="secret")

    assert original == {"symbol": "ETHUSDT"}
    assert "recvWindow" not in original
    assert "timestamp" not in original
    assert "signature" not in original


def test_build_signed_params_rejects_empty_api_secret() -> None:
    try:
        build_signed_params({"symbol": "ETHUSDT"}, api_secret="")
        assert False, "should have raised"
    except ValueError as exc:
        assert "api_secret" in str(exc)


def test_build_signed_params_accepts_fixed_timestamp() -> None:
    result = build_signed_params(
        {"symbol": "ETHUSDT"},
        api_secret="secret",
        timestamp_ms=1000,
    )
    assert result["timestamp"] == 1000


# ---------------------------------------------------------------------------
# binance_api_key_headers
# ---------------------------------------------------------------------------


def test_binance_api_key_headers_returns_x_mbx_apikey() -> None:
    headers = binance_api_key_headers("my-api-key")
    assert headers == {"X-MBX-APIKEY": "my-api-key"}


def test_binance_api_key_headers_rejects_empty_key() -> None:
    try:
        binance_api_key_headers("")
        assert False, "should have raised"
    except ValueError as exc:
        assert "api_key" in str(exc)


# ---------------------------------------------------------------------------
# build_signed_request
# ---------------------------------------------------------------------------


def test_build_signed_request_returns_correct_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.exchanges.binance.signing.current_timestamp_ms",
        lambda: 1591702613943,
    )

    req = build_signed_request(
        method="post",
        path="/fapi/v1/order",
        params={"symbol": "ETHUSDT", "side": "BUY", "quantity": "1"},
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://fapi.binance.com",
    )

    assert req.method == "POST"
    assert req.path == "/fapi/v1/order"
    assert req.base_url == "https://fapi.binance.com"
    assert req.headers == {"X-MBX-APIKEY": "test-key"}
    assert req.params["symbol"] == "ETHUSDT"
    assert req.params["side"] == "BUY"
    assert req.params["quantity"] == "1"
    assert req.params["recvWindow"] == 5000
    assert req.params["timestamp"] == 1591702613943
    assert "signature" in req.params
    assert len(req.query_string) > 0
    assert "signature=" in req.query_string


def test_build_signed_request_uppercase_method() -> None:
    req = build_signed_request(
        method="get",
        path="/fapi/v1/openOrders",
        params={"symbol": "ETHUSDT"},
        api_key="k",
        api_secret="s",
    )
    assert req.method == "GET"


def test_build_signed_request_default_base_url() -> None:
    req = build_signed_request(
        method="GET",
        path="/fapi/v1/order",
        params={},
        api_key="k",
        api_secret="s",
    )
    assert req.base_url == "https://fapi.binance.com"


# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------


def test_binance_usdm_endpoint_constants() -> None:
    assert BINANCE_USDM_BASE_URL == "https://fapi.binance.com"
    assert BINANCE_USDM_TESTNET_BASE_URL == "https://demo-fapi.binance.com"
    assert BINANCE_USDM_ORDER_PATH == "/fapi/v1/order"
    assert BINANCE_USDM_OPEN_ORDERS_PATH == "/fapi/v1/openOrders"
    assert BINANCE_USDM_POSITION_RISK_PATH == "/fapi/v2/positionRisk"


# ---------------------------------------------------------------------------
# BinanceSignedRequest repr
# ---------------------------------------------------------------------------


def test_signed_request_repr_does_not_include_sensitive_fields() -> None:
    req = BinanceSignedRequest(
        method="POST",
        path="/fapi/v1/order",
        params={
            "symbol": "ETHUSDT",
            "signature": "super-secret-signature",
        },
        headers={"X-MBX-APIKEY": "super-secret-api-key"},
        query_string="symbol=ETHUSDT&signature=super-secret-signature",
    )

    text = repr(req)

    assert "params=" not in text
    assert "headers=" not in text
    assert "query_string" not in text
    assert "super-secret-signature" not in text
    assert "super-secret-api-key" not in text
    assert "X-MBX-APIKEY" not in text

    # fields remain accessible
    assert req.params["signature"] == "super-secret-signature"
    assert req.headers["X-MBX-APIKEY"] == "super-secret-api-key"
