from __future__ import annotations

import asyncio

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import ExchangeName
from src.exchanges.okx.errors import (
    okx_error_detail_from_response,
    okx_error_kind_from_code,
    okx_exception_to_exchange_error,
    okx_retryable_from_kind,
    raise_okx_exchange_error_from_response,
)


class TestOkxErrorKindFromCode:
    def test_auth_code_maps_auth_error(self):
        assert okx_error_kind_from_code("50100") == ExchangeErrorKind.AUTH_ERROR

    def test_auth_message_maps_auth_error(self):
        assert okx_error_kind_from_code(None, "Invalid API key") == ExchangeErrorKind.AUTH_ERROR

    def test_rate_limit_code_maps_rate_limited(self):
        assert okx_error_kind_from_code(50011) == ExchangeErrorKind.RATE_LIMITED

    def test_rate_limit_message_maps_rate_limited(self):
        assert okx_error_kind_from_code(None, "Too many requests") == ExchangeErrorKind.RATE_LIMITED

    def test_timeout_message_maps_timeout_or_network(self):
        assert okx_error_kind_from_code(None, "Request timed out") == ExchangeErrorKind.REQUEST_TIMEOUT

    def test_network_message_maps_network(self):
        assert okx_error_kind_from_code(None, "Connection reset by peer") == ExchangeErrorKind.NETWORK_ERROR

    def test_invalid_symbol_message_maps_invalid_symbol(self):
        assert okx_error_kind_from_code(None, "Instrument does not exist") == ExchangeErrorKind.INVALID_SYMBOL

    def test_invalid_quantity_message_maps_invalid_quantity(self):
        assert okx_error_kind_from_code(None, "Order size below min size") == ExchangeErrorKind.INVALID_QUANTITY

    def test_invalid_price_message_maps_invalid_price(self):
        assert okx_error_kind_from_code(None, "Price precision is invalid") == ExchangeErrorKind.INVALID_PRICE

    def test_invalid_trigger_price_message_maps_invalid_trigger_price(self):
        assert okx_error_kind_from_code(None, "Stop price is invalid") == ExchangeErrorKind.INVALID_TRIGGER_PRICE

    def test_insufficient_margin_code_maps_insufficient_margin(self):
        assert okx_error_kind_from_code("51008") == ExchangeErrorKind.INSUFFICIENT_MARGIN

    def test_insufficient_margin_message_maps_insufficient_margin(self):
        assert okx_error_kind_from_code(None, "Insufficient balance") == ExchangeErrorKind.INSUFFICIENT_MARGIN

    def test_order_not_found_code_maps_order_not_found(self):
        assert okx_error_kind_from_code("51603") == ExchangeErrorKind.ORDER_NOT_FOUND

    def test_order_not_found_message_maps_order_not_found(self):
        assert okx_error_kind_from_code(None, "Order does not exist") == ExchangeErrorKind.ORDER_NOT_FOUND

    def test_reduce_only_message_maps_reduce_only_rejected(self):
        assert okx_error_kind_from_code(None, "Reduce-only order rejected") == ExchangeErrorKind.REDUCE_ONLY_REJECTED

    def test_position_mode_message_maps_position_mode_mismatch(self):
        assert okx_error_kind_from_code(None, "Position side does not match position mode") == ExchangeErrorKind.POSITION_MODE_MISMATCH

    def test_position_not_found_message_maps_position_not_found(self):
        assert okx_error_kind_from_code(None, "Position does not exist") == ExchangeErrorKind.POSITION_NOT_FOUND

    def test_unknown_code_maps_unknown_or_order_rejected_as_expected(self):
        assert okx_error_kind_from_code("99999") == ExchangeErrorKind.UNKNOWN
        assert okx_error_kind_from_code("51500") == ExchangeErrorKind.ORDER_REJECTED


class TestOkxRetryableFromKind:
    def test_rate_limited_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.RATE_LIMITED) is True

    def test_network_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.NETWORK_ERROR) is True

    def test_timeout_retryable_if_supported(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.REQUEST_TIMEOUT) is True

    def test_maintenance_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.EXCHANGE_MAINTENANCE) is True

    def test_auth_not_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.AUTH_ERROR) is False

    def test_invalid_symbol_not_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.INVALID_SYMBOL) is False

    def test_insufficient_margin_not_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.INSUFFICIENT_MARGIN) is False

    def test_order_rejected_not_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.ORDER_REJECTED) is False

    def test_unknown_timeout_message_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.UNKNOWN, message="timeout") is True

    def test_unknown_plain_message_not_retryable(self):
        assert okx_retryable_from_kind(ExchangeErrorKind.UNKNOWN, message="plain failure") is False


class TestOkxErrorDetailFromResponse:
    def test_success_response_returns_none(self):
        response = {"code": "0", "msg": "", "data": [{"sCode": "0"}]}
        assert okx_error_detail_from_response(response) is None

    def test_top_level_error(self):
        response = {"code": "50011", "msg": "Rate limit reached", "data": []}
        detail = okx_error_detail_from_response(response)
        assert detail is not None
        assert detail.exchange == ExchangeName.OKX
        assert detail.kind == ExchangeErrorKind.RATE_LIMITED
        assert detail.code == "50011"
        assert detail.retryable is True
        assert detail.raw == response

    def test_item_level_error(self):
        response = {
            "code": "0",
            "msg": "",
            "data": [
                {"ordId": "", "sCode": "51008", "sMsg": "Insufficient balance"}
            ],
        }
        detail = okx_error_detail_from_response(
            response,
            message="Failed to place market entry",
        )
        assert detail is not None
        assert detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN
        assert detail.code == "51008"
        assert "Failed to place market entry" in detail.message
        assert "Insufficient balance" in detail.message
        assert detail.raw == response

    def test_scans_all_data_items(self):
        response = {
            "code": "0",
            "data": [
                {"ordId": "1", "sCode": "0", "sMsg": ""},
                {"ordId": "", "sCode": "51603", "sMsg": "Order does not exist"},
            ],
        }
        detail = okx_error_detail_from_response(response)
        assert detail is not None
        assert detail.kind == ExchangeErrorKind.ORDER_NOT_FOUND

    def test_missing_top_level_code_but_item_level_error(self):
        response = {"data": [{"sCode": "51008", "sMsg": "Insufficient balance"}]}
        detail = okx_error_detail_from_response(response)
        assert detail is not None
        assert detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN

    def test_malformed_response(self):
        detail = okx_error_detail_from_response("not a mapping")  # type: ignore[arg-type]
        assert detail is not None
        assert detail.kind == ExchangeErrorKind.UNKNOWN
        assert detail.raw["response_type"] == "str"


class TestRaiseOkxExchangeErrorFromResponse:
    def test_success_response_does_not_raise(self):
        response = {"code": "0", "msg": "", "data": [{"sCode": "0"}]}
        assert raise_okx_exchange_error_from_response(response) is None

    def test_top_level_error_raises_exchange_error(self):
        response = {"code": "50011", "msg": "Rate limit reached", "data": []}
        with pytest.raises(ExchangeError) as exc_info:
            raise_okx_exchange_error_from_response(response)
        assert exc_info.value.detail.kind == ExchangeErrorKind.RATE_LIMITED

    def test_item_level_error_raises_exchange_error(self):
        response = {
            "code": "0",
            "data": [{"sCode": "51008", "sMsg": "Insufficient balance"}],
        }
        with pytest.raises(ExchangeError) as exc_info:
            raise_okx_exchange_error_from_response(response)
        assert exc_info.value.detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN

    def test_raw_is_preserved_on_raised_error(self):
        response = {
            "code": "0",
            "data": [{"sCode": "51603", "sMsg": "Order does not exist"}],
        }
        with pytest.raises(ExchangeError) as exc_info:
            raise_okx_exchange_error_from_response(response)
        assert exc_info.value.detail.raw == response


class TestOkxExceptionToExchangeError:
    def test_existing_exchange_error_returned_unchanged(self):
        detail = ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.INVALID_SYMBOL,
            message="bad symbol",
        )
        err = ExchangeError(detail)
        assert okx_exception_to_exchange_error(err) is err

    def test_timeout_error_maps_retryable_timeout_or_network(self):
        err = okx_exception_to_exchange_error(asyncio.TimeoutError("timed out"))
        assert err.detail.kind == ExchangeErrorKind.REQUEST_TIMEOUT
        assert err.detail.retryable is True

    def test_connection_error_maps_network_retryable(self):
        err = okx_exception_to_exchange_error(ConnectionError("connection reset"))
        assert err.detail.kind == ExchangeErrorKind.NETWORK_ERROR
        assert err.detail.retryable is True

    def test_runtime_error_with_parseable_okx_response_maps_detail(self):
        exc = RuntimeError(
            "OKX API error: method=POST endpoint=/api/v5/trade/order "
            "response={'code': '0', 'data': [{'sCode': '51008', 'sMsg': 'Insufficient balance'}]}"
        )
        err = okx_exception_to_exchange_error(exc)
        assert err.detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN
        assert err.detail.code == "51008"
        assert err.detail.raw["response"]["data"][0]["sCode"] == "51008"

    def test_runtime_error_with_unparseable_response_falls_back_to_message_mapping(self):
        exc = RuntimeError("OKX API error: request timed out response={bad")
        err = okx_exception_to_exchange_error(exc)
        assert err.detail.kind == ExchangeErrorKind.REQUEST_TIMEOUT
        assert err.detail.retryable is True

    def test_plain_runtime_insufficient_margin_maps_insufficient_margin(self):
        err = okx_exception_to_exchange_error(RuntimeError("Insufficient margin"))
        assert err.detail.kind == ExchangeErrorKind.INSUFFICIENT_MARGIN

    def test_plain_runtime_unknown_maps_unknown(self):
        err = okx_exception_to_exchange_error(RuntimeError("plain failure"))
        assert err.detail.kind == ExchangeErrorKind.UNKNOWN
