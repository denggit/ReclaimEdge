from dataclasses import fields

from src.exchanges.models import ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)


EXCHANGE_RAW_FIELD_NAMES = {"ordId", "algoId", "instId", "posSide", "sCode", "sMsg"}


def test_semantic_request_has_no_okx_specific_fields():
    field_names = {field.name for field in fields(BrokerSemanticRequest)}

    assert field_names.isdisjoint(EXCHANGE_RAW_FIELD_NAMES)


def test_semantic_result_has_no_okx_specific_fields():
    field_names = {field.name for field in fields(BrokerSemanticResult)}

    assert field_names.isdisjoint(EXCHANGE_RAW_FIELD_NAMES)


def test_semantic_request_defaults():
    request = BrokerSemanticRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        action=BrokerSemanticAction.OPEN_POSITION,
    )

    assert request.role == BrokerSemanticOrderRole.UNKNOWN
    assert request.reduce_only is False
    assert request.close_position is False
    assert request.metadata == {}


def test_semantic_result_defaults():
    result = BrokerSemanticResult(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        action=BrokerSemanticAction.OPEN_POSITION,
    )

    assert result.ok is False
    assert result.message == ""
    assert result.related_order_ids == ()


def test_order_roles_cover_reclaimedge_runtime_roles():
    roles = {role.name for role in BrokerSemanticOrderRole}

    assert {
        "CORE_TP",
        "PROTECTIVE_SL",
        "SIDECAR_ENTRY",
        "SIDECAR_TP",
        "MARKET_EXIT",
        "MIDDLE_RUNNER_SL",
        "THREE_STAGE_SL",
        "TREND_RUNNER_SL",
    }.issubset(roles)


def test_semantic_models_do_not_contain_exchange_raw_fields():
    source = _read_semantic_models_source()

    assert all(field_name not in source for field_name in EXCHANGE_RAW_FIELD_NAMES)


def _read_semantic_models_source() -> str:
    from pathlib import Path

    return Path("src/exchanges/semantic_models.py").read_text()
