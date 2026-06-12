from src.position_management.protective_sl_policy import (
    has_existing_protective_sl,
    is_existing_sl_at_least_as_protective,
)


def test_long_existing_higher_is_at_least_as_protective() -> None:
    assert is_existing_sl_at_least_as_protective(
        side="LONG",
        existing_sl_price=105.0,
        candidate_sl_price=104.0,
    )


def test_long_existing_lower_is_weaker() -> None:
    assert not is_existing_sl_at_least_as_protective(
        side="LONG",
        existing_sl_price=103.0,
        candidate_sl_price=104.0,
    )


def test_short_existing_lower_is_at_least_as_protective() -> None:
    assert is_existing_sl_at_least_as_protective(
        side="SHORT",
        existing_sl_price=95.0,
        candidate_sl_price=96.0,
    )


def test_short_existing_higher_is_weaker() -> None:
    assert not is_existing_sl_at_least_as_protective(
        side="SHORT",
        existing_sl_price=97.0,
        candidate_sl_price=96.0,
    )


def test_candidate_none_does_not_compare_as_stronger() -> None:
    assert has_existing_protective_sl(
        old_sl_order_id="old-sl",
        old_sl_price=105.0,
        old_protected=True,
    )
    assert not is_existing_sl_at_least_as_protective(
        side="LONG",
        existing_sl_price=105.0,
        candidate_sl_price=None,
    )


def test_missing_old_order_id_is_not_existing_protection() -> None:
    assert not has_existing_protective_sl(
        old_sl_order_id=None,
        old_sl_price=105.0,
        old_protected=True,
    )
