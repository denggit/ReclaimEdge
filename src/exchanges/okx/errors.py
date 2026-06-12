from __future__ import annotations

from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import ExchangeName


def okx_exception_to_exchange_error(exc: Exception, *, message: str | None = None) -> ExchangeError:
    """Wrap any exception into an ExchangeError with ExchangeName.OKX.

    If *exc* is already an ``ExchangeError``, it is returned unchanged.
    Otherwise, a new ``ExchangeError`` is created with:
      - ``exchange=OKX``
      - ``kind=UNKNOWN``
      - ``message`` = *message* if provided, else ``str(exc)``
      - ``raw`` contains ``exception_type``.

    Detailed OKX ``sCode`` / ``sMsg`` mapping will be added in a later step.
    """
    if isinstance(exc, ExchangeError):
        return exc

    return ExchangeError(
        ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.UNKNOWN,
            message=message or str(exc),
            raw={"exception_type": type(exc).__name__},
        )
    )
