from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from src.live.supervisor.multi_symbol_supervisor import MultiSymbolSupervisor


@dataclass(frozen=True)
class _Config:
    child_name: str


class _FakeSupervisor:
    def __init__(self, child_name: str, *, fail: bool = False) -> None:
        self.config = _Config(child_name)
        self.fail = fail
        self.started = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.stop_requested = False
        self.shutdown_called = 0

    def request_stop(self) -> None:
        self.stop_requested = True
        self.stop_event.set()

    async def shutdown(self) -> None:
        self.shutdown_called += 1
        self.request_stop()

    async def run(self) -> None:
        self.started.set()
        if self.fail:
            raise RuntimeError(f"{self.config.child_name} failed")
        await self.stop_event.wait()


@pytest.mark.asyncio
async def test_multi_symbol_supervisor_starts_two_supervisors_concurrently() -> None:
    eth = _FakeSupervisor("reclaim-worker-ETH-USDT-SWAP")
    btc = _FakeSupervisor("reclaim-worker-BTC-USDT-SWAP")
    supervisor = MultiSymbolSupervisor([eth, btc])

    task = asyncio.create_task(supervisor.run())
    await asyncio.wait_for(eth.started.wait(), timeout=1)
    await asyncio.wait_for(btc.started.wait(), timeout=1)

    supervisor.request_stop()
    return_code = await asyncio.wait_for(task, timeout=1)

    assert return_code == 0
    assert eth.stop_requested is True
    assert btc.stop_requested is True


@pytest.mark.asyncio
async def test_one_supervisor_exception_does_not_cancel_other() -> None:
    eth = _FakeSupervisor("reclaim-worker-ETH-USDT-SWAP", fail=True)
    btc = _FakeSupervisor("reclaim-worker-BTC-USDT-SWAP")
    supervisor = MultiSymbolSupervisor([eth, btc])

    task = asyncio.create_task(supervisor.run())
    await asyncio.wait_for(btc.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert task.done() is False
    assert btc.stop_requested is False

    supervisor.request_stop()
    return_code = await asyncio.wait_for(task, timeout=1)

    assert return_code == 1
    assert len(supervisor.task_results) == 2
    assert any(result.error for result in supervisor.task_results)


@pytest.mark.asyncio
async def test_shutdown_notifies_all_supervisors() -> None:
    eth = _FakeSupervisor("reclaim-worker-ETH-USDT-SWAP")
    btc = _FakeSupervisor("reclaim-worker-BTC-USDT-SWAP")
    supervisor = MultiSymbolSupervisor([eth, btc])

    await supervisor.shutdown()

    assert eth.stop_requested is True
    assert btc.stop_requested is True
    assert eth.shutdown_called == 1
    assert btc.shutdown_called == 1
