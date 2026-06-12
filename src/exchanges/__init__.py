#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : __init__.py
@Description: Exchange abstraction layer – ports and adapter skeleton.

This package defines generic broker ports (BrokerClient, BrokerSemanticExecutor)
and domain models (BrokerOrder, BrokerPosition, ...) that are independent of
any specific exchange.  Concrete adapters live in sub‑packages (okx/, binance/,
bybit/ …).
"""
