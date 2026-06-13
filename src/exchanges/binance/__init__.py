#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : __init__.py
@Description: Binance exchange adapter package.

Real broker execution is deferred to a later task.
"""

from src.exchanges.binance.client import BinanceBrokerClient

__all__ = ["BinanceBrokerClient"]
