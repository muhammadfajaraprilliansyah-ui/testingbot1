"""Benchmark pembanding: buy&hold universe, IHSG, dan agent acak."""

from typing import Dict, List

import numpy as np
import pandas as pd

from .agent import RandomAgent
from .backtest import common_dates, run_backtest
from .config import SimConfig


def buy_and_hold(symbol_data: Dict[str, pd.DataFrame], cfg: SimConfig) -> pd.DataFrame:
    """Equal-weight, beli di close hari pertama, jual di close hari terakhir, biaya 1x."""
    dates = common_dates(symbol_data, cfg)
    n = len(symbol_data)
    closes = pd.DataFrame({s: df.loc[dates, "close"] for s, df in symbol_data.items()})
    alloc = cfg.initial_capital / n
    shares = alloc / (1 + cfg.buy_fee) / closes.iloc[0]
    eq = (closes * shares).sum(axis=1)
    eq.iloc[-1] -= (closes.iloc[-1] * shares).sum() * cfg.sell_fee
    return pd.DataFrame({"date": dates, "equity": eq.values})


def index_buy_and_hold(index_df: pd.DataFrame, dates: List[pd.Timestamp], cfg: SimConfig) -> pd.DataFrame:
    close = index_df["close"].reindex(dates).ffill().bfill()
    eq = cfg.initial_capital * close / close.iloc[0]
    return pd.DataFrame({"date": dates, "equity": eq.values})


def random_baseline(symbol_data: Dict[str, pd.DataFrame], cfg: SimConfig, runs: int = 10) -> List[float]:
    """Return akhir (%) dari beberapa agent acak dengan aturan eksekusi yang sama."""
    out = []
    for k in range(runs):
        res = run_backtest(symbol_data, RandomAgent(seed=k), cfg)
        out.append(float(res.equity["equity"].iloc[-1] / cfg.initial_capital * 100 - 100))
    return out
