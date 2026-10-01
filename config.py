"""Konfigurasi global dan parameter simulasi."""

from dataclasses import dataclass
from typing import Optional

LOT_SIZE = 100  # 1 lot = 100 lembar saham di BEI

DEFAULT_UNIVERSE = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "ICBP.JK", "INDF.JK", "ANTM.JK", "MDKA.JK",
]

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass
class SimConfig:
    initial_capital: float = 100_000_000
    risk_per_trade: float = 0.01        # fraksi equity yang boleh hilang jika SL kena
    max_positions: int = 5
    max_position_pct: float = 0.25      # fraksi equity maksimum per posisi
    buy_fee: float = 0.0015
    sell_fee: float = 0.0025
    slippage: float = 0.0005
    min_rr: float = 1.0                 # minimum take_profit_pct / stop_loss_pct
    min_avg_traded_value: float = 5_000_000_000  # filter likuiditas (Rp, rata-rata 20 hari)
    decision_every_n_days: int = 1
    max_ai_calls: int = 250
    anonymize: bool = True              # simbol -> STOCK_xx, tanggal -> D0, D1, ...
    seed: int = 42
    sim_start: Optional[str] = None     # "YYYY-MM-DD", batasi jendela uji
    sim_end: Optional[str] = None
