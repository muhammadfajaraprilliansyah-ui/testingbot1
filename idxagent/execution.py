"""Mesin eksekusi virtual: tick size IDX, fill order, SL/TP, sizing.

Model eksekusi:
  * Keputusan dibuat pada CLOSE hari t.
  * Order BUY/EXIT terisi pada OPEN hari t+1 (plus slippage).
  * SL/TP level dihitung dari harga fill sebenarnya (persentase dari agent).
  * SL/TP dicek tiap candle; gap di bawah SL terisi di open, bukan di harga SL.
  * Jika SL dan TP sama-sama tersentuh dalam satu candle, SL dianggap lebih dulu.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import LOT_SIZE, SimConfig


# ---------------------------------------------------------------- tick size
def tick_size(price: float) -> int:
    """Fraksi harga BEI (reguler)."""
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25


def round_tick(price: float, mode: str = "nearest") -> float:
    t = tick_size(price)
    q = price / t
    if mode == "up":
        q = math.ceil(q - 1e-9)
    elif mode == "down":
        q = math.floor(q + 1e-9)
    else:
        q = round(q)
    return float(q * t)


# ------------------------------------------------------------ data classes
@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: str
    entry_idx: int
    entry_value: float
    entry_fee: float


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    quantity: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    holding_days: int
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float


# ----------------------------------------------------------- order parsing
def parse_levels(item: Dict, cfg: SimConfig) -> Tuple[Optional[Dict], Optional[str]]:
    """Validasi SL/TP/size (semua dalam persen) dari keputusan BUY."""
    try:
        sl = float(item["stop_loss_pct"])
        tp = float(item["take_profit_pct"])
        size = item.get("size_pct_equity")
        size = cfg.max_position_pct * 100 if size in (None, "") else float(size)
    except (KeyError, TypeError, ValueError):
        return None, "stop_loss_pct / take_profit_pct tidak ada atau bukan angka"
    if not all(math.isfinite(x) for x in (sl, tp, size)):
        return None, "nilai tidak finite"
    if not 0.5 <= sl <= 30:
        return None, "stop_loss_pct harus 0.5-30"
    if not 0.5 <= tp <= 100:
        return None, "take_profit_pct harus 0.5-100"
    if tp / sl < cfg.min_rr:
        return None, f"risk/reward < {cfg.min_rr}"
    size = min(size, cfg.max_position_pct * 100)
    if size <= 0:
        return None, "size_pct_equity <= 0"
    return {"sl": sl, "tp": tp, "size": size}, None


# --------------------------------------------------------------- fills
def open_position(order: Dict, symbol: str, bar, date_str: str, idx: int,
                  equity: float, cash: float, cfg: SimConfig):
    """Isi BUY pada open. Return (Position|None, cash_baru, alasan_gagal)."""
    fill = min(float(bar["open"]) * (1 + cfg.slippage), float(bar["high"]))
    fill = round_tick(fill, "up")
    stop = round_tick(fill * (1 - order["sl"] / 100), "down")
    target = round_tick(fill * (1 + order["tp"] / 100), "up")
    if stop >= fill or target <= fill:
        return None, cash, "level SL/TP kolaps setelah pembulatan tick"

    risk_per_share = fill - stop
    qty_risk = equity * cfg.risk_per_trade / risk_per_share
    qty_cap = equity * order["size"] / 100 / fill
    qty_cash = cash / (fill * (1 + cfg.buy_fee))
    qty = int(min(qty_risk, qty_cap, qty_cash) // LOT_SIZE) * LOT_SIZE
    if qty < LOT_SIZE:
        return None, cash, "ukuran posisi < 1 lot (risk/cap/cash)"

    value = fill * qty
    fee = value * cfg.buy_fee
    pos = Position(symbol, qty, fill, stop, target, date_str, idx, value, fee)
    return pos, cash - value - fee, None


def check_bar_exit(pos: Position, bar) -> Optional[Tuple[float, str]]:
    o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
    if o <= pos.stop_loss:
        return o, "STOP_LOSS_GAP"
    if l <= pos.stop_loss:
        return pos.stop_loss, "STOP_LOSS"
    if o >= pos.take_profit:
        return o, "TAKE_PROFIT_GAP"
    if h >= pos.take_profit:
        return pos.take_profit, "TAKE_PROFIT"
    return None


def close_position(pos: Position, raw_price: float, date_str: str, idx: int,
                   reason: str, cfg: SimConfig) -> Tuple[Trade, float]:
    price = round_tick(raw_price * (1 - cfg.slippage), "down")
    gross_value = price * pos.quantity
    sell_fee = gross_value * cfg.sell_fee
    gross_pnl = (price - pos.entry_price) * pos.quantity
    fees = pos.entry_fee + sell_fee
    net = gross_pnl - fees
    invested = pos.entry_value + pos.entry_fee
    trade = Trade(pos.symbol, pos.entry_date, date_str, pos.quantity,
                  pos.entry_price, price, pos.stop_loss, pos.take_profit,
                  reason, idx - pos.entry_idx, gross_pnl, fees, net,
                  net / invested * 100)
    return trade, gross_value - sell_fee
