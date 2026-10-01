"""Loop simulasi harian dengan eksekusi next-open."""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from .agent import AgentFatalError
from .config import SimConfig
from .data import make_snapshot
from .execution import (Position, Trade, check_bar_exit, close_position,
                        open_position, parse_levels)


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity: pd.DataFrame
    log: List[Dict]
    ai_calls: int
    usage: Dict = field(default_factory=dict)
    budget_exhausted_on: Optional[str] = None


def common_dates(symbol_data: Dict[str, pd.DataFrame], cfg: SimConfig) -> List[pd.Timestamp]:
    idx = None
    for df in symbol_data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    if idx is None or len(idx) == 0:
        raise ValueError("Tidak ada tanggal yang sama antar simbol.")
    idx = idx.sort_values()
    if cfg.sim_start:
        idx = idx[idx >= pd.Timestamp(cfg.sim_start)]
    if cfg.sim_end:
        idx = idx[idx <= pd.Timestamp(cfg.sim_end)]
    if len(idx) < 5:
        raise ValueError("Jendela simulasi terlalu pendek (< 5 hari).")
    return list(idx)


def run_backtest(symbol_data: Dict[str, pd.DataFrame], agent, cfg: SimConfig,
                 progress_cb: Optional[Callable[[int, int, int], None]] = None) -> BacktestResult:
    dates = common_dates(symbol_data, cfg)
    symbols = sorted(symbol_data)

    # ---- anonimisasi (alias simbol dikocok per seed)
    if cfg.anonymize:
        shuffled = symbols[:]
        random.Random(cfg.seed).shuffle(shuffled)
        alias = {s: f"STOCK_{i + 1:02d}" for i, s in enumerate(shuffled)}
    else:
        alias = {s: s.replace(".JK", "") for s in symbols}
    alias_to_sym = {a: s for s, a in alias.items()}

    def label(i: int) -> str:
        return f"D{i}" if cfg.anonymize else dates[i].date().isoformat()

    cash = float(cfg.initial_capital)
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    rows: List[Dict] = []
    log: List[Dict] = []
    pending: List[Dict] = []
    ai_calls = 0
    exhausted_on = None
    bars = {}

    def record_close(trade: Trade, entry_idx: int, exit_idx: int):
        trades.append(trade)
        agent.add_experience({
            "symbol": alias[trade.symbol], "entry": label(entry_idx), "exit": label(exit_idx),
            "return_pct": round(trade.return_pct, 2), "exit_reason": trade.exit_reason,
            "days_held": trade.holding_days,
        })

    for i, date in enumerate(dates):
        dstr = date.date().isoformat()
        bars = {s: symbol_data[s].loc[date] for s in symbols}

        # ---- 1. isi order dari keputusan kemarin (open hari ini)
        if pending:
            for o in [p for p in pending if p["action"] == "EXIT"]:
                pos = positions.get(o["symbol"])
                if pos is None:
                    continue
                trade, cash_in = close_position(pos, float(bars[o["symbol"]]["open"]), dstr, i, "CLAUDE_EXIT", cfg)
                cash += cash_in
                record_close(trade, pos.entry_idx, i)
                del positions[o["symbol"]]

            for o in [p for p in pending if p["action"] == "BUY"]:
                sym = o["symbol"]
                if sym in positions or len(positions) >= cfg.max_positions:
                    log.append({"date": label(i), "rejected": alias[sym], "reason": "slot penuh / sudah dimiliki"})
                    continue
                equity_open = cash + sum(float(bars[s]["open"]) * p.quantity for s, p in positions.items())
                pos, new_cash, why = open_position(o, sym, bars[sym], dstr, i, equity_open, cash, cfg)
                if pos is None:
                    log.append({"date": label(i), "rejected": alias[sym], "reason": why})
                    continue
                positions[sym], cash = pos, new_cash
            pending = []

        # ---- 2. cek SL/TP intraday (termasuk posisi yang baru dibuka)
        for sym in list(positions):
            res = check_bar_exit(positions[sym], bars[sym])
            if res:
                price, reason = res
                pos = positions[sym]
                trade, cash_in = close_position(pos, price, dstr, i, reason, cfg)
                cash += cash_in
                record_close(trade, pos.entry_idx, i)
                del positions[sym]

        # ---- 3. mark-to-market di close
        pos_value = sum(float(bars[s]["close"]) * p.quantity for s, p in positions.items())
        equity = cash + pos_value
        rows.append({"date": date, "equity": equity, "cash": cash,
                     "position_value": pos_value, "open_positions": len(positions)})
        if progress_cb:
            progress_cb(i + 1, len(dates), ai_calls)

        # ---- 4. keputusan (untuk eksekusi besok)
        if i == len(dates) - 1 or i % cfg.decision_every_n_days != 0:
            continue
        if ai_calls >= cfg.max_ai_calls:
            exhausted_on = exhausted_on or label(i)
            continue

        market = {}
        for s in symbols:
            row = symbol_data[s].loc[date]
            if row["avg_traded_value20"] >= cfg.min_avg_traded_value:
                market[alias[s]] = make_snapshot(row)

        open_list = []
        for s, p in positions.items():
            c = float(bars[s]["close"])
            open_list.append({
                "symbol": alias[s],
                "days_held": i - p.entry_idx,
                "unrealized_pct": round((c / p.entry_price - 1) * 100, 2),
                "weight_pct": round(c * p.quantity / equity * 100, 1),
                "stop_loss_pct_from_entry": round((1 - p.stop_loss / p.entry_price) * 100, 2),
                "take_profit_pct_from_entry": round((p.take_profit / p.entry_price - 1) * 100, 2),
                "distance_to_stop_pct": round((c / p.stop_loss - 1) * 100, 2),
                "distance_to_target_pct": round((p.take_profit / c - 1) * 100, 2),
            })

        ctx = {
            "day": label(i),
            "rules": {
                "max_positions": cfg.max_positions,
                "open_slots": cfg.max_positions - len(positions),
                "max_position_pct": cfg.max_position_pct * 100,
                "risk_per_trade_pct": cfg.risk_per_trade * 100,
                "round_trip_cost_pct": round((cfg.buy_fee + cfg.sell_fee + 2 * cfg.slippage) * 100, 3),
                "fill": "next trading day open",
            },
            "portfolio": {
                "equity_change_pct": round((equity / cfg.initial_capital - 1) * 100, 2),
                "cash_pct": round(cash / equity * 100, 1),
                "open_positions": open_list,
            },
            "market": market,
        }

        try:
            out = agent.decide(ctx)
            ai_calls += 1
            log.append({"date": label(i), "decision": out})
        except AgentFatalError:
            raise
        except Exception as e:  # noqa: BLE001
            log.append({"date": label(i), "error": str(e)})
            continue

        seen = set()
        for item in out.get("decisions", []) or []:
            try:
                action = str(item.get("action", "")).upper()
                sym = alias_to_sym.get(str(item.get("symbol", "")).upper().replace(".JK", ""))
                if sym is None or sym in seen:
                    raise ValueError("simbol tidak dikenal / duplikat dalam satu respons")
                if action == "EXIT":
                    if sym not in positions:
                        raise ValueError("tidak ada posisi untuk di-EXIT")
                    pending.append({"action": "EXIT", "symbol": sym})
                elif action == "BUY":
                    if alias[sym] not in market:
                        raise ValueError("simbol tidak lolos filter likuiditas / tidak tersedia")
                    if sym in positions:
                        raise ValueError("posisi sudah terbuka")
                    exits = sum(p["action"] == "EXIT" for p in pending)
                    buys = sum(p["action"] == "BUY" for p in pending)
                    if len(positions) - exits + buys >= cfg.max_positions:
                        raise ValueError("slot posisi penuh")
                    levels, err = parse_levels(item, cfg)
                    if err:
                        raise ValueError(err)
                    pending.append({"action": "BUY", "symbol": sym, **levels})
                else:
                    raise ValueError(f"action tidak dikenal: {action}")
                seen.add(sym)
            except Exception as e:  # noqa: BLE001
                log.append({"date": label(i), "rejected": str(item.get("symbol")), "reason": str(e)})

    # ---- tutup semua posisi di akhir uji
    last = len(dates) - 1
    for sym in list(positions):
        pos = positions[sym]
        trade, cash_in = close_position(pos, float(bars[sym]["close"]), dates[last].date().isoformat(),
                                        last, "END_OF_TEST", cfg)
        cash += cash_in
        record_close(trade, pos.entry_idx, last)
        del positions[sym]

    equity_df = pd.DataFrame(rows)
    last_row = equity_df.index[-1]
    equity_df.loc[last_row, ["equity", "cash"]] = cash
    equity_df.loc[last_row, ["position_value", "open_positions"]] = 0
    return BacktestResult(trades, equity_df, log, ai_calls, dict(getattr(agent, "usage", {})), exhausted_on)
