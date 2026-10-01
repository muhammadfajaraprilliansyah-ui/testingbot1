"""Manajemen penyimpanan persisten hasil run backtest ke disk (JSON/Parquet).

Setiap run disimpan ke dalam direktori mandiri di bawah runs/:
  - summary.json: metadata run, metrik performa, dan token usage
  - config.json: parameter SimConfig yang digunakan
  - trades.json & trades.parquet: daftar trade historis
  - equity.parquet & equity.csv: kurva equity dan alokasi harian
  - decisions.json: log keputusan harian Claude/agent
  - agent_memory.json: rekaman memori dan catatan pembelajaran agent
"""

from dataclasses import asdict
from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .config import SimConfig
from .execution import Trade
from .metrics import calculate_metrics


def _json_serial(obj: Any) -> Any:
    """Helper serialisasi untuk tipe non-standar ke JSON."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Trade):
        return asdict(obj)
    if isinstance(obj, SimConfig):
        return asdict(obj)
    raise TypeError(f"Tipe {type(obj)} tidak dapat diserialisasi ke JSON")


def sanitize_run_name(name: str) -> str:
    """Membersihkan string nama agar aman digunakan sebagai nama direktori."""
    clean = re.sub(r"[^\w\-]+", "_", name.strip()).strip("_")
    return clean[:40] if clean else "run"


def save_run(
    result: BacktestResult,
    cfg: SimConfig,
    agent: Any = None,
    metrics: Optional[Dict] = None,
    name: str = "",
    base_dir: str = "runs",
    save_parquet: bool = True,
) -> str:
    """Menyimpan artefak run ke direktori disk.

    Return: path absolut/relatif direktori run yang dibuat.
    """
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = sanitize_run_name(name) if name else "run"
    run_id = f"{ts_str}_{clean_name}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    if metrics is None:
        metrics = calculate_metrics(result.trades, result.equity, cfg.initial_capital)

    # 1. config.json
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=_json_serial, ensure_ascii=False)

    # 2. summary.json
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            if np.isinf(v):
                clean_metrics[k] = "inf" if v > 0 else "-inf"
            elif np.isnan(v):
                clean_metrics[k] = None
            else:
                clean_metrics[k] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            clean_metrics[k] = int(v)
        else:
            clean_metrics[k] = v

    summary = {
        "run_id": run_id,
        "name": name or run_id,
        "timestamp": datetime.now().isoformat(),
        "ai_calls": result.ai_calls,
        "usage": result.usage,
        "budget_exhausted_on": result.budget_exhausted_on,
        "metrics": clean_metrics,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_serial, ensure_ascii=False)

    # 3. trades (json & parquet)
    trades_data = [asdict(t) for t in result.trades]
    trades_json_path = os.path.join(run_dir, "trades.json")
    with open(trades_json_path, "w", encoding="utf-8") as f:
        json.dump(trades_data, f, indent=2, default=_json_serial, ensure_ascii=False)

    if save_parquet:
        trades_df = pd.DataFrame(trades_data)
        if not trades_df.empty:
            trades_df.to_parquet(os.path.join(run_dir, "trades.parquet"), index=False)

    # 4. equity (csv & parquet)
    if result.equity is not None and not result.equity.empty:
        eq_df = result.equity.copy()
        if "date" in eq_df.columns:
            eq_df["date"] = eq_df["date"].astype(str)
        eq_df.to_csv(os.path.join(run_dir, "equity.csv"), index=False)
        if save_parquet:
            eq_df.to_parquet(os.path.join(run_dir, "equity.parquet"), index=False)

    # 5. decisions.json
    decisions_path = os.path.join(run_dir, "decisions.json")
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(result.log, f, indent=2, default=_json_serial, ensure_ascii=False)

    # 6. agent_memory.json (jika tersedia)
    if agent is not None and hasattr(agent, "memory_summary"):
        mem_path = os.path.join(run_dir, "agent_memory.json")
        try:
            mem_data = agent.memory_summary()
            with open(mem_path, "w", encoding="utf-8") as f:
                json.dump(mem_data, f, indent=2, default=_json_serial, ensure_ascii=False)
        except Exception:
            pass

    return run_dir


def load_run(run_dir: str) -> Dict[str, Any]:
    """Membaca hasil run lengkap dari direktori run."""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Direktori run '{run_dir}' tidak ditemukan.")

    # 1. summary
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 2. config
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = SimConfig(**cfg_dict)

    # 3. trades
    trades = []
    trades_parquet = os.path.join(run_dir, "trades.parquet")
    trades_json = os.path.join(run_dir, "trades.json")
    if os.path.exists(trades_parquet):
        tdf = pd.read_parquet(trades_parquet)
        trades = [Trade(**row) for row in tdf.to_dict(orient="records")]
    elif os.path.exists(trades_json):
        with open(trades_json, "r", encoding="utf-8") as f:
            t_list = json.load(f)
        trades = [Trade(**t) for t in t_list]

    # 4. equity
    eq_parquet = os.path.join(run_dir, "equity.parquet")
    eq_csv = os.path.join(run_dir, "equity.csv")
    if os.path.exists(eq_parquet):
        equity_df = pd.read_parquet(eq_parquet)
    elif os.path.exists(eq_csv):
        equity_df = pd.read_csv(eq_csv)
    else:
        equity_df = pd.DataFrame()
    if not equity_df.empty and "date" in equity_df.columns:
        equity_df["date"] = pd.to_datetime(equity_df["date"])

    # 5. decisions
    decisions = []
    dec_path = os.path.join(run_dir, "decisions.json")
    if os.path.exists(dec_path):
        with open(dec_path, "r", encoding="utf-8") as f:
            decisions = json.load(f)

    # 6. agent_memory
    agent_memory = {}
    mem_path = os.path.join(run_dir, "agent_memory.json")
    if os.path.exists(mem_path):
        with open(mem_path, "r", encoding="utf-8") as f:
            agent_memory = json.load(f)

    res = BacktestResult(
        trades=trades,
        equity=equity_df,
        log=decisions,
        ai_calls=summary.get("ai_calls", 0),
        usage=summary.get("usage", {}),
        budget_exhausted_on=summary.get("budget_exhausted_on"),
    )

    return {
        "run_id": summary.get("run_id", os.path.basename(run_dir)),
        "name": summary.get("name", os.path.basename(run_dir)),
        "timestamp": summary.get("timestamp"),
        "res": res,
        "cfg": cfg,
        "metrics": summary.get("metrics", {}),
        "agent_memory": agent_memory,
        "path": run_dir,
    }


def list_runs(base_dir: str = "runs") -> List[Dict[str, Any]]:
    """Mendaftar semua metadata run yang tersimpan di disk, diurutkan terbaru."""
    if not os.path.exists(base_dir):
        return []

    items = []
    for entry in sorted(os.listdir(base_dir), reverse=True):
        entry_path = os.path.join(base_dir, entry)
        summary_path = os.path.join(entry_path, "summary.json")
        if os.path.isdir(entry_path) and os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                items.append({
                    "run_id": summary.get("run_id", entry),
                    "name": summary.get("name", entry),
                    "timestamp": summary.get("timestamp"),
                    "ai_calls": summary.get("ai_calls", 0),
                    "trades": summary.get("metrics", {}).get("trades", 0),
                    "return_pct": summary.get("metrics", {}).get("return_pct", 0.0),
                    "win_rate": summary.get("metrics", {}).get("win_rate", 0.0),
                    "sharpe": summary.get("metrics", {}).get("sharpe", 0.0),
                    "max_drawdown_pct": summary.get("metrics", {}).get("max_drawdown_pct", 0.0),
                    "path": entry_path,
                })
            except Exception:
                continue
    return items


def compare_saved_runs(base_dir: str = "runs") -> pd.DataFrame:
    """Mengembalikan DataFrame ringkasan komparasi metrik seluruh run yang tersimpan."""
    runs = list_runs(base_dir)
    if not runs:
        return pd.DataFrame()
    rows = []
    for r in runs:
        summary_path = os.path.join(r["path"], "summary.json")
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            m = s.get("metrics", {})
            rows.append({
                "run_id": r["run_id"],
                "name": r["name"],
                "ai_calls": s.get("ai_calls", 0),
                "trades": m.get("trades", 0),
                "win_rate_pct": m.get("win_rate", 0.0),
                "return_pct": m.get("return_pct", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
                "max_dd_pct": m.get("max_drawdown_pct", 0.0),
                "sharpe": m.get("sharpe", 0.0),
                "timestamp": r.get("timestamp"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)
