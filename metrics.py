"""Metrik performa."""

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_metrics(trades: List, equity: pd.DataFrame, initial_capital: float) -> Dict:
    base = {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "return_pct": 0.0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_exposure_pct": 0.0}
    if equity.empty:
        return base

    eq = equity["equity"].astype(float)
    base["net_pnl"] = float(eq.iloc[-1] - initial_capital)
    base["return_pct"] = base["net_pnl"] / initial_capital * 100
    base["max_drawdown_pct"] = float((eq / eq.cummax() - 1).min() * 100)
    daily = eq.pct_change().dropna()
    if len(daily) > 1 and daily.std() > 0:
        base["sharpe"] = float(daily.mean() / daily.std() * np.sqrt(252))
    base["avg_exposure_pct"] = float((equity["position_value"] / eq).mean() * 100)

    if not trades:
        return base
    df = pd.DataFrame([asdict(t) for t in trades])
    wins, losses = df[df["net_pnl"] > 0], df[df["net_pnl"] < 0]
    gp, gl = wins["net_pnl"].sum(), abs(losses["net_pnl"].sum())
    base.update({
        "trades": len(df),
        "win_rate": len(wins) / len(df) * 100,
        "profit_factor": (np.inf if gp > 0 else 0.0) if gl == 0 else gp / gl,
        "avg_win": float(wins["net_pnl"].mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses["net_pnl"].mean()) if not losses.empty else 0.0,
    })
    return base
