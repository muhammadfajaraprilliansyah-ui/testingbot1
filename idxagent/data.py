"""Download data, indikator teknikal, dan snapshot skala-bebas untuk Claude.

Semua indikator bersifat kausal (hanya memakai data sampai baris itu),
sehingga snapshot hari t aman dari look-ahead bias di level data.
Snapshot sengaja TIDAK memuat harga absolut, supaya simbol/tanggal bisa
dianonimkan tanpa bocor lewat level harga.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_ROWS = 30


def download_symbol(symbol: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period=period, interval="1d",
                     auto_adjust=False, progress=False)
    return clean_ohlcv(df, symbol)


def clean_ohlcv(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"No market data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower().strip() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing {missing}")
    df = df[required].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()
    df = df[(df["volume"] > 0) & (df["high"] >= df["low"])]
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    df["return_1d"] = c.pct_change()
    df["return_5d"] = c.pct_change(5)
    df["return_20d"] = c.pct_change(20)

    for n in (5, 20, 50, 200):
        df[f"ma{n}"] = c.rolling(n).mean()

    delta = c.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100.0

    prev_close = c.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / c * 100

    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    df["avg_traded_value20"] = (c * df["volume"]).rolling(20).mean()

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid * 100

    # shift(1): support/resistance tidak memakai candle hari ini
    df["support20"] = df["low"].rolling(20).min().shift(1)
    df["resistance20"] = df["high"].rolling(20).max().shift(1)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = (c - df["open"]) / c * 100
    df["lower_wick_ratio"] = ((df[["open", "close"]].min(axis=1) - df["low"]) / rng).fillna(0)
    df["upper_wick_ratio"] = ((df["high"] - df[["open", "close"]].max(axis=1)) / rng).fillna(0)

    df["volatility20"] = df["return_1d"].rolling(20).std() * np.sqrt(252) * 100

    # MA200 boleh NaN (periode pendek) -> jangan ikut dropna
    core = [col for col in df.columns if col != "ma200"]
    return df.dropna(subset=core)


def make_snapshot(row: pd.Series) -> Dict:
    """Fitur skala-bebas dari satu baris indikator (hari keputusan)."""
    c = float(row["close"])

    def r(x, n=2):
        return round(float(x), n)

    band = float(row["bb_upper"] - row["bb_lower"])
    bb_pos = (c - float(row["bb_lower"])) / band if band > 0 else 0.5
    ma200 = row["ma200"]

    return {
        "ret_1d_pct": r(row["return_1d"] * 100),
        "ret_5d_pct": r(row["return_5d"] * 100),
        "ret_20d_pct": r(row["return_20d"] * 100),
        "close_vs_ma5_pct": r((c / row["ma5"] - 1) * 100),
        "close_vs_ma20_pct": r((c / row["ma20"] - 1) * 100),
        "close_vs_ma50_pct": r((c / row["ma50"] - 1) * 100),
        "close_vs_ma200_pct": r((c / ma200 - 1) * 100) if pd.notna(ma200) else None,
        "ma20_vs_ma50_pct": r((row["ma20"] / row["ma50"] - 1) * 100),
        "rsi14": r(row["rsi14"]),
        "atr_pct": r(row["atr_pct"]),
        "volatility20_pct": r(row["volatility20"]),
        "volume_ratio": r(row["volume_ratio"]),
        "bb_width_pct": r(row["bb_width"]),
        "bb_position": r(bb_pos, 3),
        "dist_to_support20_pct": r((c / row["support20"] - 1) * 100),
        "dist_to_resistance20_pct": r((row["resistance20"] / c - 1) * 100),
        "positive_candle": bool(row["close"] > row["open"]),
        "body_pct": r(row["body_pct"]),
        "lower_wick_ratio": r(row["lower_wick_ratio"], 3),
        "upper_wick_ratio": r(row["upper_wick_ratio"], 3),
    }


def prepare_market_data(symbols: List[str], period: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    result, errors = {}, []
    for s in symbols:
        try:
            df = calculate_indicators(download_symbol(s, period))
            if len(df) < MIN_ROWS:
                raise ValueError("Data terlalu sedikit.")
            result[s] = df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: {e}")
    return result, errors
