"""Streamlit UI: Autonomous Claude IDX Trading Lab (simulasi, tanpa broker)."""

import json
import os
from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from idxagent.agent import ClaudeAgent
from idxagent.backtest import common_dates, run_backtest
from idxagent.benchmarks import buy_and_hold, index_buy_and_hold, random_baseline
from idxagent.config import DEFAULT_MODEL, DEFAULT_UNIVERSE, SimConfig
from idxagent.data import clean_ohlcv, prepare_market_data
from idxagent.metrics import calculate_metrics
from idxagent.storage import list_runs, load_run, save_run

st.set_page_config(page_title="Autonomous Claude IDX Trading Lab", page_icon="🤖", layout="wide")
st.title("🤖 Autonomous Claude IDX Trading Lab")
st.caption("Simulasi historis. Tanpa broker, tanpa uang nyata. Bukan saran keuangan.")


@st.cache_data(show_spinner=False)
def load_data(symbols: tuple, period: str):
    return prepare_market_data(list(symbols), period)


@st.cache_data(show_spinner=False)
def load_index(period: str):
    import yfinance as yf
    return clean_ohlcv(yf.download("^JKSE", period=period, interval="1d",
                                   auto_adjust=False, progress=False), "^JKSE")


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Konfigurasi")
api_key = sb.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", "")).strip()
model = sb.text_input("Model", value=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)).strip()
capital = sb.number_input("Modal virtual (Rp)", min_value=1_000_000, value=100_000_000, step=1_000_000)
risk = sb.slider("Risiko maksimum / trade (%)", 0.1, 5.0, 1.0, 0.1)
max_pos = sb.slider("Maksimum posisi terbuka", 1, 10, 5)
max_pct = sb.slider("Maksimum modal / posisi (%)", 5, 50, 25) / 100
buy_fee = sb.number_input("Buy fee", 0.0, 0.02, 0.0015, 0.0001, format="%.4f")
sell_fee = sb.number_input("Sell fee", 0.0, 0.02, 0.0025, 0.0001, format="%.4f")
slip = sb.number_input("Slippage", 0.0, 0.02, 0.0005, 0.0001, format="%.4f")
period = sb.selectbox("Periode data", ["1y", "2y", "5y"], index=1)
every_n = sb.slider("Keputusan Claude setiap N hari bursa", 1, 10, 1)
max_calls = sb.number_input("Maks. panggilan Claude", 1, 2000, 250, 10)
anonymize = sb.checkbox("Anonimkan simbol & tanggal (disarankan)", value=True)
use_window = sb.checkbox("Batasi jendela uji (mis. out-of-sample setelah cutoff model)")
start = end = None
if use_window:
    start_val = sb.date_input("Mulai", value=date(2026, 7, 1))
    end_val = sb.date_input("Akhir", value=date.today())
    if start_val > end_val:
        sb.error("Tanggal 'Mulai' tidak boleh melebihi tanggal 'Akhir'.")
    start = start_val.isoformat()
    end = end_val.isoformat()
n_random = sb.slider("Jumlah agent acak (baseline)", 0, 20, 10)
symbols = sb.multiselect("Universe IDX", DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)

if not api_key:
    st.info("💡 Masukkan Anthropic API Key di sidebar untuk menjalankan simulasi Claude.")

# Riwayat run tersimpan di disk
saved_disk_runs = list_runs("runs")
if saved_disk_runs:
    with sb.expander(f"📁 Run di Disk ({len(saved_disk_runs)})", expanded=False):
        run_opts = {f"{r['name']} ({r['timestamp'][:16] if r.get('timestamp') else ''})": r["path"] for r in saved_disk_runs}
        chosen_label = sb.selectbox("Pilih riwayat run:", list(run_opts.keys()))
        if sb.button("📂 Muat run terpilih"):
            try:
                loaded = load_run(run_opts[chosen_label])
                loaded_name = f"{loaded['name']} (disk)"
                if not any(r["name"] == loaded_name for r in st.session_state.runs):
                    bench = {"Claude": loaded["res"].equity[["date", "equity"]]} if loaded["res"].equity is not None and not loaded["res"].equity.empty else {}
                    st.session_state.runs.append({
                        "name": loaded_name,
                        "res": loaded["res"],
                        "agent": None,
                        "agent_memory": loaded.get("agent_memory", {}),
                        "metrics": loaded["metrics"],
                        "bench": bench,
                        "rand": [],
                        "cfg": loaded["cfg"],
                        "saved_dir": loaded["path"],
                    })
                    st.success(f"Run '{loaded_name}' berhasil dimuat!")
                    st.rerun()
            except Exception as err:
                sb.error(f"Gagal memuat: {err}")

cfg = SimConfig(initial_capital=capital, risk_per_trade=risk / 100, max_positions=max_pos,
                max_position_pct=max_pct, buy_fee=buy_fee, sell_fee=sell_fee, slippage=slip,
                decision_every_n_days=every_n, max_ai_calls=int(max_calls),
                anonymize=anonymize, sim_start=start, sim_end=end)

if "runs" not in st.session_state:
    st.session_state.runs = []

# --------------------------------------------------------------------- run
if st.button("🚀 Jalankan simulasi", type="primary"):
    if not api_key:
        st.error("Masukkan Anthropic API Key di sidebar terlebih dahulu.")
        st.stop()
    if not symbols:
        st.error("Pilih minimal satu saham.")
        st.stop()
    if use_window and start and end and start > end:
        st.error("Rentang tanggal tidak valid: tanggal 'Mulai' melebihi tanggal 'Akhir'.")
        st.stop()

    with st.spinner("Mengambil data..."):
        symbol_data, errors = load_data(tuple(symbols), period)
    if errors:
        with st.expander(f"Error data ({len(errors)})"):
            st.write("\n".join(errors))
    if not symbol_data:
        st.error("Tidak ada data yang berhasil diambil.")
        st.stop()

    try:
        dates = common_dates(symbol_data, cfg)
        n_days = len(dates)
    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal menentukan tanggal simulasi: {e}")
        st.stop()

    expected = n_days // every_n
    if expected > max_calls:
        st.warning(f"Perkiraan {expected} keputusan, tapi batas panggilan {int(max_calls)}. "
                   "Setelah batas tercapai Claude berhenti memutuskan.")

    bar = st.progress(0.0, text="Simulasi berjalan...")
    agent = ClaudeAgent(api_key=api_key, model=model)
    try:
        res = run_backtest(
            symbol_data, agent, cfg,
            progress_cb=lambda i, n, c: bar.progress(
                min(1.0, max(0.0, i / n)),
                text=f"Hari {i}/{n} | panggilan Claude: {c}"
            )
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Simulasi gagal: {e}")
        st.stop()
    bar.empty()

    bench = {}
    if res.equity is not None and not res.equity.empty:
        bench["Claude"] = res.equity[["date", "equity"]]
    try:
        bench["Buy & hold universe"] = buy_and_hold(symbol_data, cfg)
    except Exception:  # noqa: BLE001
        pass
    try:
        bench["IHSG buy & hold"] = index_buy_and_hold(load_index(period), common_dates(symbol_data, cfg), cfg)
    except Exception:  # noqa: BLE001
        pass
    rand = random_baseline(symbol_data, cfg, n_random) if n_random else []

    m = calculate_metrics(res.trades, res.equity, capital)
    run_name = f"Run {len(st.session_state.runs) + 1}"
    saved_dir = save_run(res, cfg, agent=agent, metrics=m, name=run_name, base_dir="runs")
    st.success(f"Hasil simulasi berhasil disimpan ke disk: `{saved_dir}`")

    st.session_state.runs.append({
        "name": run_name, "res": res, "agent": agent,
        "metrics": m, "bench": bench, "rand": rand, "cfg": cfg,
        "saved_dir": saved_dir,
    })

# ------------------------------------------------------------------ results
if st.session_state.runs:
    if len(st.session_state.runs) > 1:
        run_names = [r["name"] for r in st.session_state.runs]
        selected_run_name = st.selectbox("Pilih hasil run untuk ditampilkan:", run_names, index=len(run_names) - 1)
        run = next(r for r in st.session_state.runs if r["name"] == selected_run_name)
    else:
        run = st.session_state.runs[-1]

    res, m = run["res"], run["metrics"]
    st.header(f"Hasil: {run['name']}")
    if res.budget_exhausted_on:
        st.warning(f"Batas panggilan Claude tercapai pada {res.budget_exhausted_on}; sisa hari tanpa keputusan baru.")

    cols = st.columns(7)
    cols[0].metric("Panggilan Claude", res.ai_calls)
    cols[1].metric("Trades", m["trades"])
    cols[2].metric("Win rate", f"{m['win_rate']:.1f}%")
    cols[3].metric("Profit factor", "∞" if np.isinf(m["profit_factor"]) else f"{m['profit_factor']:.2f}")
    cols[4].metric("Return", f"{m['return_pct']:.2f}%")
    cols[5].metric("Max DD", f"{m['max_drawdown_pct']:.2f}%")
    cols[6].metric("Sharpe", f"{m['sharpe']:.2f}")

    st.subheader("Equity vs benchmark (dinormalisasi ke 100)")
    fig = go.Figure()
    has_benchmark_trace = False
    for name, df in run["bench"].items():
        if df is not None and not df.empty and len(df) > 0 and df["equity"].iloc[0] != 0:
            fig.add_trace(go.Scatter(x=df["date"], y=df["equity"] / df["equity"].iloc[0] * 100, mode="lines", name=name))
            has_benchmark_trace = True
    if has_benchmark_trace:
        fig.update_layout(height=430, yaxis_title="Indeks (awal = 100)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kurva equity belum tersedia.")

    if run["rand"]:
        r = np.array(run["rand"])
        st.info(f"Agent acak (n={len(r)}, aturan sama): median {np.median(r):.1f}%, "
                f"rentang {r.min():.1f}% sampai {r.max():.1f}%. Claude: {m['return_pct']:.1f}%.")

    u = res.usage
    if u:
        st.caption(f"Token: input {u.get('input_tokens', 0):,} | output {u.get('output_tokens', 0):,} | "
                   f"cache read {u.get('cache_read_tokens', 0):,} | cache write {u.get('cache_write_tokens', 0):,}")

    st.subheader("Trade journal")
    if res.trades:
        tdf = pd.DataFrame([asdict(t) for t in res.trades])
        st.dataframe(tdf, use_container_width=True)
        st.dataframe(tdf["exit_reason"].value_counts().rename_axis("exit_reason").reset_index(name="count"))
        st.download_button("Unduh trade journal (CSV)", tdf.to_csv(index=False), "trades.csv", mime="text/csv")
    else:
        st.warning("Tidak ada trade.")

    st.subheader("Log keputusan Claude")
    if res.log:
        st.dataframe(pd.DataFrame([{"date": x.get("date"),
                                    "entry": json.dumps({k: v for k, v in x.items() if k != "date"}, ensure_ascii=False)}
                                   for x in res.log]), use_container_width=True)
    else:
        st.info("Log keputusan kosong.")

    st.subheader("Memori & catatan belajar")
    if run.get("agent") and hasattr(run["agent"], "memory_summary"):
        st.json(run["agent"].memory_summary())
    elif run.get("agent_memory"):
        st.json(run["agent_memory"])
    else:
        st.info("Catatan memori tidak tersedia.")

    if len(st.session_state.runs) > 1:
        st.subheader("Perbandingan semua run (variasi antar-run LLM)")
        rows_comp = []
        for r in st.session_state.runs:
            row_dict = {"run": r["name"]}
            for k, v in r["metrics"].items():
                if np.isinf(v):
                    row_dict[k] = "∞"
                elif isinstance(v, (int, np.integer)):
                    row_dict[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    row_dict[k] = round(float(v), 2)
                else:
                    row_dict[k] = v
            rows_comp.append(row_dict)
        st.dataframe(pd.DataFrame(rows_comp), use_container_width=True)
    st.caption("Satu run belum cukup: jalankan beberapa kali dan lihat rentangnya.")
