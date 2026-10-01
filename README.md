# 🤖 Autonomous Claude IDX Trading Lab

Simulator trading saham IDX di mana **Claude** memutuskan semuanya (pilih saham, BUY/EXIT,
stop loss, take profit, ukuran posisi, dan belajar dari trade sebelumnya), sedangkan **Python**
hanya menyediakan data, mengeksekusi order virtual, menjaga aturan risiko, dan menghitung performa.

> **Tanpa broker. Tanpa uang nyata. Bukan saran keuangan.** Ini alat riset metodologi.

## Instalasi

```bash
git clone <url-repo-anda>
cd idx-claude-trader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # isi ANTHROPIC_API_KEY, lalu: export $(cat .env | xargs)
streamlit run app.py
```

Mode headless: `python cli.py --start 2026-07-01 --calls 80`

Tes: `pip install -r requirements-dev.txt && pytest` (atau `python run_tests.py`).

## Arsitektur

| Modul | Tugas |
|---|---|
| `idxagent/data.py` | Download yfinance, indikator kausal, snapshot skala-bebas |
| `idxagent/agent.py` | `ClaudeAgent` (tool use, prompt caching, memori) dan `RandomAgent` (baseline) |
| `idxagent/execution.py` | Tick size IDX, fill, SL/TP, sizing berbasis risiko, biaya |
| `idxagent/backtest.py` | Loop harian, anonimisasi, validasi order |
| `idxagent/benchmarks.py` | Buy & hold universe, IHSG, agent acak |
| `idxagent/metrics.py` | Return, drawdown, profit factor, Sharpe, exposure |

## Perbaikan dibanding prototipe awal

**Validitas**
- Keputusan di *close* hari t, order terisi di **open hari t+1**. Claude tidak bisa memilih harga isi.
- Gap di bawah SL terisi di **open**, bukan di harga SL. Jika SL dan TP tersentuh satu candle, SL menang.
- Harga dibulatkan ke **tick size BEI**.
- **Anonimisasi**: simbol menjadi `STOCK_xx`, tanggal menjadi `D0, D1, ...`, dan snapshot hanya berisi fitur
  skala-bebas (persen, rasio), tanpa harga absolut. Ini mengurangi kebocoran memori model.
  SL/TP dinyatakan dalam persen dari harga fill.
- **Jendela uji** (`sim_start`/`sim_end`) untuk menguji periode setelah cutoff pengetahuan model.
- **Benchmark** bawaan: buy & hold universe, IHSG, dan N agent acak dengan aturan eksekusi sama.
- Filter likuiditas dilakukan Python, bukan Claude.

**Bug yang diperbaiki**
- `dropna()` karena MA200 membuat periode 6mo/1y gagal. Kini MA200 opsional.
- Batas panggilan API tidak lagi memotong diam-diam (ada peringatan dan `budget_exhausted_on`).
- Satu keputusan malformed tidak lagi mematikan simulasi; ditolak dan dicatat di log.
- `learning_note` dan `market_view` kini disimpan dan dikirim balik sebagai memori.
- Output via **tool use** (JSON valid), **prompt caching**, token usage dicatat.
- Hasil disimpan di `st.session_state`, ada progress bar, dan riwayat antar-run.

## Cara memakai dengan jujur

1. Jalankan dengan **anonimisasi ON**. Bandingkan dengan OFF; selisih besar = tanda kebocoran memori.
2. Uji pada jendela **setelah cutoff model** (data terbaru, jumlah trade sedikit).
3. Jalankan **beberapa kali** dan lihat rentang hasil (LLM tidak deterministik).
4. Claude harus mengalahkan **agent acak dan buy & hold** setelah biaya. Jika tidak, belum ada edge.
5. Lanjut ke **paper trading forward-test** berminggu-minggu sebelum menyentuh uang nyata.

## Keterbatasan yang diketahui

- Belum memodelkan batas ARA/ARB, suspensi, corporate action, dan antrean order.
- Data yfinance tidak sempurna dan universe default adalah blue chip hari ini (survivorship bias).
- Isi di open + slippage tetap-persen adalah aproksimasi likuiditas.
- Jendela setelah cutoff pendek, jadi secara statistik lemah.

## Roadmap

Walk-forward (fase belajar vs uji), universe lebih luas dengan delisting, ARA/ARB, penyimpanan hasil ke disk,
paper trading real-time.

## Lisensi

MIT. Gunakan dengan risiko sendiri.
