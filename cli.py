"""Headless runner: python cli.py --start 2026-07-01 --calls 80"""

import argparse
import os
import sys

from idxagent.agent import ClaudeAgent
from idxagent.backtest import run_backtest
from idxagent.config import DEFAULT_MODEL, DEFAULT_UNIVERSE, SimConfig
from idxagent.data import prepare_market_data
from idxagent.metrics import calculate_metrics
from idxagent.storage import compare_saved_runs, list_runs, save_run


def load_dotenv(path: str = ".env") -> None:
    """Memuat variabel dari file .env sederhana jika belum ada di os.environ."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Headless runner Autonomous Claude IDX Trading Lab")
    p.add_argument("--period", default="2y", help="Periode yfinance (default: 2y)")
    p.add_argument("--start", help="Tanggal mulai simulasi (YYYY-MM-DD)")
    p.add_argument("--end", help="Tanggal akhir simulasi (YYYY-MM-DD)")
    p.add_argument("--every", type=int, default=1, help="Keputusan Claude setiap N hari bursa (default: 1)")
    p.add_argument("--calls", type=int, default=250, help="Maks. panggilan Claude (default: 250)")
    p.add_argument("--model", default=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL), help=f"Model Claude (default: {DEFAULT_MODEL})")
    p.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"), help="Anthropic API Key (atau via env ANTHROPIC_API_KEY)")
    p.add_argument("--no-anonymize", action="store_true", help="Nonaktifkan anonimisasi simbol/tanggal")
    p.add_argument("--name", default="", help="Nama label untuk run ini")
    p.add_argument("--output-dir", default="runs", help="Direktori penyimpanan run (default: runs)")
    p.add_argument("--no-save", action="store_true", help="Jangan simpan hasil ke disk")
    p.add_argument("--list-runs", action="store_true", help="Tampilkan daftar run yang tersimpan di disk dan keluar")
    p.add_argument("--compare-runs", action="store_true", help="Tampilkan tabel perbandingan metrik run di disk dan keluar")
    a = p.parse_args()

    if a.list_runs:
        runs = list_runs(a.output_dir)
        if not runs:
            print(f"Tidak ada run yang tersimpan di '{a.output_dir}'.")
        else:
            print(f"\n--- DAFTAR RUN TERSIMPAN ({len(runs)}) ---")
            for r in runs:
                print(f"[{r['timestamp'][:19] if r.get('timestamp') else '-'}] {r['name']:<25} | "
                      f"Trades: {r['trades']:>3} | WinRate: {r['win_rate']:>5.1f}% | "
                      f"Return: {r['return_pct']:>6.2f}% | MaxDD: {r['max_drawdown_pct']:>6.2f}% | "
                      f"Dir: {r['path']}")
        return

    if a.compare_runs:
        df = compare_saved_runs(a.output_dir)
        if df.empty:
            print(f"Tidak ada data run untuk dibandingkan di '{a.output_dir}'.")
        else:
            print("\n--- PERBANDINGAN METRIK SELURUH RUN ---")
            print(df.to_string(index=False))
        return

    api_key = (a.api_key or "").strip()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY tidak ditemukan.")
        print("Silakan setel variabel lingkungan ANTHROPIC_API_KEY, buat file .env, atau berikan argumen --api-key.")
        sys.exit(1)

    cfg = SimConfig(
        sim_start=a.start,
        sim_end=a.end,
        decision_every_n_days=a.every,
        max_ai_calls=a.calls,
        anonymize=not a.no_anonymize
    )

    print(f"Mengambil data pasar ({a.period}) untuk {len(DEFAULT_UNIVERSE)} simbol...")
    data, errors = prepare_market_data(DEFAULT_UNIVERSE, a.period)
    for e in errors:
        print("data error:", e)
    if not data:
        print("Error: Tidak ada data pasar yang berhasil diambil.")
        sys.exit(1)

    try:
        agent = ClaudeAgent(api_key, a.model.strip())
        print(f"Memulai backtest: model={a.model}, window={a.start or 'all'} s/d {a.end or 'all'}, max_calls={a.calls}...")
        res = run_backtest(
            data, agent, cfg,
            progress_cb=lambda i, n, c: print(f"\rKemajuan: {i}/{n} hari | Panggilan Claude: {c}", end="", flush=True)
        )
        print()
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh pengguna (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        print(f"\nSimulasi gagal: {e}")
        sys.exit(1)

    metrics = calculate_metrics(res.trades, res.equity, cfg.initial_capital)
    print("\n--- METRIK PERFORMA ---")
    for k, v in metrics.items():
        print(f"{k:>18}: {v:,.2f}" if isinstance(v, float) else f"{k:>18}: {v}")
    print("usage:", res.usage)
    if res.budget_exhausted_on:
        print(f"Peringatan: Kuota panggilan Claude habis pada {res.budget_exhausted_on}")

    if not a.no_save:
        run_name = a.name or f"Claude_{a.model.split('-')[0]}"
        saved_dir = save_run(res, cfg, agent=agent, metrics=metrics, name=run_name, base_dir=a.output_dir)
        print(f"\n[Storage] Hasil run berhasil disimpan ke: {saved_dir}")


if __name__ == "__main__":
    main()
