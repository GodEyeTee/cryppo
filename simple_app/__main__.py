from __future__ import annotations

import os
import sys
import subprocess
from typing import List

import pandas as pd

from .config import AppConfig, parse_date
from .binance import download_klines
from .process import process_df
from .io import ensure_dir, save_df, load_df
from .analyze import compute_stats, daily_ohlcv, plot_price


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def prompt_default(prompt: str, default: str) -> str:
    out = _safe_input(f"{prompt} [{default}]: ").strip()
    return out or default


def run_download(cfg: AppConfig) -> None:
    print("\n== Download ==")
    symbol = prompt_default("Symbol", cfg.symbol)
    tfs = prompt_default("Timeframes (comma)", ",".join(cfg.timeframes)).split(",")
    start = prompt_default("Start (YYYY-MM-DD)", cfg.start)
    end = prompt_default("End   (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)", cfg.end)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    out_root = cfg.raw_dir()
    ensure_dir(out_root)
    for tf in [tf.strip() for tf in tfs if tf.strip()]:
        print(f"Downloading {symbol} {tf}...")
        df = download_klines(symbol, tf, start_dt, end_dt)
        if df.empty:
            print(f"No data for {symbol} {tf}")
            continue
        out_dir = os.path.join(out_root, symbol, tf)
        ensure_dir(out_dir)
        out_file = os.path.join(
            out_dir,
            f"{symbol.lower()}_{tf}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet",
        )
        path = save_df(df, out_file)
        print(f"Saved: {path} ({len(df)} rows)")


def run_process(cfg: AppConfig) -> None:
    print("\n== Process ==")
    in_path = prompt_default("Input file (.csv/.parquet)", "")
    while not in_path or not os.path.exists(in_path):
        in_path = input("  File not found. Enter path again: ").strip()

    df = load_df(in_path)
    out = process_df(df)

    out_dir = cfg.processed_dir()
    ensure_dir(out_dir)
    base = os.path.basename(in_path)
    out_path = os.path.join(out_dir, base)
    out_path = save_df(out, out_path)
    print(f"Saved processed: {out_path} ({len(out)} rows, {len(out.columns)} cols)")


def run_analyze(cfg: AppConfig) -> None:
    print("\n== Analyze ==")
    in_path = prompt_default("Input file (.parquet/.csv)", "")
    while not in_path or not os.path.exists(in_path):
        in_path = input("  File not found. Enter path again: ").strip()

    df = load_df(in_path)
    info = compute_stats(df)
    print("\nStats:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    want_daily = prompt_default("Show daily OHLCV summary? (y/n)", "y").lower() == "y"
    if want_daily:
        try:
            d = daily_ohlcv(df)
            print(d.head(10))
        except Exception as e:
            print(f"  Daily summary error: {e}")

    want_plot = prompt_default("Plot price? (y/n)", "n").lower() == "y"
    if want_plot:
        try:
            plot_price(df)
        except Exception as e:
            print(f"  Plot error: {e}")


def run_all(cfg: AppConfig) -> None:
    print("\n== Run All (Download -> Process -> Analyze) ==")
    # Download default first timeframe only for speed; user can extend later
    symbol = cfg.symbol
    tf = cfg.timeframes[0]
    start_dt = cfg.start_dt()
    end_dt = cfg.end_dt()
    print(f"Downloading {symbol} {tf} {cfg.start} -> {cfg.end}")
    df = download_klines(symbol, tf, start_dt, end_dt)
    if df.empty:
        print("No data downloaded. Aborting.")
        return
    raw_dir = os.path.join(cfg.raw_dir(), symbol, tf)
    ensure_dir(raw_dir)
    raw_file = os.path.join(
        raw_dir,
        f"{symbol.lower()}_{tf}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet",
    )
    raw_path = save_df(df, raw_file)
    print(f"Saved raw: {raw_path}")

    proc = process_df(df)
    proc_dir = cfg.processed_dir()
    ensure_dir(proc_dir)
    proc_file = os.path.join(
        proc_dir,
        f"{symbol.lower()}_{tf}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet",
    )
    proc_path = save_df(proc, proc_file)
    print(f"Saved processed: {proc_path}")

    info = compute_stats(proc)
    print("\nProcessed stats:")
    for k, v in info.items():
        print(f"  {k}: {v}")


def run_train(cfg: AppConfig) -> None:
    print("\n== Train ==")
    # Suggest last processed from simple_app default
    default_proc = os.path.join(
        cfg.processed_dir(),
        f"{cfg.symbol.lower()}_{cfg.timeframes[0]}_{cfg.start_dt().strftime('%Y%m%d')}_{cfg.end_dt().strftime('%Y%m%d')}.parquet",
    )
    in_path = prompt_default("Processed input (.parquet)", default_proc if os.path.exists(default_proc) else "")
    while not in_path or not os.path.exists(in_path):
        in_path = _safe_input("  File not found. Enter path again: ").strip()

    out_dir = prompt_default("Output models dir", "outputs/models")
    model_type = prompt_default("Model type (dqn/double_dqn/dueling_dqn)", "double_dqn")
    window_size = prompt_default("Window size", "60")
    batch_size = prompt_default("Batch size", "64")
    epochs = prompt_default("Epochs (timesteps)", "1000")
    use_gpu = prompt_default("Use GPU? (y/n)", "n").lower() == "y"

    cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "train",
        "model",
        "--input",
        in_path,
        "--output",
        out_dir,
        "--model-type",
        model_type,
        "--window-size",
        window_size,
        "--batch-size",
        batch_size,
        "--epochs",
        epochs,
    ]
    cmd.append("--use-gpu" if use_gpu else "--no-gpu")

    print("\nRunning:")
    print(" ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")


def main() -> None:
    cfg = AppConfig()
    print("Simple CRYPPO — One-Button Menu")
    print("================================")
    print("1) Run All (default)")
    print("2) Download")
    print("3) Process")
    print("4) Analyze")
    print("5) Train")
    print("6) Exit")

    raw = _safe_input("Select [1]: ")
    choice = (raw.strip() if raw is not None else "") or "1"
    if choice == "1":
        run_all(cfg)
    elif choice == "2":
        run_download(cfg)
    elif choice == "3":
        run_process(cfg)
    elif choice == "4":
        run_analyze(cfg)
    elif choice == "5":
        run_train(cfg)
    else:
        print("Bye.")


if __name__ == "__main__":
    main()
