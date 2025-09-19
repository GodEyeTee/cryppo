from __future__ import annotations

import os
import sys
import json
import time
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

USE_GPU_AVAILABLE = bool(torch and torch.cuda.is_available())

from .config import AppConfig
from .binance import download_klines
from .process import process_df
from .io import ensure_dir, save_df
from .analyze import compute_stats, daily_ohlcv

# Import CLI handlers programmatically
from src.cli.commands import train_commands as train_cli
from src.cli.commands import backtest_commands as backtest_cli


def newest_dir(root: str, prefix: str) -> str | None:
    root_p = Path(root)
    if not root_p.exists():
        return None
    cands = [p for p in root_p.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def run_pipeline() -> int:
    cfg = AppConfig()

    # Output roots (clean, self-contained)
    out_root = Path("oneclick")
    raw_root = out_root / "raw"
    proc_root = out_root / "processed"
    report_root = out_root / "reports"
    model_root = out_root / "models"
    backtest_root = out_root / "backtest"
    for p in (raw_root, proc_root, report_root, model_root, backtest_root):
        ensure_dir(str(p))

    symbol = cfg.symbol
    timeframe = cfg.timeframes[0]
    start = cfg.start_dt()
    end = cfg.end_dt()

    # 1) Download
    print(f"[1/8] Download {symbol} {timeframe} {cfg.start} -> {cfg.end}")
    try:
        df_raw = download_klines(symbol, timeframe, start, end)
        if df_raw.empty:
            print("  No data downloaded.")
            return 2
        raw_dir = raw_root / symbol / timeframe
        ensure_dir(str(raw_dir))
        raw_file = raw_dir / f"{symbol.lower()}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        save_df(df_raw, str(raw_file))
        print(f"  Saved: {raw_file} ({len(df_raw)} rows)")
    except Exception as e:
        print(f"  Download error: {e}")
        return 2

    # 2) Process
    print("[2/8] Process data + indicators")
    try:
        df_proc = process_df(df_raw)
        proc_file = proc_root / raw_file.name
        save_df(df_proc, str(proc_file))
        print(f"  Saved: {proc_file} ({len(df_proc)} rows, {len(df_proc.columns)} cols)")
    except Exception as e:
        print(f"  Process error: {e}")
        return 3

    # 3) Analyze -> reports
    print("[3/8] Analyze + daily summary")
    try:
        stats = compute_stats(df_proc)
        (report_root / "data").mkdir(parents=True, exist_ok=True)
        with open(report_root / "data" / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        daily = daily_ohlcv(df_raw)
        daily.to_csv(report_root / "data" / "daily_ohlcv.csv", index=False)
        print("  Saved reports: stats.json, daily_ohlcv.csv")
    except Exception as e:
        print(f"  Analyze error: {e}")
        return 4

    # 4) Train
    print("[4/8] Train model (double_dqn)")
    try:
        args = SimpleNamespace(
            input=str(proc_file),
            output=str(model_root),
            model_type="double_dqn",
            window_size=60,
            batch_size=64,
            epochs=100000,  # timesteps for environment training
            learning_rate=None,
            discount_factor=None,
            target_update=None,
            validation_ratio=0.1,
            test_ratio=0.1,
            use_gpu=USE_GPU_AVAILABLE,
            tensorboard=False,
            seed=42,
            config=None,
        )
        device_msg = 'GPU' if USE_GPU_AVAILABLE else 'CPU'
        print(f'  Training device: {device_msg}')
        train_cli.handle_model(args)
        # Find newest model dir
        model_dir = newest_dir(str(model_root), "double_dqn_")
        if not model_dir:
            print("  Could not locate created model directory.")
            return 5
        model_path = str(Path(model_dir) / "model.pt")
        print(f"  Model: {model_path}")
    except Exception as e:
        print(f"  Train error: {e}")
        return 5

    # 5) Test/Evaluate
    print("[5/8] Evaluate model on processed data")
    try:
        test_out = report_root / "test_metrics.json"
        args_eval = SimpleNamespace(
            model=model_path,
            input=str(proc_file),
            output=str(test_out),
            batch_size=None,
            window_size=None,
            metrics="all",
            plot=False,
            use_gpu=USE_GPU_AVAILABLE,
        )
        train_cli.handle_evaluate(args_eval)
        print(f"  Saved: {test_out}")
    except Exception as e:
        print(f"  Evaluate error: {e}")
        return 6

    # 6) Backtest run
    print("[6/8] Backtest run")
    try:
        args_bt = SimpleNamespace(
            model=model_path,
            input=str(proc_file),
            output=str(backtest_root),
            start_date=None,
            end_date=None,
            initial_balance=10_000.0,
            leverage=3.0,
            fee_rate=0.0025,
            stop_loss=None,
            take_profit=None,
            window_size=None,
            batch_size=None,
            use_gpu=USE_GPU_AVAILABLE,
            plot=False,
            verbose=False,
            config=None,
        )
        backtest_cli.handle_run(args_bt)
        # newest backtest dir
        bt_dir = newest_dir(str(backtest_root), "backtest_")
        if not bt_dir:
            print("  Could not locate backtest results.")
            return 7
        print(f"  Backtest folder: {bt_dir}")
    except Exception as e:
        print(f"  Backtest error: {e}")
        return 7

    # 7) Backtest analyze (metrics report)
    print("[7/8] Backtest analyze -> report")
    try:
        bt_report = report_root / "backtest_metrics.json"
        args_bta = SimpleNamespace(
            input=str(bt_dir),
            output=str(bt_report),
            metrics="all",
            benchmark=None,
            plot=None,
            period="daily",
        )
        backtest_cli.handle_analyze(args_bta)
        print(f"  Saved: {bt_report}")
    except Exception as e:
        print(f"  Backtest analyze error: {e}")
        return 8

    print("[8/8] Done.")
    print(f"Output root: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
