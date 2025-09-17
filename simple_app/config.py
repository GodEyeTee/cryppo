from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


def parse_date(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid date: {value}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"
    )


@dataclass
class AppConfig:
    symbol: str = "BTCUSDT"
    timeframes: List[str] = field(default_factory=lambda: ["1m", "1h", "1d"])
    start: str = "2023-01-01"
    end: str = "2024-01-01"  # inclusive end-of-day for 2023
    data_dir: str = "simple_data"
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"
    file_format: str = "parquet"

    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, self.raw_subdir)

    def processed_dir(self) -> str:
        return os.path.join(self.data_dir, self.processed_subdir)

    def start_dt(self) -> datetime:
        return parse_date(self.start)

    def end_dt(self) -> datetime:
        return parse_date(self.end)

