from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class WaveSequence:
    indices: List[int]
    prices: List[float]
    direction: int
    ratios: Dict[str, float]
    score: float


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist
    })


def stochastic(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    lowest_low = series_low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = series_high.rolling(window=k_period, min_periods=k_period).max()
    percent_k = 100 * (series_close - lowest_low) / (highest_high - lowest_low + 1e-12)
    if slowing > 1:
        percent_k = percent_k.rolling(window=slowing, min_periods=slowing).mean()
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({
        f"stoch_k_{k_period}": percent_k,
        f"stoch_d_{k_period}_{d_period}": percent_d
    })


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi_14"] = rsi(out["close"], 14)
    out = out.join(macd(out["close"]))
    out = out.join(stochastic(out["high"], out["low"], out["close"], 14, 3, 3))
    return out


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    body = result["close"] - result["open"]
    candle_range = (result["high"] - result["low"]).replace(0, np.nan)
    upper_wick = result["high"] - result[["open", "close"]].max(axis=1)
    lower_wick = result[["open", "close"]].min(axis=1) - result["low"]

    result["pa_body_to_range"] = body / (candle_range + 1e-12)
    result["pa_upper_wick_ratio"] = upper_wick / (candle_range + 1e-12)
    result["pa_lower_wick_ratio"] = lower_wick / (candle_range + 1e-12)
    result["pa_is_bull"] = (result["close"] > result["open"]).astype(int)
    result["pa_range"] = candle_range
    result["pa_body_abs"] = body.abs()

    prev_open = result["open"].shift(1)
    prev_close = result["close"].shift(1)
    result["pa_bull_engulf"] = ((result["close"] > result["open"]) & (prev_close < prev_open) & (result["close"] > prev_open) & (result["open"] < prev_close)).astype(int)
    result["pa_bear_engulf"] = ((result["close"] < result["open"]) & (prev_close > prev_open) & (result["close"] < prev_open) & (result["open"] > prev_close)).astype(int)

    result["pa_inside_bar"] = ((result["high"] <= result["high"].shift(1)) & (result["low"] >= result["low"].shift(1))).astype(int)
    result["pa_outside_bar"] = ((result["high"] >= result["high"].shift(1)) & (result["low"] <= result["low"].shift(1))).astype(int)

    return result


def add_fibonacci_levels(df: pd.DataFrame, period: int = 120) -> pd.DataFrame:
    fib_cols = ["fib_0", "fib_236", "fib_382", "fib_500", "fib_618", "fib_786", "fib_1000"]
    data = {col: np.full(len(df), np.nan) for col in fib_cols}

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for idx in range(period, len(df)):
        window_high = highs[idx - period:idx].max()
        window_low = lows[idx - period:idx].min()
        distance = window_high - window_low
        if distance <= 0:
            continue
        data["fib_0"][idx] = window_high
        data["fib_1000"][idx] = window_low
        data["fib_236"][idx] = window_high - 0.236 * distance
        data["fib_382"][idx] = window_high - 0.382 * distance
        data["fib_500"][idx] = window_high - 0.5 * distance
        data["fib_618"][idx] = window_high - 0.618 * distance
        data["fib_786"][idx] = window_high - 0.786 * distance

    fib_df = pd.DataFrame(data, index=df.index)
    return df.join(fib_df)


def _identify_swings(close: np.ndarray, window: int = 6) -> List[Tuple[int, float, str]]:
    pivots: List[Tuple[int, float, str]] = []
    length = len(close)
    if length < window * 2 + 1:
        return pivots

    for idx in range(window, length - window):
        section = close[idx - window: idx + window + 1]
        if close[idx] == section.max():
            pivots.append((idx, close[idx], 'H'))
        elif close[idx] == section.min():
            pivots.append((idx, close[idx], 'L'))

    if not pivots:
        return pivots

    filtered: List[Tuple[int, float, str]] = []
    for idx, price, kind in sorted(pivots, key=lambda x: x[0]):
        if not filtered:
            filtered.append((idx, price, kind))
            continue
        last_idx, last_price, last_kind = filtered[-1]
        if last_kind == kind:
            if (kind == 'H' and price >= last_price) or (kind == 'L' and price <= last_price):
                filtered[-1] = (idx, price, kind)
            continue
        filtered.append((idx, price, kind))

    return filtered


def _validate_wave(segment: List[Tuple[int, float, str]]) -> WaveSequence | None:
    kinds = [k for _, _, k in segment]
    prices = [p for _, p, _ in segment]
    indices = [i for i, _, _ in segment]

    if kinds == ['L', 'H', 'L', 'H', 'L', 'H']:
        direction = 1
    elif kinds == ['H', 'L', 'H', 'L', 'H', 'L']:
        direction = -1
    else:
        return None

    p0, p1, p2, p3, p4, p5 = prices
    eps = 1e-9

    if direction == 1:
        if not (p1 > p0 and p3 > p1 and p5 > p3):
            return None
        if p2 <= p0 + eps or p4 <= p1 - eps:
            return None
    else:
        if not (p1 < p0 and p3 < p1 and p5 < p3):
            return None
        if p2 >= p0 - eps or p4 >= p1 + eps:
            return None

    wave1 = abs(p1 - p0)
    wave2 = abs(p2 - p1)
    wave3 = abs(p3 - p2)
    wave4 = abs(p4 - p3)
    wave5 = abs(p5 - p4)

    if wave1 < eps or wave3 < eps or wave5 < eps:
        return None

    if direction == 1:
        if p3 <= p1 or p5 <= p3:
            return None
    else:
        if p3 >= p1 or p5 >= p3:
            return None

    # Elliott heuristics
    if wave3 <= wave1 * 0.5:
        return None
    if wave5 < wave1 * 0.3:
        return None
    if max(wave2, wave4) >= wave3:
        return None

    ratios: Dict[str, float] = {}
    ratios['wave2_retrace'] = wave2 / (wave1 + eps)
    ratios['wave3_extension'] = wave3 / (wave1 + eps)
    ratios['wave4_retrace'] = wave4 / (wave3 + eps)
    ratios['wave5_extension'] = wave5 / (wave3 + eps)

    targets = {
        'wave2_retrace': 0.618,
        'wave3_extension': 1.618,
        'wave4_retrace': 0.382,
        'wave5_extension': 0.618,
    }
    score = 0.0
    count = 0
    for key, target in targets.items():
        value = ratios.get(key)
        if value is None or not np.isfinite(value):
            continue
        diff = abs(value - target)
        normalised = max(0.0, 1.0 - min(diff / (target + eps), 1.0))
        score += normalised
        count += 1
    score = score / count if count else 0.0

    return WaveSequence(indices=indices, prices=prices, direction=direction, ratios=ratios, score=score)


def _elliott_wave_features(df: pd.DataFrame, swing_window: int = 6) -> pd.DataFrame:
    close = df['close'].to_numpy()
    pivots = _identify_swings(close, swing_window)
    if len(pivots) < 6:
        return pd.DataFrame(index=df.index)

    sequences: List[WaveSequence] = []
    for idx in range(5, len(pivots)):
        segment = pivots[idx - 5: idx + 1]
        wave = _validate_wave(segment)
        if wave is not None:
            sequences.append(wave)

    if not sequences:
        return pd.DataFrame(index=df.index)

    n = len(df)
    wave_label = np.zeros(n, dtype=float)
    wave_direction = np.zeros(n, dtype=float)
    wave_progress = np.full(n, np.nan)
    wave_score = np.full(n, np.nan)
    wave2_retrace = np.full(n, np.nan)
    wave3_ext = np.full(n, np.nan)
    wave4_retrace = np.full(n, np.nan)
    wave5_ext = np.full(n, np.nan)

    for seq in sequences:
        for wave_idx in range(1, 6):
            start_idx = seq.indices[wave_idx - 1]
            end_idx = seq.indices[wave_idx]
            if end_idx <= start_idx:
                continue
            start_price = seq.prices[wave_idx - 1]
            end_price = seq.prices[wave_idx]
            delta = end_price - start_price
            if seq.direction == -1:
                delta = start_price - end_price
            if abs(delta) < 1e-9:
                delta = 1e-9

            wave_label[start_idx:end_idx + 1] = wave_idx * seq.direction
            wave_direction[start_idx:end_idx + 1] = seq.direction
            wave_score[start_idx:end_idx + 1] = seq.score

            for pos in range(start_idx, end_idx + 1):
                price = close[pos]
                if seq.direction == 1:
                    progress = (price - start_price) / (end_price - start_price + 1e-9)
                else:
                    progress = (start_price - price) / (start_price - end_price + 1e-9)
                wave_progress[pos] = np.clip(progress, 0.0, 1.0)

            wave2_retrace[start_idx:end_idx + 1] = seq.ratios.get('wave2_retrace')
            wave3_ext[start_idx:end_idx + 1] = seq.ratios.get('wave3_extension')
            wave4_retrace[start_idx:end_idx + 1] = seq.ratios.get('wave4_retrace')
            wave5_ext[start_idx:end_idx + 1] = seq.ratios.get('wave5_extension')

    return pd.DataFrame({
        'ew_wave_label': wave_label,
        'ew_wave_direction': wave_direction,
        'ew_wave_progress': wave_progress,
        'ew_wave_quality': wave_score,
        'ew_wave2_retrace': wave2_retrace,
        'ew_wave3_extension': wave3_ext,
        'ew_wave4_retrace': wave4_retrace,
        'ew_wave5_extension': wave5_ext,
    }, index=df.index)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)

    out = df.copy()
    out['return'] = np.log(out['close'].clip(lower=1e-12)).diff()

    out = add_momentum_indicators(out)
    out = add_price_action_features(out)
    out = add_fibonacci_levels(out, period=120)
    elliott_df = _elliott_wave_features(out)
    out = out.join(elliott_df)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(method='ffill', inplace=True)
    out.fillna(method='bfill', inplace=True)

    return out
