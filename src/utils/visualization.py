import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import logging

logger = logging.getLogger('utils.visualization')

# กำหนดสไตล์การพล็อต
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

def plot_equity_curve(
    equity_curve: Union[List[float], np.ndarray, pd.Series],
    title: str = 'Equity Curve',
    figsize: Tuple[int, int] = (12, 6),
    show_drawdown: bool = True,
    timestamps: Optional[List[Any]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้างกราฟเส้นทุน (Equity Curve)

    Parameters:
    equity_curve (List[float], np.ndarray, pd.Series): ข้อมูลเส้นทุน
    title (str): ชื่อกราฟ
    figsize (Tuple[int, int]): ขนาดของกราฟ
    show_drawdown (bool): แสดงจุดที่เกิด drawdown
    timestamps (List[Any], optional): เวลาสำหรับแกน x
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(equity_curve, pd.Series) and timestamps is None:
        # ใช้ index ของ Series เป็น timestamps
        timestamps = equity_curve.index
        equity_data = equity_curve.values
    else:
        equity_data = np.asarray(equity_curve)
    
    # พล็อตเส้นทุน
    if timestamps is not None:
        ax.plot(timestamps, equity_data, linewidth=2, label='Equity')
        
        # จัดการรูปแบบแกน x หากเป็นข้อมูลวันที่
        if isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
    else:
        ax.plot(equity_data, linewidth=2, label='Equity')
    
    # แสดง drawdown หากต้องการ
    if show_drawdown and len(equity_data) > 1:
        # คำนวณ running maximum
        running_max = np.maximum.accumulate(equity_data)
        
        # คำนวณ drawdown ในแต่ละจุด
        drawdowns = (running_max - equity_data) / running_max
        
        # หาค่า drawdown สูงสุด
        max_dd = np.max(drawdowns)
        max_dd_idx = np.argmax(drawdowns)
        
        # หาจุดเริ่มต้นของ drawdown
        peak_idx = np.argmax(equity_data[:max_dd_idx+1])
        
        # เพิ่มการแสดง drawdown
        if timestamps is not None:
            ax.fill_between(
                timestamps, 
                equity_data, 
                running_max, 
                color='red', 
                alpha=0.3, 
                label=f'Drawdown (Max: {max_dd:.2%})'
            )
            
            # แสดงจุด peak และ trough
            ax.plot(
                timestamps[peak_idx], 
                equity_data[peak_idx], 
                'go', 
                markersize=8, 
                label='Peak'
            )
            ax.plot(
                timestamps[max_dd_idx], 
                equity_data[max_dd_idx], 
                'ro', 
                markersize=8, 
                label='Trough'
            )
        else:
            ax.fill_between(
                range(len(equity_data)), 
                equity_data, 
                running_max, 
                color='red', 
                alpha=0.3, 
                label=f'Drawdown (Max: {max_dd:.2%})'
            )
            
            # แสดงจุด peak และ trough
            ax.plot(
                peak_idx, 
                equity_data[peak_idx], 
                'go', 
                markersize=8, 
                label='Peak'
            )
            ax.plot(
                max_dd_idx, 
                equity_data[max_dd_idx], 
                'ro', 
                markersize=8, 
                label='Trough'
            )
    
    # กำหนดชื่อแกนและชื่อกราฟ
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time' if timestamps is None else 'Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    
    # เพิ่ม legend
    ax.legend()
    
    # จัดรูปแบบแกน y ให้เป็นสกุลเงิน
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def plot_returns_distribution(
    returns: Union[List[float], np.ndarray, pd.Series],
    title: str = 'Returns Distribution',
    figsize: Tuple[int, int] = (12, 6),
    bins: int = 50,
    kde: bool = True,
    show_stats: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้างกราฟการกระจายของผลตอบแทน

    Parameters:
    returns (List[float], np.ndarray, pd.Series): ข้อมูลผลตอบแทน
    title (str): ชื่อกราฟ
    figsize (Tuple[int, int]): ขนาดของกราฟ
    bins (int): จำนวน bins ของ histogram
    kde (bool): แสดง kernel density estimate
    show_stats (bool): แสดงสถิติ
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    returns_data = np.asarray(returns)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # พล็อต histogram และ kde หากต้องการ
    sns.histplot(returns_data, bins=bins, kde=kde, ax=ax, color='skyblue')
    
    # พล็อตค่าเฉลี่ยและศูนย์
    ax.axvline(np.mean(returns_data), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns_data):.4f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero')
    
    # แสดงสถิติหากต้องการ
    if show_stats:
        stats_text = (
            f"Mean: {np.mean(returns_data):.4f}\n"
            f"Median: {np.median(returns_data):.4f}\n"
            f"Std Dev: {np.std(returns_data):.4f}\n"
            f"Skewness: {pd.Series(returns_data).skew():.4f}\n"
            f"Kurtosis: {pd.Series(returns_data).kurtosis():.4f}\n"
            f"Min: {np.min(returns_data):.4f}\n"
            f"Max: {np.max(returns_data):.4f}"
        )
        
        # ตำแหน่งของข้อความสถิติ (ที่มุมขวาบน)
        ax.text(
            0.95, 0.95, 
            stats_text, 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    # กำหนดชื่อแกนและชื่อกราฟ
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Returns', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # เพิ่ม legend
    ax.legend()
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def plot_drawdown_periods(
    equity_curve: Union[List[float], np.ndarray, pd.Series],
    timestamps: Optional[List[Any]] = None,
    min_drawdown: float = 0.05,
    title: str = 'Drawdown Periods',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้างกราฟแสดงช่วงเวลาที่เกิด drawdown

    Parameters:
    equity_curve (List[float], np.ndarray, pd.Series): ข้อมูลเส้นทุน
    timestamps (List[Any], optional): เวลาสำหรับแกน x
    min_drawdown (float): drawdown ขั้นต่ำที่จะแสดง (เช่น 0.05 = 5%)
    title (str): ชื่อกราฟ
    figsize (Tuple[int, int]): ขนาดของกราฟ
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    if isinstance(equity_curve, pd.Series) and timestamps is None:
        timestamps = equity_curve.index
        equity_data = equity_curve.values
    else:
        equity_data = np.asarray(equity_curve)
        if timestamps is None:
            timestamps = np.arange(len(equity_data))
    
    # คำนวณ running maximum
    running_max = np.maximum.accumulate(equity_data)
    
    # คำนวณ drawdown ในแต่ละจุด
    drawdowns = (running_max - equity_data) / running_max
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # พล็อตเส้นทุน
    ax.plot(timestamps, equity_data, linewidth=1.5, color='blue', label='Equity')
    
    # หาช่วงที่เกิด drawdown ที่สำคัญ
    in_drawdown = False
    drawdown_starts = []
    drawdown_ends = []
    drawdown_values = []
    
    for i in range(len(drawdowns)):
        if not in_drawdown and drawdowns[i] >= min_drawdown:
            # เริ่มต้น drawdown
            in_drawdown = True
            # หาจุดเริ่มต้นที่แท้จริง (peak)
            peak_idx = np.argmax(equity_data[:i+1])
            drawdown_starts.append(peak_idx)
        elif in_drawdown and drawdowns[i] < min_drawdown:
            # สิ้นสุด drawdown
            in_drawdown = False
            drawdown_ends.append(i)
            # เก็บค่า drawdown สูงสุดในช่วงนี้
            dd_period = drawdowns[drawdown_starts[-1]:i+1]
            max_dd_idx = np.argmax(dd_period) + drawdown_starts[-1]
            drawdown_values.append((max_dd_idx, drawdowns[max_dd_idx]))
    
    # จัดการกรณี drawdown ยังไม่สิ้นสุด
    if in_drawdown:
        drawdown_ends.append(len(drawdowns) - 1)
        dd_period = drawdowns[drawdown_starts[-1]:]
        max_dd_idx = np.argmax(dd_period) + drawdown_starts[-1]
        drawdown_values.append((max_dd_idx, drawdowns[max_dd_idx]))
    
    # แสดงช่วงเวลาที่เกิด drawdown
    for i in range(len(drawdown_starts)):
        start_idx = drawdown_starts[i]
        end_idx = drawdown_ends[i]
        max_dd_idx, max_dd = drawdown_values[i]
        
        # แรเงาพื้นที่ drawdown
        ax.fill_between(
            timestamps[start_idx:end_idx+1],
            equity_data[start_idx:end_idx+1],
            running_max[start_idx:end_idx+1],
            color='red',
            alpha=0.3
        )
        
        # แสดงค่า drawdown สูงสุด
        ax.annotate(
            f'{max_dd:.2%}',
            xy=(timestamps[max_dd_idx], equity_data[max_dd_idx]),
            xytext=(timestamps[max_dd_idx], equity_data[max_dd_idx] * 0.9),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=10
        )
    
    # กำหนดชื่อแกนและชื่อกราฟ
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time' if not isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)) else 'Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    
    # จัดการรูปแบบแกน x หากเป็นข้อมูลวันที่
    if isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
    
    # จัดรูปแบบแกน y ให้เป็นสกุลเงิน
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    # เพิ่ม legend
    ax.legend()
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def plot_trade_history(
    trade_returns: Union[List[float], np.ndarray, pd.Series],
    trade_timestamps: Optional[List[Any]] = None,
    trade_durations: Optional[List[int]] = None,
    title: str = 'Trade History',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้างกราฟประวัติการเทรด

    Parameters:
    trade_returns (List[float], np.ndarray, pd.Series): ผลตอบแทนของแต่ละการเทรด
    trade_timestamps (List[Any], optional): เวลาของแต่ละการเทรด
    trade_durations (List[int], optional): ระยะเวลาของแต่ละการเทรด
    title (str): ชื่อกราฟ
    figsize (Tuple[int, int]): ขนาดของกราฟ
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    returns_data = np.asarray(trade_returns)
    
    # ถ้ามี timestamps ให้ใช้ timestamps แทน index
    if trade_timestamps is not None:
        x_values = trade_timestamps
        x_label = 'Date'
    else:
        x_values = range(len(returns_data))
        x_label = 'Trade Number'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # แยกผลตอบแทนเป็นกำไรและขาดทุน
    gains = returns_data > 0
    losses = returns_data < 0
    zeros = returns_data == 0
    
    # พล็อตกราฟแท่ง
    bar_width = 0.8
    bars_gain = ax.bar(
        [x for i, x in enumerate(x_values) if gains[i]], 
        [r for i, r in enumerate(returns_data) if gains[i]],
        bar_width, color='green', alpha=0.7, label='Winning Trades'
    )
    bars_loss = ax.bar(
        [x for i, x in enumerate(x_values) if losses[i]], 
        [r for i, r in enumerate(returns_data) if losses[i]],
        bar_width, color='red', alpha=0.7, label='Losing Trades'
    )
    bars_zero = ax.bar(
        [x for i, x in enumerate(x_values) if zeros[i]], 
        [r for i, r in enumerate(returns_data) if zeros[i]],
        bar_width, color='gray', alpha=0.7, label='Flat Trades'
    )
    
    # เพิ่มเส้นแนวนอนที่ศูนย์
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # แสดงระยะเวลาของการเทรดหากมี
    if trade_durations is not None:
        # สร้างแกน y ที่สอง
        ax2 = ax.twinx()
        
        # พล็อตเส้นระยะเวลาการเทรด
        ax2.plot(x_values, trade_durations, 'o--', color='purple', alpha=0.5, markersize=4, label='Duration')
        
        # กำหนดชื่อแกน y ที่สอง
        ax2.set_ylabel('Trade Duration (periods)', color='purple', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # เพิ่ม legend สำหรับแกน y ที่สอง
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        # เพิ่ม legend สำหรับแกน y แรก
        ax.legend(loc='upper left')
    
    # กำหนดชื่อแกนและชื่อกราฟ
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Trade Return', fontsize=12)
    
    # จัดการรูปแบบแกน x หากเป็นข้อมูลวันที่
    if trade_timestamps is not None and isinstance(trade_timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
    
    # จัดรูปแบบแกน y ให้เป็นเปอร์เซ็นต์
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))
    
    # แสดงสถิติการเทรด
    win_rate = np.sum(gains) / len(returns_data)
    avg_win = np.mean(returns_data[gains]) if np.any(gains) else 0
    avg_loss = np.mean(returns_data[losses]) if np.any(losses) else 0
    profit_factor = (np.sum(returns_data[gains]) / -np.sum(returns_data[losses])) if np.any(losses) and np.sum(returns_data[losses]) != 0 else float('inf')
    
    stats_text = (
        f"Total Trades: {len(returns_data)}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Avg Win: {avg_win:.2%}\n"
        f"Avg Loss: {avg_loss:.2%}\n"
        f"Profit Factor: {profit_factor:.2f}"
    )
    
    # ตำแหน่งของข้อความสถิติ (ที่มุมขวาบน)
    ax.text(
        0.95, 0.95, 
        stats_text, 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def plot_underwater_chart(
    equity_curve: Union[List[float], np.ndarray, pd.Series],
    timestamps: Optional[List[Any]] = None,
    title: str = 'Underwater Chart',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้างกราฟ Underwater (Drawdown)

    Parameters:
    equity_curve (List[float], np.ndarray, pd.Series): ข้อมูลเส้นทุน
    timestamps (List[Any], optional): เวลาสำหรับแกน x
    title (str): ชื่อกราฟ
    figsize (Tuple[int, int]): ขนาดของกราฟ
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    if isinstance(equity_curve, pd.Series) and timestamps is None:
        timestamps = equity_curve.index
        equity_data = equity_curve.values
    else:
        equity_data = np.asarray(equity_curve)
        if timestamps is None:
            timestamps = np.arange(len(equity_data))
    
    # คำนวณ running maximum
    running_max = np.maximum.accumulate(equity_data)
    
    # คำนวณ drawdown ในแต่ละจุด
    drawdowns = (running_max - equity_data) / running_max
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # พล็อตกราฟ underwater
    ax.fill_between(timestamps, 0, -drawdowns, color='red', alpha=0.5)
    ax.plot(timestamps, -drawdowns, color='black', linewidth=1)
    
    # เพิ่มเส้นแนวนอนที่ศูนย์
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # กำหนดชื่อแกนและชื่อกราฟ
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time' if not isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)) else 'Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    
    # จัดการรูปแบบแกน x หากเป็นข้อมูลวันที่
    if isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
    
    # จัดรูปแบบแกน y ให้เป็นเปอร์เซ็นต์
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{-x:.0%}'))
    
    # หาค่า max drawdown และแสดง
    max_dd = np.max(drawdowns)
    max_dd_idx = np.argmax(drawdowns)
    
    # แสดงค่า max drawdown
    ax.annotate(
        f'Max DD: {max_dd:.2%}',
        xy=(timestamps[max_dd_idx], -max_dd),
        xytext=(timestamps[max_dd_idx], -max_dd * 0.5 if max_dd > 0.1 else -max_dd * 0.8),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def plot_monthly_returns_heatmap(
    returns: Union[List[float], np.ndarray, pd.Series],
    timestamps: List[pd.Timestamp],
    title: str = 'Monthly Returns',
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'RdYlGn',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้าง heatmap ของผลตอบแทนรายเดือน

    Parameters:
    returns (List[float], np.ndarray, pd.Series): ข้อมูลผลตอบแทน
    timestamps (List[pd.Timestamp]): เวลาของผลตอบแทน
    title (str): ชื่อกราฟ
    figsize (Tuple[int, int]): ขนาดของกราฟ
    cmap (str): colormap ที่ใช้
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    # สร้าง DataFrame จากข้อมูล
    returns_data = pd.Series(returns, index=timestamps)
    
    # จัดกลุ่มตามเดือนและปี แล้วคำนวณผลตอบแทนสะสมรายเดือน
    returns_data = (1 + returns_data).groupby([returns_data.index.year, returns_data.index.month]).prod() - 1
    
    # แปลงเป็น DataFrame
    monthly_returns = returns_data.unstack(level=0)
    
    # เปลี่ยนชื่อ index เป็นชื่อเดือน
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    monthly_returns.index = month_names[:len(monthly_returns)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # สร้าง heatmap
    sns.heatmap(
        monthly_returns,
        annot=True,
        fmt='.2%',
        cmap=cmap,
        center=0,
        linewidths=1,
        ax=ax,
        cbar_kws={'label': 'Monthly Return'}
    )
    
    # กำหนดชื่อแกนและชื่อกราฟ
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    
    # เพิ่มค่าเฉลี่ยรายเดือน (ในแนวตั้ง)
    monthly_means = monthly_returns.mean(axis=1)
    ax.text(
        monthly_returns.shape[1] + 0.5, 
        0.5, 
        'Avg',
        fontsize=10,
        fontweight='bold',
        horizontalalignment='center',
        verticalalignment='center'
    )
    for i, month_mean in enumerate(monthly_means):
        ax.text(
            monthly_returns.shape[1] + 0.5, 
            i + 0.5, 
            f'{month_mean:.2%}',
            fontsize=9,
            horizontalalignment='center',
            verticalalignment='center',
            color='black' if abs(month_mean) < 0.1 else ('white' if month_mean < 0 else 'black'),
            bbox=dict(facecolor='green' if month_mean > 0 else 'red', alpha=0.7)
        )
    
    # เพิ่มค่าเฉลี่ยรายปี (ในแนวนอน)
    yearly_means = monthly_returns.mean(axis=0)
    for i, year in enumerate(monthly_returns.columns):
        yearly_mean = yearly_means[year]
        ax.text(
            i + 0.5, 
            monthly_returns.shape[0] + 0.5, 
            f'{yearly_mean:.2%}',
            fontsize=9,
            horizontalalignment='center',
            verticalalignment='center',
            color='black' if abs(yearly_mean) < 0.1 else ('white' if yearly_mean < 0 else 'black'),
            bbox=dict(facecolor='green' if yearly_mean > 0 else 'red', alpha=0.7)
        )
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def create_performance_tearsheet(
    equity_curve: Union[List[float], np.ndarray, pd.Series],
    returns: Union[List[float], np.ndarray, pd.Series],
    trade_returns: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    timestamps: Optional[List[pd.Timestamp]] = None,
    trade_timestamps: Optional[List[pd.Timestamp]] = None,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    สร้างรายงานผลการเทรด (Performance Tearsheet)

    Parameters:
    equity_curve (List[float], np.ndarray, pd.Series): ข้อมูลเส้นทุน
    returns (List[float], np.ndarray, pd.Series): ข้อมูลผลตอบแทน
    trade_returns (List[float], np.ndarray, pd.Series, optional): ผลตอบแทนของแต่ละการเทรด
    timestamps (List[pd.Timestamp], optional): เวลาของผลตอบแทน
    trade_timestamps (List[pd.Timestamp], optional): เวลาของการเทรด
    figsize (Tuple[int, int]): ขนาดของกราฟ
    save_path (str, optional): พาธสำหรับบันทึกไฟล์กราฟ

    Returns:
    plt.Figure: Figure object ของกราฟ
    """
    # สร้าง DataFrame สำหรับข้อมูล
    if isinstance(equity_curve, pd.Series) and timestamps is None:
        timestamps = equity_curve.index
    
    equity_data = np.asarray(equity_curve)
    returns_data = np.asarray(returns)
    
    # คำนวณสถิติต่างๆ
    if len(returns_data) > 0:
        total_return = (equity_data[-1] / equity_data[0]) - 1 if len(equity_data) > 1 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns_data)) - 1 if len(returns_data) > 0 else 0
        sharpe_ratio = (np.mean(returns_data) / np.std(returns_data, ddof=1)) * np.sqrt(252) if len(returns_data) > 1 else 0
        max_dd, _, _ = (0, 0, 0) if len(equity_data) <= 1 else calculate_max_drawdown(equity_data)
        
        win_rate = 0
        profit_factor = 0
        if trade_returns is not None and len(trade_returns) > 0:
            trade_returns_data = np.asarray(trade_returns)
            wins = np.sum(trade_returns_data > 0)
            win_rate = wins / len(trade_returns_data)
            
            gross_profit = np.sum(trade_returns_data[trade_returns_data > 0])
            gross_loss = abs(np.sum(trade_returns_data[trade_returns_data < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        total_return = 0
        annualized_return = 0
        sharpe_ratio = 0
        max_dd = 0
        win_rate = 0
        profit_factor = 0
    
    # สร้าง figure ที่มีหลาย subplot
    fig = plt.figure(figsize=figsize)
    
    # กำหนด GridSpec เพื่อจัดวาง subplot
    gs = fig.add_gridspec(4, 2)
    
    # Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    if timestamps is not None:
        ax1.plot(timestamps, equity_data, linewidth=2)
        if isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
    else:
        ax1.plot(equity_data, linewidth=2)
    
    ax1.set_title('Equity Curve', fontsize=14)
    ax1.set_ylabel('Equity', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    if len(equity_data) > 1:
        running_max = np.maximum.accumulate(equity_data)
        drawdowns = (running_max - equity_data) / running_max
        
        if timestamps is not None:
            ax2.fill_between(timestamps, 0, -drawdowns, color='red', alpha=0.5)
            ax2.plot(timestamps, -drawdowns, color='black', linewidth=1)
        else:
            x_values = range(len(drawdowns))
            ax2.fill_between(x_values, 0, -drawdowns, color='red', alpha=0.5)
            ax2.plot(x_values, -drawdowns, color='black', linewidth=1)
    
    ax2.set_title('Drawdown Chart', fontsize=14)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{-x:.0%}'))
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Returns Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if len(returns_data) > 0:
        sns.histplot(returns_data, bins=50, kde=True, ax=ax3, color='skyblue')
        ax3.axvline(np.mean(returns_data), color='r', linestyle='--', linewidth=2)
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax3.set_title('Returns Distribution', fontsize=14)
    ax3.set_xlabel('Return', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    
    # Trade Returns (หากมี)
    ax4 = fig.add_subplot(gs[2, :])
    if trade_returns is not None and len(trade_returns) > 0:
        trade_returns_data = np.asarray(trade_returns)
        
        # แยกผลตอบแทนเป็นกำไรและขาดทุน
        gains = trade_returns_data > 0
        losses = trade_returns_data < 0
        
        # กำหนด x values
        if trade_timestamps is not None:
            x_values = trade_timestamps
        else:
            x_values = range(len(trade_returns_data))
        
        # พล็อตกราฟแท่ง
        bar_width = 0.8
        ax4.bar(
            [x for i, x in enumerate(x_values) if gains[i]], 
            [r for i, r in enumerate(trade_returns_data) if gains[i]],
            bar_width, color='green', alpha=0.7
        )
        ax4.bar(
            [x for i, x in enumerate(x_values) if losses[i]], 
            [r for i, r in enumerate(trade_returns_data) if losses[i]],
            bar_width, color='red', alpha=0.7
        )
        
        # จัดการรูปแบบแกน x หากเป็นข้อมูลวันที่
        if trade_timestamps is not None and isinstance(trade_timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('Trade Returns', fontsize=14)
    ax4.set_xlabel('Trade' if trade_timestamps is None else 'Date', fontsize=12)
    ax4.set_ylabel('Return', fontsize=12)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))
    
    # แสดงสถิติ
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    stats_text = (
        f"Performance Metrics\n\n"
        f"Total Return: {total_return:.2%}\n"
        f"Annualized Return: {annualized_return:.2%}\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Max Drawdown: {max_dd:.2%}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Total Trades: {len(trade_returns) if trade_returns is not None else 0}\n"
    )
    
    ax5.text(
        0.5, 0.5,
        stats_text,
        fontsize=12,
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # จัดรูปแบบทั้งหมด
    plt.tight_layout()
    
    # บันทึกไฟล์หากต้องการ
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def save_figure(fig: plt.Figure, save_path: str, dpi: int = 300, transparent: bool = False):
    """
    บันทึกไฟล์กราฟ

    Parameters:
    fig (plt.Figure): Figure object ของกราฟ
    save_path (str): พาธสำหรับบันทึกไฟล์กราฟ
    dpi (int): ความละเอียดของรูปภาพ
    transparent (bool): กำหนดให้พื้นหลังโปร่งใส

    Returns:
    bool: True หากบันทึกสำเร็จ, False หากไม่สำเร็จ
    """
    try:
        # สร้างโฟลเดอร์หากไม่มี
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # บันทึกไฟล์
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=transparent)
        logger.info(f"บันทึกกราฟที่ {save_path}")
        return True
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการบันทึกกราฟ: {e}")
        return False

class TradingDashboard:
    """
    แดชบอร์ดสำหรับแสดงผลการเทรดแบบเรียลไทม์
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), update_interval: int = 1000):
        """
        กำหนดค่าเริ่มต้นสำหรับ TradingDashboard
        
        Parameters:
        figsize (Tuple[int, int]): ขนาดของกราฟ
        update_interval (int): ช่วงเวลาในการอัพเดต (มิลลิวินาที)
        """
        self.figsize = figsize
        self.update_interval = update_interval
        
        # สร้าง figure และ axes
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.fig.tight_layout(pad=3.0)
        
        # ข้อมูลที่ใช้แสดง
        self.equity_data = []
        self.return_data = []
        self.position_data = []
        self.trade_data = []
        self.timestamps = []
        
        # กำหนดชื่อกราฟ
        self.axes[0, 0].set_title('Equity Curve', fontsize=14)
        self.axes[0, 1].set_title('Drawdown', fontsize=14)
        self.axes[1, 0].set_title('Position History', fontsize=14)
        self.axes[1, 1].set_title('Performance Metrics', fontsize=14)
        
        # ปิดการแสดงแกนสำหรับกราฟเมตริก
        self.axes[1, 1].axis('off')
        
        # สร้างวัตถุ Line2D สำหรับการอัพเดต
        self.equity_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        self.drawdown_fill = None
        self.position_line, = self.axes[1, 0].plot([], [], 'g-', linewidth=2)
        
        # กำหนดค่าแกน y สำหรับกราฟตำแหน่ง
        self.axes[1, 0].set_ylim(-1.5, 1.5)
        self.axes[1, 0].set_yticks([-1, 0, 1])
        self.axes[1, 0].set_yticklabels(['Short', 'Flat', 'Long'])
        
        # เพิ่มเส้นแนวนอนที่ศูนย์สำหรับกราฟตำแหน่ง
        self.axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # จัดรูปแบบ
        self.fig.tight_layout()
    
    def initialize_plot(self, initial_equity: float):
        """
        กำหนดค่าเริ่มต้นของกราฟ
        
        Parameters:
        initial_equity (float): เงินทุนเริ่มต้น
        """
        # เพิ่มข้อมูลเริ่มต้น
        self.equity_data = [initial_equity]
        self.return_data = []
        self.position_data = [0]  # เริ่มต้นไม่มีตำแหน่ง
        self.trade_data = []
        self.timestamps = [0]  # เริ่มต้นที่เวลา 0
        
        # อัพเดตกราฟ
        self.update_plot()
    
    def update_data(self, equity: float, position: int, timestamp: Optional[float] = None, 
                   trade_return: Optional[float] = None):
        """
        อัพเดตข้อมูลสำหรับแสดงผล
        
        Parameters:
        equity (float): มูลค่าพอร์ตล่าสุด
        position (int): ตำแหน่งปัจจุบัน (-1: Short, 0: Flat, 1: Long)
        timestamp (float, optional): เวลาปัจจุบัน
        trade_return (float, optional): ผลตอบแทนจากการเทรดครั้งล่าสุด
        """
        # อัพเดตเวลาหากไม่ได้ระบุ
        if timestamp is None:
            timestamp = self.timestamps[-1] + 1 if self.timestamps else 0
        
        # เพิ่มข้อมูลใหม่
        self.equity_data.append(equity)
        self.position_data.append(position)
        self.timestamps.append(timestamp)
        
        # คำนวณผลตอบแทน
        if len(self.equity_data) > 1:
            ret = (equity / self.equity_data[-2]) - 1
            self.return_data.append(ret)
        
        # เพิ่มข้อมูลการเทรดหากมี
        if trade_return is not None:
            self.trade_data.append((timestamp, trade_return))
        
        # อัพเดตกราฟ
        self.update_plot()
    
    def update_plot(self):
        """
        อัพเดตการแสดงผลของกราฟ
        """
        if not self.timestamps:
            return
        
        # อัพเดต Equity Curve
        self.equity_line.set_data(self.timestamps, self.equity_data)
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()
        
        # อัพเดต Drawdown
        self.axes[0, 1].clear()
        self.axes[0, 1].set_title('Drawdown', fontsize=14)
        
        if len(self.equity_data) > 1:
            # คำนวณ drawdown
            running_max = np.maximum.accumulate(self.equity_data)
            drawdowns = (running_max - self.equity_data) / running_max
            
            # แสดง drawdown
            self.axes[0, 1].fill_between(
                self.timestamps, 
                0, 
                -drawdowns, 
                color='red', 
                alpha=0.5
            )
            self.axes[0, 1].plot(self.timestamps, -drawdowns, 'k-', linewidth=1)
            
            # กำหนดค่าแกน y
            self.axes[0, 1].set_ylim(min(-np.max(drawdowns) * 1.1, -0.05), 0.01)
            
            # จัดรูปแบบแกน y ให้เป็นเปอร์เซ็นต์
            self.axes[0, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{-x:.0%}'))
        
        # อัพเดต Position History
        self.position_line.set_data(self.timestamps, self.position_data)
        self.axes[1, 0].relim()
        self.axes[1, 0].set_xlim(min(self.timestamps), max(self.timestamps))
        
        # อัพเดต Performance Metrics
        self.axes[1, 1].clear()
        self.axes[1, 1].axis('off')
        
        # คำนวณเมตริกประสิทธิภาพ
        if len(self.equity_data) > 1 and len(self.return_data) > 0:
            total_return = (self.equity_data[-1] / self.equity_data[0]) - 1
            
            # คำนวณ Sharpe Ratio (annualized)
            sharpe_ratio = (np.mean(self.return_data) / np.std(self.return_data, ddof=1)) * np.sqrt(252) if len(self.return_data) > 1 else 0
            
            # คำนวณ Max Drawdown
            running_max = np.maximum.accumulate(self.equity_data)
            drawdowns = (running_max - self.equity_data) / running_max
            max_dd = np.max(drawdowns)
            
            # คำนวณสถิติการเทรด
            wins = [t[1] for t in self.trade_data if t[1] > 0]
            losses = [t[1] for t in self.trade_data if t[1] < 0]
            
            win_rate = len(wins) / len(self.trade_data) if self.trade_data else 0
            
            profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
            
            # แสดงเมตริก
            metrics_text = (
                f"Total Return: {total_return:.2%}\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Max Drawdown: {max_dd:.2%}\n"
                f"Win Rate: {win_rate:.2%}\n"
                f"Profit Factor: {profit_factor:.2f}\n"
                f"Total Trades: {len(self.trade_data)}\n"
                f"Current Equity: {self.equity_data[-1]:,.2f}\n"
                f"Current Position: {'Long' if self.position_data[-1] > 0 else ('Short' if self.position_data[-1] < 0 else 'Flat')}"
            )
            
            self.axes[1, 1].text(
                0.5, 0.5,
                metrics_text,
                fontsize=12,
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
            )
        
        # จัดรูปแบบ
        self.fig.tight_layout()
        plt.pause(0.001)
    
    def show(self):
        """
        แสดงแดชบอร์ด
        """
        plt.show()
    
    def save(self, save_path: str, dpi: int = 300):
        """
        บันทึกแดชบอร์ดเป็นไฟล์รูปภาพ
        
        Parameters:
        save_path (str): พาธสำหรับบันทึกไฟล์กราฟ
        dpi (int): ความละเอียดของรูปภาพ
        
        Returns:
        bool: True หากบันทึกสำเร็จ, False หากไม่สำเร็จ
        """
        return save_figure(self.fig, save_path, dpi)
    
    def close(self):
        """
        ปิดแดชบอร์ด
        """
        plt.close(self.fig)