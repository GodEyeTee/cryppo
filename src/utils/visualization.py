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
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(equity_curve, pd.Series) and timestamps is None:
        timestamps = equity_curve.index
        equity_data = equity_curve.values
    else:
        equity_data = np.asarray(equity_curve)
    
    if timestamps is not None:
        ax.plot(timestamps, equity_data, linewidth=2, label='Equity')
        
        if isinstance(timestamps[0], (pd.Timestamp, pd.DatetimeIndex)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
    else:
        ax.plot(equity_data, linewidth=2, label='Equity')
    
    if show_drawdown and len(equity_data) > 1:
        running_max = np.maximum.accumulate(equity_data)
        drawdowns = (running_max - equity_data) / running_max
        max_dd = np.max(drawdowns)
        max_dd_idx = np.argmax(drawdowns)
        peak_idx = np.argmax(equity_data[:max_dd_idx+1])
        
        if timestamps is not None:
            ax.fill_between(timestamps, equity_data, running_max, color='red', alpha=0.3, 
                          label=f'Drawdown (Max: {max_dd:.2%})')
            ax.plot(timestamps[peak_idx], equity_data[peak_idx], 'go', markersize=8, label='Peak')
            ax.plot(timestamps[max_dd_idx], equity_data[max_dd_idx], 'ro', markersize=8, label='Trough')
        else:
            ax.fill_between(range(len(equity_data)), equity_data, running_max, color='red', 
                          alpha=0.3, label=f'Drawdown (Max: {max_dd:.2%})')
            ax.plot(peak_idx, equity_data[peak_idx], 'go', markersize=8, label='Peak')
            ax.plot(max_dd_idx, equity_data[max_dd_idx], 'ro', markersize=8, label='Trough')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time' if timestamps is None else 'Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig

def save_figure(fig: plt.Figure, save_path: str, dpi: int = 300, transparent: bool = False):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=transparent)
        logger.info(f"Saved chart to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving chart: {e}")
        return False