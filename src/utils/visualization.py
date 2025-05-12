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
    # Existing implementation...
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

# Add missing functions
def plot_returns_distribution(returns, title="Returns Distribution", figsize=(12, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(returns, kde=True, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Return (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.axvline(0, color='r', linestyle='--')
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig

def plot_drawdown_periods(equity_curve, title="Drawdown Periods", figsize=(12, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    equity_data = np.asarray(equity_curve)
    running_max = np.maximum.accumulate(equity_data)
    drawdowns = (running_max - equity_data) / running_max
    ax.plot(drawdowns * 100, color='red', linewidth=2)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.grid(True)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig

def plot_trade_history(trades_df, title="Trade History", figsize=(12, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    if 'timestamp' in trades_df.columns:
        x = trades_df['timestamp']
    else:
        x = range(len(trades_df))
    
    if 'price' in trades_df.columns:
        ax.plot(x, trades_df['price'], label='Price', color='blue')
    
    if 'action' in trades_df.columns and 'price' in trades_df.columns:
        buys = trades_df[trades_df['action'] == 1]
        sells = trades_df[trades_df['action'] == 3]
        shorts = trades_df[trades_df['action'] == 2]
        
        if not buys.empty:
            buy_x = buys.index if 'timestamp' not in trades_df.columns else buys['timestamp']
            ax.scatter(buy_x, buys['price'], marker='^', color='green', s=100, label='Buy')
        
        if not sells.empty:
            sell_x = sells.index if 'timestamp' not in trades_df.columns else sells['timestamp']
            ax.scatter(sell_x, sells['price'], marker='v', color='red', s=100, label='Sell')
        
        if not shorts.empty:
            short_x = shorts.index if 'timestamp' not in trades_df.columns else shorts['timestamp']
            ax.scatter(short_x, shorts['price'], marker='s', color='purple', s=100, label='Short')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig

def plot_underwater_chart(equity_curve, title="Underwater Chart", figsize=(12, 6), save_path=None):
    # This is just another name for the drawdown chart
    return plot_drawdown_periods(equity_curve, title, figsize, save_path)

def plot_monthly_returns_heatmap(returns, title="Monthly Returns", figsize=(12, 8), save_path=None):
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        return None
    
    monthly_returns = returns.resample('M').mean().to_frame()
    monthly_returns['year'] = monthly_returns.index.year
    monthly_returns['month'] = monthly_returns.index.month
    
    pivot_table = monthly_returns.pivot_table(
        values=monthly_returns.columns[0], 
        index='year', 
        columns='month'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_table, cmap='RdYlGn', annot=True, fmt='.2%', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig

def create_performance_tearsheet(equity_curve, returns, metrics, save_path=None):
    fig = plt.figure(figsize=(12, 15))
    
    # 1. Equity Curve
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax1.plot(equity_curve, label='Equity', color='blue', linewidth=2)
    ax1.set_title('Equity Curve', fontsize=14)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Equity')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Returns Distribution
    ax2 = plt.subplot2grid((4, 2), (1, 0))
    sns.histplot(returns, kde=True, ax=ax2)
    ax2.set_title('Returns Distribution', fontsize=14)
    ax2.set_xlabel('Return (%)')
    ax2.axvline(0, color='r', linestyle='--')
    
    # 3. Drawdown Periods
    ax3 = plt.subplot2grid((4, 2), (1, 1))
    equity_data = np.asarray(equity_curve)
    running_max = np.maximum.accumulate(equity_data)
    drawdowns = (running_max - equity_data) / running_max
    ax3.plot(drawdowns * 100, color='red', linewidth=2)
    ax3.set_title('Drawdown Periods', fontsize=14)
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True)
    ax3.invert_yaxis()
    
    # 4. Metrics Table
    ax4 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    ax4.axis('off')
    
    metrics_text = (
        f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
        f"CAGR: {metrics.get('cagr', 0):.2f}%\n"
        f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
        f"Win Rate: {metrics.get('win_rate', 0):.2f}%\n"
        f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        f"Total Trades: {metrics.get('total_trades', 0)}\n"
    )
    
    ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12)
    
    # 5. Monthly Returns (if available)
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        ax5 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
        try:
            monthly_returns = returns.resample('M').mean().to_frame()
            monthly_returns['year'] = monthly_returns.index.year
            monthly_returns['month'] = monthly_returns.index.month
            
            pivot_table = monthly_returns.pivot_table(
                values=monthly_returns.columns[0], 
                index='year', 
                columns='month'
            )
            
            sns.heatmap(pivot_table, cmap='RdYlGn', annot=True, fmt='.2%', ax=ax5)
            ax5.set_title('Monthly Returns', fontsize=14)
            ax5.set_xlabel('Month')
            ax5.set_ylabel('Year')
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax5.set_xticklabels(month_names)
        except:
            ax5.text(0.5, 0.5, "Monthly returns not available", ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_backtest_results(trades_df, raw_data=None, metrics=None):
    """Plot backtest results for CLI commands"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price series
    if 'timestamp' in trades_df.columns:
        x = trades_df['timestamp']
    else:
        x = range(len(trades_df))
    
    if 'price' in trades_df.columns:
        ax1.plot(x, trades_df['price'], label='Price', color='blue')
    
    # Plot portfolio value
    if 'portfolio_value' in trades_df.columns:
        ax1.plot(x, trades_df['portfolio_value'], label='Portfolio', color='green')
    
    # Plot trade actions
    if 'action' in trades_df.columns and 'price' in trades_df.columns:
        mask_buy = trades_df['action'] == 1
        mask_sell = trades_df['action'] == 3
        mask_short = trades_df['action'] == 2
        
        if mask_buy.any():
            ax1.scatter(x[mask_buy], trades_df.loc[mask_buy, 'price'], 
                      marker='^', color='green', s=100, label='Buy')
        
        if mask_sell.any():
            ax1.scatter(x[mask_sell], trades_df.loc[mask_sell, 'price'], 
                      marker='v', color='red', s=100, label='Sell')
        
        if mask_short.any():
            ax1.scatter(x[mask_short], trades_df.loc[mask_short, 'price'], 
                      marker='s', color='purple', s=100, label='Short')
    
    ax1.set_title('Backtest Results', fontsize=16)
    ax1.set_ylabel('Price / Portfolio Value', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # Plot position
    if 'position' in trades_df.columns:
        # Convert position from string to numeric
        pos_numeric = trades_df['position'].map({'none': 0, 'long': 1, 'short': -1})
        ax2.plot(x, pos_numeric, color='purple', drawstyle='steps')
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Short', 'None', 'Long'])
        ax2.grid(True)
    
    # Plot rewards/returns
    if 'reward' in trades_df.columns:
        colors = ['green' if r >= 0 else 'red' for r in trades_df['reward']]
        ax3.bar(x, trades_df['reward'], color=colors, alpha=0.7)
        ax3.set_ylabel('Reward', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.grid(True)
    
    # Add metrics text box if available
    if metrics:
        metrics_text = (
            f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
            f"Win Rate: {metrics.get('win_rate', 0):.2f}%"
        )
        ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, 
               bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_backtest_analysis(trades_df, metrics=None, raw_data=None, plot_type='all', period='daily'):
    """Create analysis plots for backtest results"""
    if plot_type == 'all':
        fig = plt.figure(figsize=(16, 16))
        
        # 1. Equity curve
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        if 'portfolio_value' in trades_df.columns:
            ax1.plot(trades_df['portfolio_value'], label='Portfolio Value', color='blue')
            ax1.set_title('Equity Curve', fontsize=14)
            ax1.set_ylabel('Portfolio Value')
            ax1.grid(True)
    
        # 2. Drawdown
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        if 'portfolio_value' in trades_df.columns:
            equity = trades_df['portfolio_value'].values
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max * 100
            ax2.plot(drawdowns, color='red')
            ax2.set_title('Drawdown', fontsize=14)
            ax2.set_ylabel('Drawdown (%)')
            ax2.invert_yaxis()
            ax2.grid(True)
        
        # 3. Return Distribution
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        if 'portfolio_value' in trades_df.columns:
            returns = trades_df['portfolio_value'].pct_change().dropna() * 100
            sns.histplot(returns, kde=True, ax=ax3)
            ax3.axvline(0, color='r', linestyle='--')
            ax3.set_title('Daily Returns Distribution', fontsize=14)
            ax3.set_xlabel('Return (%)')
        
        # 4. Trade Analysis
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        if 'action' in trades_df.columns:
            action_counts = trades_df['action'].value_counts().sort_index()
            ax4.bar(action_counts.index, action_counts.values, color='purple')
            ax4.set_title('Trade Actions', fontsize=14)
            ax4.set_xlabel('Action (0=None, 1=Long, 2=Short, 3=Exit)')
            ax4.set_ylabel('Count')
            ax4.set_xticks(action_counts.index)
        
        # 5. Metrics
        ax5 = plt.subplot2grid((3, 2), (2, 1))
        ax5.axis('off')
        if metrics:
            metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
            ax5.text(0.1, 0.5, metrics_text, ha='left', va='center', fontsize=12)
        else:
            ax5.text(0.5, 0.5, "No metrics available", ha='center', va='center')
    
    elif plot_type == 'equity':
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'portfolio_value' in trades_df.columns:
            ax.plot(trades_df['portfolio_value'], label='Portfolio Value', color='blue')
            ax.set_title('Equity Curve', fontsize=16)
            ax.set_ylabel('Portfolio Value')
            ax.grid(True)
    
    elif plot_type == 'drawdown':
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'portfolio_value' in trades_df.columns:
            equity = trades_df['portfolio_value'].values
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max * 100
            ax.plot(drawdowns, color='red')
            ax.set_title('Drawdown', fontsize=16)
            ax.set_ylabel('Drawdown (%)')
            ax.invert_yaxis()
            ax.grid(True)
    
    elif plot_type == 'returns':
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'portfolio_value' in trades_df.columns:
            if period == 'daily':
                freq = 'D'
                title = 'Daily Returns'
            elif period == 'weekly':
                freq = 'W'
                title = 'Weekly Returns'
            else:
                freq = 'M'
                title = 'Monthly Returns'
            
            # Convert to series with datetime index if needed
            if 'timestamp' in trades_df.columns:
                portfolio = pd.Series(trades_df['portfolio_value'].values, 
                                     index=pd.to_datetime(trades_df['timestamp']))
            else:
                portfolio = trades_df['portfolio_value']
            
            # Resample and calculate returns
            period_returns = portfolio.resample(freq).last().pct_change().dropna() * 100
            sns.histplot(period_returns, kde=True, ax=ax)
            ax.axvline(0, color='r', linestyle='--')
            ax.set_title(f'{title} Distribution', fontsize=16)
            ax.set_xlabel('Return (%)')
    
    elif plot_type == 'trades':
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'price' in trades_df.columns:
            if 'timestamp' in trades_df.columns:
                x = trades_df['timestamp']
            else:
                x = range(len(trades_df))
            
            ax.plot(x, trades_df['price'], label='Price', color='blue')
            
            if 'action' in trades_df.columns:
                mask_buy = trades_df['action'] == 1
                mask_sell = trades_df['action'] == 3
                mask_short = trades_df['action'] == 2
                
                if mask_buy.any():
                    ax.scatter(x[mask_buy], trades_df.loc[mask_buy, 'price'], 
                             marker='^', color='green', s=100, label='Buy')
                
                if mask_sell.any():
                    ax.scatter(x[mask_sell], trades_df.loc[mask_sell, 'price'], 
                             marker='v', color='red', s=100, label='Sell')
                
                if mask_short.any():
                    ax.scatter(x[mask_short], trades_df.loc[mask_short, 'price'], 
                             marker='s', color='purple', s=100, label='Short')
            
            ax.set_title('Trade History', fontsize=16)
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    return fig

class TradingDashboard:
    """Interactive dashboard for trading performance analysis"""
    def __init__(self, equity_curve, trades=None, raw_data=None):
        self.equity_curve = equity_curve
        self.trades = trades
        self.raw_data = raw_data
        self.metrics = None
        
    def calculate_metrics(self):
        # Placeholder for calculating metrics
        pass
        
    def show(self):
        # Placeholder for showing dashboard
        pass