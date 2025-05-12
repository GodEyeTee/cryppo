import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger('utils.metrics')

def calculate_returns(
    equity_curve: Union[List[float], np.ndarray],
    log_returns: bool = False
) -> np.ndarray:
    equity = np.asarray(equity_curve)
    
    if log_returns:
        returns = np.log(equity[1:] / equity[:-1])
    else:
        returns = (equity[1:] / equity[:-1]) - 1
    
    return returns

def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    if len(returns) < 2:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = np.asarray(returns) - rf_per_period
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)
    
    if std_excess_return == 0:
        return 0.0
    
    return (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)

def calculate_max_drawdown(
    equity_curve: Union[List[float], np.ndarray]
) -> Tuple[float, int, int]:
    equity = np.asarray(equity_curve)
    
    if len(equity) < 2:
        return 0.0, 0, 0
    
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_drawdown = np.max(drawdowns)
    trough_idx = np.argmax(drawdowns)
    peak_idx = np.argmax(equity[:trough_idx+1])
    
    return max_drawdown, peak_idx, trough_idx

def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    if len(returns) < 2:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    returns_array = np.asarray(returns)
    excess_returns = returns_array - rf_per_period
    
    downside_returns = np.minimum(returns_array - target_return, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    return (np.mean(excess_returns) / downside_deviation) * np.sqrt(periods_per_year)

def calculate_win_rate(
    trade_returns: Union[List[float], np.ndarray]
) -> float:
    if len(trade_returns) == 0:
        return 0.0
    
    trades = np.asarray(trade_returns)
    wins = np.sum(trades > 0)
    
    return wins / len(trades)

def calculate_profit_factor(
    trade_returns: Union[List[float], np.ndarray]
) -> float:
    trades = np.asarray(trade_returns)
    
    if len(trades) == 0:
        return 0.0
    
    gross_profits = np.sum(trades[trades > 0])
    gross_losses = abs(np.sum(trades[trades < 0]))
    
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    
    return gross_profits / gross_losses

def calculate_cagr(
    equity_curve: Union[List[float], np.ndarray],
    periods_per_year: int = 252
) -> float:
    equity = np.asarray(equity_curve)
    
    if len(equity) <= 1:
        return 0.0
    
    years = len(equity) / periods_per_year
    return (equity[-1] / equity[0]) ** (1 / years) - 1

def calculate_volatility(
    returns: Union[List[float], np.ndarray],
    periods_per_year: int = 252
) -> float:
    if len(returns) < 2:
        return 0.0
    
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

@dataclass
class RiskMetrics:
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    calmar_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility": self.volatility,
            "downside_deviation": self.downside_deviation,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "calmar_ratio": self.calmar_ratio
        }
    
class PerformanceTracker:  
    def __init__(self, initial_equity: float = 10000.0, periods_per_year: int = 252):
        self.initial_equity = initial_equity
        self.periods_per_year = periods_per_year
        
        self.equity_curve = [initial_equity]
        self.returns = []
        self.trade_returns = []
        self.positions = []
        self.timestamps = []
        
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def update(self, equity: float, timestamp: Optional[pd.Timestamp] = None, 
              position: int = 0, trade_return: Optional[float] = None):
        self.equity_curve.append(equity)
        
        if len(self.equity_curve) > 1:
            last_return = (equity / self.equity_curve[-2]) - 1
            self.returns.append(last_return)
        
        self.positions.append(position)
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
        
        if trade_return is not None:
            self.trade_returns.append(trade_return)
            self.trades_count += 1
            
            if trade_return > 0:
                self.winning_trades += 1
            elif trade_return < 0:
                self.losing_trades += 1
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        metrics = {}
        
        if len(self.equity_curve) > 1:
            metrics['total_return'] = (self.equity_curve[-1] / self.equity_curve[0]) - 1
            metrics['cagr'] = calculate_cagr(self.equity_curve, self.periods_per_year)
        else:
            metrics['total_return'] = 0.0
            metrics['cagr'] = 0.0
        
        if len(self.returns) > 0:
            metrics['volatility'] = calculate_volatility(self.returns, self.periods_per_year)
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(self.returns, risk_free_rate, self.periods_per_year)
            metrics['sortino_ratio'] = calculate_sortino_ratio(self.returns, risk_free_rate, self.periods_per_year)
            
            max_dd, _, _ = calculate_max_drawdown(self.equity_curve)
            metrics['max_drawdown'] = max_dd
            
            if metrics['max_drawdown'] > 0 and metrics['cagr'] > 0:
                metrics['calmar_ratio'] = metrics['cagr'] / metrics['max_drawdown']
            else:
                metrics['calmar_ratio'] = 0.0
        else:
            metrics['volatility'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
            metrics['max_drawdown'] = 0.0
            metrics['calmar_ratio'] = 0.0
        
        if self.trades_count > 0:
            metrics['trades_count'] = self.trades_count
            metrics['win_rate'] = self.winning_trades / self.trades_count
            metrics['profit_factor'] = calculate_profit_factor(self.trade_returns)
            metrics['avg_trade_return'] = np.mean(self.trade_returns) if len(self.trade_returns) > 0 else 0.0
            metrics['avg_win_return'] = np.mean([r for r in self.trade_returns if r > 0]) if self.winning_trades > 0 else 0.0
            metrics['avg_loss_return'] = np.mean([r for r in self.trade_returns if r < 0]) if self.losing_trades > 0 else 0.0
        else:
            metrics['trades_count'] = 0
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['avg_trade_return'] = 0.0
            metrics['avg_win_return'] = 0.0
            metrics['avg_loss_return'] = 0.0
        
        return metrics
    
    def get_risk_metrics(self, risk_free_rate: float = 0.0) -> RiskMetrics:
        if len(self.returns) < 2:
            return RiskMetrics()
        
        metrics = RiskMetrics()
        
        metrics.sharpe_ratio = calculate_sharpe_ratio(self.returns, risk_free_rate, self.periods_per_year)
        metrics.sortino_ratio = calculate_sortino_ratio(self.returns, risk_free_rate, self.periods_per_year)
        
        metrics.max_drawdown, peak_idx, trough_idx = calculate_max_drawdown(self.equity_curve)
        metrics.max_drawdown_duration = trough_idx - peak_idx
        
        metrics.volatility = calculate_volatility(self.returns, self.periods_per_year)
        
        downside_returns = np.minimum(np.asarray(self.returns), 0)
        metrics.downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.periods_per_year)
        
        if len(self.returns) > 0:
            metrics.var_95 = np.percentile(self.returns, 5)
            metrics.var_99 = np.percentile(self.returns, 1)
            
            var_95_mask = np.asarray(self.returns) <= metrics.var_95
            metrics.cvar_95 = np.mean(np.asarray(self.returns)[var_95_mask]) if np.any(var_95_mask) else metrics.var_95
        
        if metrics.max_drawdown > 0:
            total_return = (self.equity_curve[-1] / self.equity_curve[0]) - 1
            years = len(self.equity_curve) / self.periods_per_year
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            metrics.calmar_ratio = annual_return / metrics.max_drawdown
        
        return metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        data = {
            'equity': self.equity_curve[1:],
            'position': self.positions
        }
        
        if len(self.returns) > 0:
            data['return'] = self.returns
        
        if len(self.timestamps) > 0:
            df = pd.DataFrame(data, index=self.timestamps)
        else:
            df = pd.DataFrame(data)
        
        return df
    
    def reset(self, initial_equity: Optional[float] = None):
        if initial_equity is not None:
            self.initial_equity = initial_equity
        
        self.equity_curve = [self.initial_equity]
        self.returns = []
        self.trade_returns = []
        self.positions = []
        self.timestamps = []
        
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0