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
    """
    คำนวณอัตราผลตอบแทนจากเส้นทุน

    Parameters:
    equity_curve (List[float] or np.ndarray): เส้นทุนหรือเงินทุนในแต่ละช่วงเวลา
    log_returns (bool): หากเป็น True จะคำนวณ log returns แทน simple returns

    Returns:
    np.ndarray: อัตราผลตอบแทนในแต่ละช่วงเวลา
    """
    equity = np.asarray(equity_curve)
    
    if log_returns:
        # Log returns: log(P_t / P_{t-1})
        returns = np.log(equity[1:] / equity[:-1])
    else:
        # Simple returns: (P_t / P_{t-1}) - 1
        returns = (equity[1:] / equity[:-1]) - 1
    
    return returns

def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    คำนวณอัตราส่วนชาร์ป (Sharpe Ratio)

    Parameters:
    returns (List[float] or np.ndarray): อัตราผลตอบแทนในแต่ละช่วงเวลา
    risk_free_rate (float): อัตราผลตอบแทนที่ปราศจากความเสี่ยง (annualized)
    periods_per_year (int): จำนวนช่วงเวลาต่อปี (252: วันทำการ, 365: วันปกติ, 12: เดือน)

    Returns:
    float: อัตราส่วนชาร์ป (annualized)
    """
    if len(returns) < 2:
        return 0.0
    
    # แปลง risk-free rate เป็นรายช่วงเวลา
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # คำนวณ excess returns
    excess_returns = np.asarray(returns) - rf_per_period
    
    # คำนวณ annualized Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)
    
    if std_excess_return == 0:
        return 0.0
    
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)
    
    return sharpe_ratio

def calculate_max_drawdown(
    equity_curve: Union[List[float], np.ndarray]
) -> Tuple[float, int, int]:
    """
    คำนวณ maximum drawdown และช่วงเวลาที่เกิด

    Parameters:
    equity_curve (List[float] or np.ndarray): เส้นทุนหรือเงินทุนในแต่ละช่วงเวลา

    Returns:
    Tuple[float, int, int]: (max_drawdown, peak_idx, trough_idx)
        - max_drawdown: ค่า drawdown สูงสุด (เป็นสัดส่วน)
        - peak_idx: ดัชนีของจุด peak
        - trough_idx: ดัชนีของจุด trough
    """
    equity = np.asarray(equity_curve)
    
    if len(equity) < 2:
        return 0.0, 0, 0
    
    # คำนวณ running maximum
    running_max = np.maximum.accumulate(equity)
    
    # คำนวณ drawdown ในแต่ละจุด
    drawdowns = (running_max - equity) / running_max
    
    # หาค่า drawdown สูงสุด
    max_drawdown = np.max(drawdowns)
    
    # หาช่วงเวลาที่เกิด max drawdown
    trough_idx = np.argmax(drawdowns)
    
    # หาจุด peak ก่อนหน้าที่เกิด max drawdown
    peak_idx = np.argmax(equity[:trough_idx+1])
    
    return max_drawdown, peak_idx, trough_idx

def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    คำนวณอัตราส่วนซอร์ติโน (Sortino Ratio)

    Parameters:
    returns (List[float] or np.ndarray): อัตราผลตอบแทนในแต่ละช่วงเวลา
    risk_free_rate (float): อัตราผลตอบแทนที่ปราศจากความเสี่ยง (annualized)
    periods_per_year (int): จำนวนช่วงเวลาต่อปี
    target_return (float): อัตราผลตอบแทนเป้าหมายต่อช่วงเวลา

    Returns:
    float: อัตราส่วนซอร์ติโน (annualized)
    """
    if len(returns) < 2:
        return 0.0
    
    # แปลง risk-free rate เป็นรายช่วงเวลา
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # คำนวณ excess returns
    returns_array = np.asarray(returns)
    excess_returns = returns_array - rf_per_period
    
    # คำนวณ downside deviation
    downside_returns = np.minimum(returns_array - target_return, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    # คำนวณ Sortino ratio
    sortino_ratio = (np.mean(excess_returns) / downside_deviation) * np.sqrt(periods_per_year)
    
    return sortino_ratio

def calculate_win_rate(
    trade_returns: Union[List[float], np.ndarray]
) -> float:
    """
    คำนวณอัตราการชนะ (Win Rate)

    Parameters:
    trade_returns (List[float] or np.ndarray): ผลตอบแทนของแต่ละการเทรด

    Returns:
    float: อัตราการชนะ (0.0 - 1.0)
    """
    if len(trade_returns) == 0:
        return 0.0
    
    trades = np.asarray(trade_returns)
    wins = np.sum(trades > 0)
    
    return wins / len(trades)

def calculate_profit_factor(
    trade_returns: Union[List[float], np.ndarray]
) -> float:
    """
    คำนวณ Profit Factor

    Parameters:
    trade_returns (List[float] or np.ndarray): ผลตอบแทนของแต่ละการเทรด

    Returns:
    float: Profit Factor (gross profits / gross losses)
    """
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
    """
    คำนวณอัตราการเติบโตเฉลี่ยต่อปี (CAGR)

    Parameters:
    equity_curve (List[float] or np.ndarray): เส้นทุนหรือเงินทุนในแต่ละช่วงเวลา
    periods_per_year (int): จำนวนช่วงเวลาต่อปี

    Returns:
    float: CAGR (Compound Annual Growth Rate)
    """
    equity = np.asarray(equity_curve)
    
    if len(equity) <= 1:
        return 0.0
    
    # จำนวนปี
    years = len(equity) / periods_per_year
    
    # คำนวณ CAGR
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    
    return cagr

def calculate_volatility(
    returns: Union[List[float], np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    คำนวณความผันผวน (annualized)

    Parameters:
    returns (List[float] or np.ndarray): อัตราผลตอบแทนในแต่ละช่วงเวลา
    periods_per_year (int): จำนวนช่วงเวลาต่อปี

    Returns:
    float: ความผันผวนรายปี
    """
    if len(returns) < 2:
        return 0.0
    
    # คำนวณความผันผวนรายปี
    volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    return volatility

@dataclass
class RiskMetrics:
    """
    เก็บข้อมูลการวัดความเสี่ยงต่างๆ
    """
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    var_99: float = 0.0  # Value at Risk (99%)
    cvar_95: float = 0.0  # Conditional Value at Risk (95%)
    calmar_ratio: float = 0.0  # Returns / Max Drawdown
    
    def to_dict(self) -> Dict[str, float]:
        """แปลงเป็น dict"""
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
    """
    ติดตามและคำนวณประสิทธิภาพของระบบเทรด
    """
    
    def __init__(self, initial_equity: float = 10000.0, periods_per_year: int = 252):
        """
        กำหนดค่าเริ่มต้นสำหรับ PerformanceTracker
        
        Parameters:
        initial_equity (float): เงินทุนเริ่มต้น
        periods_per_year (int): จำนวนช่วงเวลาต่อปี
        """
        self.initial_equity = initial_equity
        self.periods_per_year = periods_per_year
        
        # ข้อมูลที่ติดตาม
        self.equity_curve = [initial_equity]
        self.returns = []
        self.trade_returns = []
        self.positions = []
        self.timestamps = []
        
        # สถิติการเทรด
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def update(self, equity: float, timestamp: Optional[pd.Timestamp] = None, 
              position: int = 0, trade_return: Optional[float] = None):
        """
        อัพเดตข้อมูลประสิทธิภาพ
        
        Parameters:
        equity (float): มูลค่าพอร์ตล่าสุด
        timestamp (pd.Timestamp, optional): เวลาที่อัพเดต
        position (int): ตำแหน่งปัจจุบัน (0: ไม่มี, 1: Long, -1: Short)
        trade_return (float, optional): ผลตอบแทนจากการเทรดครั้งล่าสุด
        """
        # เพิ่มข้อมูลใหม่
        self.equity_curve.append(equity)
        
        if len(self.equity_curve) > 1:
            # คำนวณผลตอบแทน
            last_return = (equity / self.equity_curve[-2]) - 1
            self.returns.append(last_return)
        
        # บันทึกข้อมูลอื่นๆ
        self.positions.append(position)
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
        
        # บันทึกข้อมูลการเทรดหากมี
        if trade_return is not None:
            self.trade_returns.append(trade_return)
            self.trades_count += 1
            
            if trade_return > 0:
                self.winning_trades += 1
            elif trade_return < 0:
                self.losing_trades += 1
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        คำนวณสถิติประสิทธิภาพและความเสี่ยงทั้งหมด
        
        Parameters:
        risk_free_rate (float): อัตราผลตอบแทนที่ปราศจากความเสี่ยง (annualized)
        
        Returns:
        Dict[str, float]: สถิติต่างๆ
        """
        metrics = {}
        
        # สถิติผลตอบแทน
        if len(self.equity_curve) > 1:
            metrics['total_return'] = (self.equity_curve[-1] / self.equity_curve[0]) - 1
            metrics['cagr'] = calculate_cagr(self.equity_curve, self.periods_per_year)
        else:
            metrics['total_return'] = 0.0
            metrics['cagr'] = 0.0
        
        # สถิติความเสี่ยง
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
        
        # สถิติการเทรด
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
        """
        คำนวณและคืนค่าเมตริกความเสี่ยง
        
        Parameters:
        risk_free_rate (float): อัตราผลตอบแทนที่ปราศจากความเสี่ยง (annualized)
        
        Returns:
        RiskMetrics: เมตริกความเสี่ยงต่างๆ
        """
        if len(self.returns) < 2:
            return RiskMetrics()
        
        # สร้าง RiskMetrics
        metrics = RiskMetrics()
        
        # คำนวณเมตริกต่างๆ
        metrics.sharpe_ratio = calculate_sharpe_ratio(self.returns, risk_free_rate, self.periods_per_year)
        metrics.sortino_ratio = calculate_sortino_ratio(self.returns, risk_free_rate, self.periods_per_year)
        
        metrics.max_drawdown, peak_idx, trough_idx = calculate_max_drawdown(self.equity_curve)
        metrics.max_drawdown_duration = trough_idx - peak_idx
        
        metrics.volatility = calculate_volatility(self.returns, self.periods_per_year)
        
        # คำนวณ downside deviation
        downside_returns = np.minimum(np.asarray(self.returns), 0)
        metrics.downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.periods_per_year)
        
        # คำนวณ Value at Risk (VaR)
        if len(self.returns) > 0:
            metrics.var_95 = np.percentile(self.returns, 5)
            metrics.var_99 = np.percentile(self.returns, 1)
            
            # คำนวณ Conditional Value at Risk (CVaR)
            var_95_mask = np.asarray(self.returns) <= metrics.var_95
            metrics.cvar_95 = np.mean(np.asarray(self.returns)[var_95_mask]) if np.any(var_95_mask) else metrics.var_95
        
        # คำนวณ Calmar ratio
        if metrics.max_drawdown > 0:
            total_return = (self.equity_curve[-1] / self.equity_curve[0]) - 1
            years = len(self.equity_curve) / self.periods_per_year
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            metrics.calmar_ratio = annual_return / metrics.max_drawdown
        
        return metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        แปลงข้อมูลประสิทธิภาพเป็น DataFrame
        
        Returns:
        pd.DataFrame: DataFrame ของข้อมูลประสิทธิภาพ
        """
        data = {
            'equity': self.equity_curve[1:],  # ตัดค่าเริ่มต้นออก
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
        """
        รีเซ็ตข้อมูลประสิทธิภาพ
        
        Parameters:
        initial_equity (float, optional): เงินทุนเริ่มต้นใหม่
        """
        if initial_equity is not None:
            self.initial_equity = initial_equity
        
        # รีเซ็ตข้อมูลที่ติดตาม
        self.equity_curve = [self.initial_equity]
        self.returns = []
        self.trade_returns = []
        self.positions = []
        self.timestamps = []
        
        # รีเซ็ตสถิติการเทรด
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0