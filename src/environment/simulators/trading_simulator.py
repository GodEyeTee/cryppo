import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    timestamp: datetime
    action: str
    price: float
    units: float
    amount: float
    fee: float
    balance_before: float
    balance_after: float
    position_type: str
    
class TradingSimulator:
    def __init__(
        self,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.0025,
        slippage: float = 0.0005,
        leverage: float = 1.0,
        liquidation_threshold: float = 0.8,
        position_size: Union[float, str] = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        max_positions: int = 1,
        config = None
    ):
        self.config = config if config is not None else get_config()
        env_config = self.config.extract_subconfig("environment")
        
        self.initial_balance = initial_balance or env_config.get("initial_balance", 10000.0)
        self.transaction_fee = transaction_fee or env_config.get("fee_rate", 0.0025)
        self.slippage = slippage or env_config.get("slippage", 0.0005)
        self.leverage = leverage or env_config.get("leverage", 1.0)
        self.liquidation_threshold = liquidation_threshold or env_config.get("liquidation_threshold", 0.8)
        self.position_size = position_size if position_size != 'auto' else env_config.get("position_size", 1.0)
        self.stop_loss = stop_loss or env_config.get("stop_loss", None)
        self.take_profit = take_profit or env_config.get("take_profit", None)
        self.max_positions = max_positions or env_config.get("max_positions", 1)
        
        self.reset()
        logger.info(f"Created TradingSimulator (balance={self.initial_balance}, fee={self.transaction_fee}, leverage={self.leverage})")
    
    def reset(self, initial_balance: Optional[float] = None) -> None:
        self.balance = initial_balance if initial_balance is not None else self.initial_balance
        self.position_type = 'none'
        self.position_price = 0.0
        self.position_start_time = None
        self.units = 0.0
        self.margin = 0.0
        
        self.profit = 0.0
        self.total_profit = 0.0
        self.total_fee = 0.0
        self.num_trades = 0
        self.num_profitable_trades = 0
        self.num_losing_trades = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance if initial_balance is not None else self.initial_balance
        self.transactions = []
    
    def open_long_position(self, price: float, units: Optional[float] = None, timestamp: Optional[datetime] = None) -> bool:
        if self.position_type != 'none':
            logger.warning(f"Position {self.position_type} already open, cannot open Long position")
            return False
        
        if units is None:
            # Use fraction of balance for position value including fees and slippage
            if isinstance(self.position_size, str) and self.position_size.lower() == 'full':
                budget = self.balance
            else:
                budget = self.balance * float(self.position_size)
            total_multiplier = 1.0 + float(self.transaction_fee) + float(self.slippage)
            position_value = budget / total_multiplier
            units = position_value / price
        else:
            position_value = price * units
        
        fee = position_value * self.transaction_fee
        slippage_amount = position_value * self.slippage
        total_cost = position_value + fee + slippage_amount
        
        if total_cost > self.balance:
            logger.warning(f"Insufficient funds: need {total_cost}, have {self.balance}")
            return False
        
        self.position_type = 'long'
        self.position_price = price
        self.position_start_time = timestamp if timestamp is not None else datetime.now()
        self.units = units
        self.margin = 0.0
        
        balance_before = self.balance
        self.balance -= total_cost
        self.total_fee += fee
        self.num_trades += 1
        
        self.transactions.append(Transaction(
            timestamp=self.position_start_time,
            action='buy',
            price=price,
            units=units,
            amount=position_value,
            fee=fee,
            balance_before=balance_before,
            balance_after=self.balance,
            position_type=self.position_type
        ))
        
        logger.info(f"Opened Long position: {units} units @ {price} = {position_value} + fee {fee}")
        return True
    
    def open_short_position(self, price: float, units: Optional[float] = None, timestamp: Optional[datetime] = None) -> bool:
        if self.position_type != 'none':
            logger.warning(f"Position {self.position_type} already open, cannot open Short position")
            return False
        
        if self.leverage <= 1.0:
            logger.warning(f"Cannot open Short position because leverage = {self.leverage} (must be > 1.0)")
            return False
        
        if units is None:
            # Use fraction of balance as total cash outlay (margin + fees + slippage)
            if isinstance(self.position_size, str) and self.position_size.lower() == 'full':
                budget = self.balance
            else:
                budget = self.balance * float(self.position_size)
            # required_balance = margin + fee + slippage, where fee/slippage are on position_value
            # position_value = margin * leverage
            # required_balance = margin + (margin*leverage)*(fee+slip)
            # Solve margin such that required_balance <= budget
            fee_slip = float(self.transaction_fee) + float(self.slippage)
            denom = 1.0 + self.leverage * fee_slip
            margin = budget / denom
            position_value = margin * self.leverage
            units = position_value / price
        else:
            position_value = price * units
            margin = position_value / self.leverage
        
        fee = position_value * self.transaction_fee
        slippage_amount = position_value * self.slippage
        required_balance = margin + fee + slippage_amount
        
        if required_balance > self.balance:
            logger.warning(f"Insufficient funds: need {required_balance}, have {self.balance}")
            return False
        
        self.position_type = 'short'
        self.position_price = price
        self.position_start_time = timestamp if timestamp is not None else datetime.now()
        self.units = units
        self.margin = margin
        
        balance_before = self.balance
        self.balance -= (margin + fee + slippage_amount)
        self.total_fee += fee
        self.num_trades += 1
        
        self.transactions.append(Transaction(
            timestamp=self.position_start_time,
            action='short',
            price=price,
            units=units,
            amount=position_value,
            fee=fee,
            balance_before=balance_before,
            balance_after=self.balance,
            position_type=self.position_type
        ))
        
        logger.info(f"Opened Short position: {units} units @ {price} = {position_value} (margin: {margin}) + fee {fee}")
        return True
    
    def close_position(self, price: float, timestamp: Optional[datetime] = None) -> bool:
        if self.position_type == 'none':
            logger.warning("No open position to close")
            return False
        
        timestamp = timestamp or datetime.now()
        position_value = price * self.units
        fee = position_value * self.transaction_fee
        slippage_amount = position_value * self.slippage
        
        if self.position_type == 'long':
            profit = (price - self.position_price) * self.units - fee - slippage_amount
            action = 'sell'
        else:
            profit = (self.position_price - price) * self.units - fee - slippage_amount
            action = 'cover'
        
        balance_before = self.balance
        
        if self.position_type == 'long':
            self.balance += (self.position_price * self.units) + profit
        else:
            self.balance += self.margin + profit
        
        self.profit = profit
        self.total_profit += profit
        self.total_fee += fee
        
        if profit > 0:
            self.num_profitable_trades += 1
        else:
            self.num_losing_trades += 1
        
        self.total_return = (self.balance / self.initial_balance - 1) * 100
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        transaction = Transaction(
            timestamp=timestamp,
            action=action,
            price=price,
            units=self.units,
            amount=position_value,
            fee=fee,
            balance_before=balance_before,
            balance_after=self.balance,
            position_type=self.position_type
        )
        self.transactions.append(transaction)
        
        old_position_type = self.position_type
        self.position_type = 'none'
        self.position_price = 0.0
        self.position_start_time = None
        self.units = 0.0
        self.margin = 0.0
        
        logger.info(f"Closed {old_position_type} position: {transaction.units} units @ {price} = {position_value} + fee {fee}, profit: {profit}")
        return True
    
    def update(self, price: float) -> bool:
        if self.position_type == 'none':
            return False
        
        current_profit_percent = (price / self.position_price - 1) * 100 if self.position_type == 'long' else (self.position_price / price - 1) * 100
        
        if self.take_profit is not None and current_profit_percent >= self.take_profit:
            logger.info(f"Taking profit at {current_profit_percent:.2f}% (take profit: {self.take_profit}%)")
            return self.close_position(price)
        
        if self.stop_loss is not None and current_profit_percent <= -self.stop_loss:
            logger.info(f"Stopping loss at {current_profit_percent:.2f}% (stop loss: {self.stop_loss}%)")
            return self.close_position(price)
        
        if self.position_type == 'short':
            position_value = price * self.units
            margin_ratio = self.margin / position_value
            
            if margin_ratio < self.liquidation_threshold:
                logger.warning(f"Liquidation: Margin Ratio = {margin_ratio:.2f} (threshold: {self.liquidation_threshold})")
                return self.close_position(price)
        
        return False
    
    def get_equity(self, price: float) -> float:
        if self.position_type == 'none':
            return self.balance
        
        unrealized_profit = (price - self.position_price) * self.units if self.position_type == 'long' else (self.position_price - price) * self.units
        return self.balance + unrealized_profit
    
    def has_position(self) -> bool:
        return self.position_type != 'none'
    
    def get_position_info(self) -> Dict[str, Any]:
        return {
            'type': self.position_type,
            'price': self.position_price,
            'units': self.units,
            'margin': self.margin,
            'start_time': self.position_start_time
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        win_rate = self.num_profitable_trades / self.num_trades if self.num_trades > 0 else 0.0
        avg_profit = self.total_profit / self.num_trades if self.num_trades > 0 else 0.0
        
        profit_factor = 0.0
        if self.num_profitable_trades > 0 and self.num_losing_trades > 0:
            total_winning = self.total_profit + self.total_fee
            total_losing = self.total_fee
            profit_factor = total_winning / total_losing if total_losing > 0 else float('inf')
        
        return {
            'balance': self.balance,
            'profit': self.total_profit,
            'return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'profit_factor': profit_factor,
            'total_fee': self.total_fee
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'balance': self.balance,
            'position_type': self.position_type,
            'position_price': self.position_price,
            'position_start_time': self.position_start_time,
            'units': self.units,
            'margin': self.margin,
            'profit': self.profit,
            'total_profit': self.total_profit,
            'total_fee': self.total_fee,
            'num_trades': self.num_trades,
            'num_profitable_trades': self.num_profitable_trades,
            'num_losing_trades': self.num_losing_trades,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'peak_balance': self.peak_balance
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'TradingSimulator':
        simulator = cls(initial_balance=state_dict.get('balance', 10000.0))
        
        simulator.balance = state_dict.get('balance', 10000.0)
        simulator.position_type = state_dict.get('position_type', 'none')
        simulator.position_price = state_dict.get('position_price', 0.0)
        simulator.position_start_time = state_dict.get('position_start_time', None)
        simulator.units = state_dict.get('units', 0.0)
        simulator.margin = state_dict.get('margin', 0.0)
        simulator.profit = state_dict.get('profit', 0.0)
        simulator.total_profit = state_dict.get('total_profit', 0.0)
        simulator.total_fee = state_dict.get('total_fee', 0.0)
        simulator.num_trades = state_dict.get('num_trades', 0)
        simulator.num_profitable_trades = state_dict.get('num_profitable_trades', 0)
        simulator.num_losing_trades = state_dict.get('num_losing_trades', 0)
        simulator.total_return = state_dict.get('total_return', 0.0)
        simulator.max_drawdown = state_dict.get('max_drawdown', 0.0)
        simulator.peak_balance = state_dict.get('peak_balance', simulator.balance)
        
        return simulator
