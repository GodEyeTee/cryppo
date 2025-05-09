import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """คลาสสำหรับเก็บข้อมูลการทำธุรกรรม"""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'short', 'cover'
    price: float
    units: float
    amount: float
    fee: float
    balance_before: float
    balance_after: float
    position_type: str
    
class TradingSimulator:
    """
    จำลองการเทรดในตลาด
    
    คลาสนี้รับผิดชอบการจำลองการเทรดและติดตามยอดคงเหลือ ตำแหน่ง และประสิทธิภาพ
    การจำลองมีความสมจริงด้วยการคำนวณค่าธรรมเนียม, slippage, และการบริหารเงินทุน
    """
    
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
        """
        กำหนดค่าเริ่มต้นสำหรับ TradingSimulator
        
        Parameters:
        initial_balance (float): เงินทุนเริ่มต้น
        transaction_fee (float): ค่าธรรมเนียมการทำธุรกรรม (เปอร์เซ็นต์)
        slippage (float): Slippage ในการทำธุรกรรม (เปอร์เซ็นต์)
        leverage (float): คันเร่งสำหรับ margin trading
        liquidation_threshold (float): เกณฑ์สำหรับการชำระเงิน (เปอร์เซ็นต์ของ margin)
        position_size (float or str): ขนาดของตำแหน่ง ('full' หรือเปอร์เซ็นต์)
        stop_loss (float, optional): ระดับ Stop Loss (เปอร์เซ็นต์)
        take_profit (float, optional): ระดับ Take Profit (เปอร์เซ็นต์)
        max_positions (int): จำนวนตำแหน่งสูงสุดที่สามารถเปิดได้
        config (Config, optional): อ็อบเจ็กต์การตั้งค่า
        """
        # โหลดการตั้งค่า
        self.config = config if config is not None else get_config()
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        env_config = self.config.extract_subconfig("environment")
        
        # กำหนดค่าพารามิเตอร์
        self.initial_balance = initial_balance or env_config.get("initial_balance", 10000.0)
        self.transaction_fee = transaction_fee or env_config.get("fee_rate", 0.0025)
        self.slippage = slippage or env_config.get("slippage", 0.0005)
        self.leverage = leverage or env_config.get("leverage", 1.0)
        self.liquidation_threshold = liquidation_threshold or env_config.get("liquidation_threshold", 0.8)
        self.position_size = position_size if position_size != 'auto' else env_config.get("position_size", 1.0)
        self.stop_loss = stop_loss or env_config.get("stop_loss", None)
        self.take_profit = take_profit or env_config.get("take_profit", None)
        self.max_positions = max_positions or env_config.get("max_positions", 1)
        
        # ตัวแปรสำหรับการติดตามสถานะของการจำลอง
        self.reset()
        
        logger.info(f"สร้าง TradingSimulator (balance={self.initial_balance}, fee={self.transaction_fee}, leverage={self.leverage})")
    
    def reset(self, initial_balance: Optional[float] = None) -> None:
        """
        รีเซ็ตสถานะของตัวจำลอง
        
        Parameters:
        initial_balance (float, optional): เงินทุนเริ่มต้นใหม่
        """
        # กำหนดยอดคงเหลือเริ่มต้น
        self.balance = initial_balance if initial_balance is not None else self.initial_balance
        
        # ตัวแปรเกี่ยวกับตำแหน่ง
        self.position_type = 'none'  # 'none', 'long', 'short'
        self.position_price = 0.0
        self.position_start_time = None
        self.units = 0.0
        self.margin = 0.0
        
        # ตัวแปรสำหรับการติดตามประสิทธิภาพ
        self.profit = 0.0
        self.total_profit = 0.0
        self.total_fee = 0.0
        self.num_trades = 0
        self.num_profitable_trades = 0
        self.num_losing_trades = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance if initial_balance is not None else self.initial_balance
        
        # ประวัติการทำธุรกรรม
        self.transactions = []
    
    def open_long_position(self, price: float, units: Optional[float] = None, timestamp: Optional[datetime] = None) -> bool:
        """
        เปิดตำแหน่ง Long
        
        Parameters:
        price (float): ราคาที่จะซื้อ
        units (float, optional): จำนวนหน่วยที่จะซื้อ
        timestamp (datetime, optional): เวลาที่ทำธุรกรรม
        
        Returns:
        bool: True ถ้าการทำธุรกรรมสำเร็จ, False ถ้าไม่สำเร็จ
        """
        # ตรวจสอบว่ามีตำแหน่งเปิดอยู่แล้วหรือไม่
        if self.position_type != 'none':
            logger.warning(f"มีตำแหน่ง {self.position_type} เปิดอยู่แล้ว, ไม่สามารถเปิดตำแหน่ง Long ได้")
            return False
        
        # คำนวณจำนวนหน่วยที่จะซื้อ
        if units is None:
            if isinstance(self.position_size, str) and self.position_size.lower() == 'full':
                # ใช้เงินทุนทั้งหมด
                available_balance = self.balance
            else:
                # ใช้เงินทุนตามเปอร์เซ็นต์
                available_balance = self.balance * float(self.position_size)
            
            # คำนวณค่าธรรมเนียมล่วงหน้า
            fee = available_balance * self.transaction_fee
            available_balance -= fee
            
            # คำนวณจำนวนหน่วย
            units = available_balance / price
        
        # คำนวณมูลค่าของตำแหน่ง
        position_value = price * units
        
        # คำนวณค่าธรรมเนียม
        fee = position_value * self.transaction_fee
        
        # คำนวณ slippage
        slippage_amount = position_value * self.slippage
        
        # ตรวจสอบว่ามีเงินทุนเพียงพอหรือไม่ (รวมค่าธรรมเนียมและ slippage)
        total_cost = position_value + fee + slippage_amount
        if total_cost > self.balance:
            logger.warning(f"เงินทุนไม่เพียงพอ: ต้องการ {total_cost}, มี {self.balance}")
            return False
        
        # บันทึกสถานะตำแหน่ง
        self.position_type = 'long'
        self.position_price = price
        self.position_start_time = timestamp if timestamp is not None else datetime.now()
        self.units = units
        self.margin = 0.0  # ไม่มี margin สำหรับตำแหน่ง Long
        
        # อัพเดตยอดคงเหลือ
        balance_before = self.balance
        self.balance -= total_cost
        
        # อัพเดตสถิติ
        self.total_fee += fee
        self.num_trades += 1
        
        # บันทึกการทำธุรกรรม
        transaction = Transaction(
            timestamp=self.position_start_time,
            action='buy',
            price=price,
            units=units,
            amount=position_value,
            fee=fee,
            balance_before=balance_before,
            balance_after=self.balance,
            position_type=self.position_type
        )
        self.transactions.append(transaction)
        
        logger.info(f"เปิดตำแหน่ง Long: {units} หน่วย @ {price} = {position_value} + ค่าธรรมเนียม {fee}")
        return True
    
    def open_short_position(self, price: float, units: Optional[float] = None, timestamp: Optional[datetime] = None) -> bool:
        """
        เปิดตำแหน่ง Short
        
        Parameters:
        price (float): ราคาที่จะขาย
        units (float, optional): จำนวนหน่วยที่จะขาย
        timestamp (datetime, optional): เวลาที่ทำธุรกรรม
        
        Returns:
        bool: True ถ้าการทำธุรกรรมสำเร็จ, False ถ้าไม่สำเร็จ
        """
        # ตรวจสอบว่ามีตำแหน่งเปิดอยู่แล้วหรือไม่
        if self.position_type != 'none':
            logger.warning(f"มีตำแหน่ง {self.position_type} เปิดอยู่แล้ว, ไม่สามารถเปิดตำแหน่ง Short ได้")
            return False
        
        # ตรวจสอบว่าเปิดใช้งาน leverage หรือไม่
        if self.leverage <= 1.0:
            logger.warning(f"ไม่สามารถเปิดตำแหน่ง Short ได้เนื่องจาก leverage = {self.leverage} (ต้อง > 1.0)")
            return False
        
        # คำนวณจำนวนหน่วยที่จะขาย
        if units is None:
            if isinstance(self.position_size, str) and self.position_size.lower() == 'full':
                # ใช้เงินทุนทั้งหมด
                available_balance = self.balance
            else:
                # ใช้เงินทุนตามเปอร์เซ็นต์
                available_balance = self.balance * float(self.position_size)
            
            # คำนวณ margin
            margin = available_balance
            
            # คำนวณมูลค่าของตำแหน่ง
            position_value = margin * self.leverage
            
            # คำนวณจำนวนหน่วย
            units = position_value / price
        else:
            # คำนวณมูลค่าของตำแหน่ง
            position_value = price * units
            
            # คำนวณ margin
            margin = position_value / self.leverage
        
        # คำนวณค่าธรรมเนียม
        fee = position_value * self.transaction_fee
        
        # คำนวณ slippage
        slippage_amount = position_value * self.slippage
        
        # ตรวจสอบว่ามีเงินทุนเพียงพอหรือไม่ (margin + ค่าธรรมเนียม + slippage)
        required_balance = margin + fee + slippage_amount
        if required_balance > self.balance:
            logger.warning(f"เงินทุนไม่เพียงพอ: ต้องการ {required_balance}, มี {self.balance}")
            return False
        
        # บันทึกสถานะตำแหน่ง
        self.position_type = 'short'
        self.position_price = price
        self.position_start_time = timestamp if timestamp is not None else datetime.now()
        self.units = units
        self.margin = margin
        
        # อัพเดตยอดคงเหลือ
        balance_before = self.balance
        self.balance -= (margin + fee + slippage_amount)
        
        # อัพเดตสถิติ
        self.total_fee += fee
        self.num_trades += 1
        
        # บันทึกการทำธุรกรรม
        transaction = Transaction(
            timestamp=self.position_start_time,
            action='short',
            price=price,
            units=units,
            amount=position_value,
            fee=fee,
            balance_before=balance_before,
            balance_after=self.balance,
            position_type=self.position_type
        )
        self.transactions.append(transaction)
        
        logger.info(f"เปิดตำแหน่ง Short: {units} หน่วย @ {price} = {position_value} (margin: {margin}) + ค่าธรรมเนียม {fee}")
        return True
    
    def close_position(self, price: float, timestamp: Optional[datetime] = None) -> bool:
        """
        ปิดตำแหน่งที่เปิดอยู่
        
        Parameters:
        price (float): ราคาที่จะปิดตำแหน่ง
        timestamp (datetime, optional): เวลาที่ทำธุรกรรม
        
        Returns:
        bool: True ถ้าการทำธุรกรรมสำเร็จ, False ถ้าไม่สำเร็จ
        """
        # ตรวจสอบว่ามีตำแหน่งเปิดอยู่หรือไม่
        if self.position_type == 'none':
            logger.warning("ไม่มีตำแหน่งที่เปิดอยู่, ไม่สามารถปิดตำแหน่งได้")
            return False
        
        # กำหนดเวลาทำธุรกรรม
        if timestamp is None:
            timestamp = datetime.now()
        
        # คำนวณมูลค่าของตำแหน่ง
        position_value = price * self.units
        
        # คำนวณค่าธรรมเนียม
        fee = position_value * self.transaction_fee
        
        # คำนวณ slippage
        slippage_amount = position_value * self.slippage
        
        # คำนวณกำไร/ขาดทุน
        if self.position_type == 'long':
            # Long: ซื้อที่ราคาต่ำ, ขายที่ราคาสูง
            profit = (price - self.position_price) * self.units - fee - slippage_amount
            action = 'sell'
        else:  # 'short'
            # Short: ขายที่ราคาสูง, ซื้อที่ราคาต่ำ
            profit = (self.position_price - price) * self.units - fee - slippage_amount
            action = 'cover'
        
        # อัพเดตยอดคงเหลือ
        balance_before = self.balance
        
        if self.position_type == 'long':
            # คืนเงินทุนที่ใช้ซื้อ + กำไร/ขาดทุน
            self.balance += (self.position_price * self.units) + profit
        else:  # 'short'
            # คืน margin + กำไร/ขาดทุน
            self.balance += self.margin + profit
        
        # บันทึกกำไร/ขาดทุน
        self.profit = profit
        self.total_profit += profit
        
        # อัพเดตสถิติ
        self.total_fee += fee
        if profit > 0:
            self.num_profitable_trades += 1
        else:
            self.num_losing_trades += 1
        
        # คำนวณผลตอบแทนรวม
        self.total_return = (self.balance / self.initial_balance - 1) * 100
        
        # คำนวณ Maximum Drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # บันทึกการทำธุรกรรม
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
        
        # รีเซ็ตสถานะตำแหน่ง
        old_position_type = self.position_type
        self.position_type = 'none'
        self.position_price = 0.0
        self.position_start_time = None
        self.units = 0.0
        self.margin = 0.0
        
        logger.info(f"ปิดตำแหน่ง {old_position_type}: {transaction.units} หน่วย @ {price} = {position_value} + ค่าธรรมเนียม {fee}, กำไร: {profit}")
        return True
    
    def update(self, price: float) -> bool:
        """
        อัพเดตสถานะของตัวจำลองตามราคาปัจจุบัน
        
        Parameters:
        price (float): ราคาปัจจุบัน
        
        Returns:
        bool: True ถ้ามีการปิดตำแหน่งโดยอัตโนมัติ (stop loss, take profit, liquidation), False ถ้าไม่มี
        """
        # ถ้าไม่มีตำแหน่งเปิดอยู่ ไม่ต้องทำอะไร
        if self.position_type == 'none':
            return False
        
        # คำนวณกำไร/ขาดทุนปัจจุบัน
        if self.position_type == 'long':
            current_profit_percent = (price / self.position_price - 1) * 100
        else:  # 'short'
            current_profit_percent = (self.position_price / price - 1) * 100
        
        # ตรวจสอบ Take Profit
        if self.take_profit is not None and current_profit_percent >= self.take_profit:
            logger.info(f"ทำกำไรที่ {current_profit_percent:.2f}% (take profit: {self.take_profit}%)")
            return self.close_position(price)
        
        # ตรวจสอบ Stop Loss
        if self.stop_loss is not None and current_profit_percent <= -self.stop_loss:
            logger.info(f"ขาดทุนที่ {current_profit_percent:.2f}% (stop loss: {self.stop_loss}%)")
            return self.close_position(price)
        
        # ตรวจสอบ Liquidation (สำหรับตำแหน่ง Short เท่านั้น)
        if self.position_type == 'short':
            # คำนวณมูลค่าของตำแหน่ง
            position_value = price * self.units
            
            # คำนวณ Margin Ratio
            margin_ratio = self.margin / position_value
            
            # ตรวจสอบ Liquidation
            if margin_ratio < self.liquidation_threshold:
                logger.warning(f"Liquidation: Margin Ratio = {margin_ratio:.2f} (threshold: {self.liquidation_threshold})")
                return self.close_position(price)
        
        return False
    
    def get_equity(self, price: float) -> float:
        """
        คำนวณมูลค่าทรัพย์สินรวม (เงินทุน + กำไร/ขาดทุนที่ยังไม่ได้รับรู้)
        
        Parameters:
        price (float): ราคาปัจจุบัน
        
        Returns:
        float: มูลค่าทรัพย์สินรวม
        """
        if self.position_type == 'none':
            return self.balance
        
        if self.position_type == 'long':
            unrealized_profit = (price - self.position_price) * self.units
        else:  # 'short'
            unrealized_profit = (self.position_price - price) * self.units
        
        return self.balance + unrealized_profit
    
    def has_position(self) -> bool:
        """
        ตรวจสอบว่ามีตำแหน่งเปิดอยู่หรือไม่
        
        Returns:
        bool: True ถ้ามีตำแหน่งเปิดอยู่, False ถ้าไม่มี
        """
        return self.position_type != 'none'
    
    def get_position_info(self) -> Dict[str, Any]:
        """
        ดึงข้อมูลตำแหน่ง
        
        Returns:
        Dict[str, Any]: ข้อมูลตำแหน่ง
        """
        return {
            'type': self.position_type,
            'price': self.position_price,
            'units': self.units,
            'margin': self.margin,
            'start_time': self.position_start_time
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        ดึงเมทริกการประสิทธิภาพ
        
        Returns:
        Dict[str, Any]: เมทริกการประสิทธิภาพ
        """
        # คำนวณ Win Rate
        win_rate = 0.0
        if self.num_trades > 0:
            win_rate = self.num_profitable_trades / self.num_trades
        
        # คำนวณ Average Profit per Trade
        avg_profit = 0.0
        if self.num_trades > 0:
            avg_profit = self.total_profit / self.num_trades
        
        # คำนวณ Profit Factor
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
        """
        แปลงสถานะของตัวจำลองเป็น dict
        
        Returns:
        Dict[str, Any]: สถานะของตัวจำลอง
        """
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
        """
        สร้างตัวจำลองจาก dict
        
        Parameters:
        state_dict (Dict[str, Any]): สถานะของตัวจำลอง
        
        Returns:
        TradingSimulator: อินสแตนซ์ของ TradingSimulator
        """
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