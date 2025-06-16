"""
Elliott Wave Detection System with Market Structure Analysis
Enhanced version with proper swing detection based on HH, HL, LL, LH principles
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== Data Structures ====================

@dataclass
class SwingPoint:
    """Represents a significant swing high or low with market structure"""
    index: int
    price: float
    time: pd.Timestamp
    type: str  # 'high' or 'low'
    structure: str = None  # 'HH', 'HL', 'LL', 'LH'
    significance: float = 0.0  # How significant this swing is (0-1)
    
@dataclass
class Wave:
    """Represents an Elliott Wave"""
    start: SwingPoint
    end: SwingPoint
    wave_number: Union[int, str]  # 1-5 for impulse; A,B,C,W,X,Y,Z for corrective
    wave_type: str  # 'impulse' or 'corrective'
    degree: str
    confidence: float = 0.0
    subwaves: List['Wave'] = None
    
    def __post_init__(self):
        if self.subwaves is None:
            self.subwaves = []
    
    @property
    def length(self):
        return abs(self.end.price - self.start.price)
    
    @property
    def percentage_move(self):
        return ((self.end.price - self.start.price) / self.start.price) * 100
    
    @property
    def is_bullish(self):
        return self.end.price > self.start.price

@dataclass
class ElliottPattern:
    """Represents a complete Elliott Wave pattern"""
    waves: List[Wave]
    pattern_type: str  # 'impulse', 'zigzag', 'flat', 'triangle', 'complex', etc.
    degree: str
    confidence: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp

# ==================== Market Structure Analysis ====================

class MarketStructure:
    """Analyze market structure using HH, HL, LL, LH concepts"""
    
    @staticmethod
    def find_major_swings(data: pd.DataFrame, min_percent_move: float = 3.0, 
                         lookback: int = 20) -> List[SwingPoint]:
        """
        Find major swing points using market structure concepts
        
        Parameters:
        - min_percent_move: Minimum % move to qualify as a swing
        - lookback: Number of bars to look back for pivot confirmation
        """
        swings = []
        
        # First, find all potential pivots
        potential_highs = []
        potential_lows = []
        
        for i in range(lookback, len(data) - lookback):
            # Check for swing high
            is_high = True
            for j in range(1, lookback + 1):
                if data['High'].iloc[i] < data['High'].iloc[i-j] or \
                   data['High'].iloc[i] < data['High'].iloc[i+j]:
                    is_high = False
                    break
            
            if is_high:
                potential_highs.append(SwingPoint(
                    index=i,
                    price=data['High'].iloc[i],
                    time=data.index[i],
                    type='high'
                ))
            
            # Check for swing low
            is_low = True
            for j in range(1, lookback + 1):
                if data['Low'].iloc[i] > data['Low'].iloc[i-j] or \
                   data['Low'].iloc[i] > data['Low'].iloc[i+j]:
                    is_low = False
                    break
            
            if is_low:
                potential_lows.append(SwingPoint(
                    index=i,
                    price=data['Low'].iloc[i],
                    time=data.index[i],
                    type='low'
                ))
        
        # Filter swings by significance
        all_swings = potential_highs + potential_lows
        all_swings.sort(key=lambda x: x.index)
        
        if not all_swings:
            return []
        
        # Start with first swing
        filtered_swings = [all_swings[0]]
        
        # Filter based on minimum percentage move and alternation
        for swing in all_swings[1:]:
            last_swing = filtered_swings[-1]
            
            # Calculate percentage move
            percent_move = abs((swing.price - last_swing.price) / last_swing.price) * 100
            
            # Check if this swing is significant enough
            if percent_move >= min_percent_move:
                # If same type as last swing, only keep if more extreme
                if swing.type == last_swing.type:
                    if (swing.type == 'high' and swing.price > last_swing.price) or \
                       (swing.type == 'low' and swing.price < last_swing.price):
                        filtered_swings[-1] = swing  # Replace with more extreme swing
                else:
                    # Different type, add it
                    filtered_swings.append(swing)
                    swing.significance = min(percent_move / 10, 1.0)  # Normalize to 0-1
        
        # Label market structure
        MarketStructure._label_structure(filtered_swings)
        
        return filtered_swings
    
    @staticmethod
    def _label_structure(swings: List[SwingPoint]):
        """Label swings with market structure (HH, HL, LL, LH)"""
        if len(swings) < 3:
            return
        
        for i in range(2, len(swings)):
            current = swings[i]
            
            if current.type == 'high':
                # Compare with previous high
                prev_high = None
                for j in range(i-1, -1, -1):
                    if swings[j].type == 'high':
                        prev_high = swings[j]
                        break
                
                if prev_high:
                    if current.price > prev_high.price:
                        current.structure = 'HH'  # Higher High
                    else:
                        current.structure = 'LH'  # Lower High
            
            else:  # low
                # Compare with previous low
                prev_low = None
                for j in range(i-1, -1, -1):
                    if swings[j].type == 'low':
                        prev_low = swings[j]
                        break
                
                if prev_low:
                    if current.price > prev_low.price:
                        current.structure = 'HL'  # Higher Low
                    else:
                        current.structure = 'LL'  # Lower Low
    
    @staticmethod
    def identify_trend_changes(swings: List[SwingPoint]) -> List[int]:
        """Identify where trend changes occur based on structure breaks"""
        trend_changes = []
        
        for i in range(3, len(swings)):
            # Bullish structure break: LL followed by HH
            if (swings[i-1].structure == 'LL' and swings[i].structure == 'HH') or \
               (swings[i-1].structure == 'LH' and swings[i].structure == 'HL'):
                trend_changes.append(i)
            
            # Bearish structure break: HH followed by LL
            elif (swings[i-1].structure == 'HH' and swings[i].structure == 'LL') or \
                 (swings[i-1].structure == 'HL' and swings[i].structure == 'LH'):
                trend_changes.append(i)
        
        return trend_changes

# ==================== Elliott Wave Rules ====================

class ElliottRules:
    """Elliott Wave rules and guidelines with proper validation"""
    
    @staticmethod
    def validate_impulse(waves: List[Wave]) -> Tuple[bool, float, List[str]]:
        """Validate impulse wave with confidence score"""
        if len(waves) != 5:
            return False, 0.0, ["Impulse must have exactly 5 waves"]
        
        violations = []
        confidence = 1.0
        
        # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        wave2_retrace = abs(waves[1].length / waves[0].length)
        if wave2_retrace > 1.0:
            violations.append(f"Wave 2 retraced {wave2_retrace:.1%} of Wave 1")
            return False, 0.0, violations
        
        # Ideal retracement for Wave 2 is 50-61.8%
        if 0.5 <= wave2_retrace <= 0.618:
            confidence *= 1.2
        elif 0.618 < wave2_retrace <= 0.786:
            confidence *= 1.1
        
        # Rule 2: Wave 3 cannot be the shortest
        lengths = [waves[0].length, waves[2].length, waves[4].length]
        if waves[2].length == min(lengths):
            violations.append("Wave 3 is the shortest")
            return False, 0.0, violations
        
        # Wave 3 is often the longest
        if waves[2].length == max(lengths):
            confidence *= 1.2
        
        # Wave 3 extension check
        wave3_extend = waves[2].length / waves[0].length
        if 1.5 <= wave3_extend <= 1.7:  # Near 1.618
            confidence *= 1.3
        elif 2.5 <= wave3_extend <= 2.7:  # Near 2.618
            confidence *= 1.2
        
        # Rule 3: Wave 4 cannot overlap Wave 1 (except in diagonals)
        if waves[0].is_bullish:
            if waves[3].end.price < waves[0].end.price:
                violations.append("Wave 4 overlaps Wave 1")
                confidence *= 0.3  # Major violation but not impossible (could be diagonal)
        else:
            if waves[3].end.price > waves[0].end.price:
                violations.append("Wave 4 overlaps Wave 1")
                confidence *= 0.3
        
        # Wave 4 retracement check
        wave4_retrace = abs(waves[3].length / waves[2].length)
        if 0.35 <= wave4_retrace <= 0.4:  # Near 0.382
            confidence *= 1.1
        
        # Wave 5 relationship
        wave5_to_1 = waves[4].length / waves[0].length
        if 0.9 <= wave5_to_1 <= 1.1:  # Equal to Wave 1
            confidence *= 1.1
        elif 0.55 <= wave5_to_1 <= 0.65:  # 0.618 of Wave 1
            confidence *= 1.05
        
        # Alternation guideline
        if waves[1].percentage_move * waves[3].percentage_move < 0:  # Different depths
            confidence *= 1.1
        
        # Normalize confidence
        confidence = min(confidence, 0.95)
        confidence = max(confidence, 0.1)
        
        return len(violations) == 0, confidence, violations
    
    @staticmethod
    def validate_corrective(waves: List[Wave], pattern_type: str) -> Tuple[bool, float]:
        """Validate corrective patterns"""
        confidence = 0.6  # Base confidence for corrective
        
        if pattern_type == 'zigzag' and len(waves) >= 3:
            # Zigzag: 5-3-5 structure
            # Wave B should retrace 50-61.8% of Wave A
            b_retrace = abs(waves[1].length / waves[0].length)
            if 0.5 <= b_retrace <= 0.618:
                confidence += 0.2
            
            # Wave C often equals Wave A or 1.618 * A
            c_to_a = waves[2].length / waves[0].length
            if 0.9 <= c_to_a <= 1.1 or 1.5 <= c_to_a <= 1.7:
                confidence += 0.1
        
        elif pattern_type == 'flat' and len(waves) >= 3:
            # Flat: 3-3-5 structure
            # Wave B retraces 90%+ of Wave A
            b_retrace = abs(waves[1].length / waves[0].length)
            if b_retrace >= 0.9:
                confidence += 0.2
                if b_retrace >= 1.05:  # Expanded flat
                    confidence += 0.1
        
        elif pattern_type == 'triangle' and len(waves) >= 5:
            # Triangle: contracting pattern
            contracting = all(waves[i].length < waves[i-1].length for i in range(1, len(waves)))
            if contracting:
                confidence += 0.2
        
        elif pattern_type == 'complex' and len(waves) >= 3:
            # Complex correction (W-X-Y, etc.)
            confidence += 0.1  # Base bonus for identifying complex pattern
        
        return True, min(confidence, 0.9)

# ==================== Pattern Detection ====================

class ElliottPatternDetector:
    """Detect Elliott Wave patterns using market structure"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.market_structure = MarketStructure()
        self.rules = ElliottRules()
    
    def detect_all_patterns(self, swings: List[SwingPoint]) -> List[ElliottPattern]:
        """Detect all possible Elliott Wave patterns"""
        patterns = []
        
        # Try to find impulse waves
        patterns.extend(self._detect_impulse_patterns(swings))
        
        # Try to find corrective patterns
        patterns.extend(self._detect_corrective_patterns(swings))
        
        # Try to find complex patterns
        patterns.extend(self._detect_complex_patterns(swings))
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove overlapping patterns
        patterns = self._filter_overlapping_patterns(patterns)
        
        return patterns
    
    def _detect_impulse_patterns(self, swings: List[SwingPoint]) -> List[ElliottPattern]:
        """Detect 5-wave impulse patterns"""
        patterns = []
        
        # Need at least 6 points for 5 waves
        if len(swings) < 6:
            return patterns
        
        # Try different starting points
        for i in range(len(swings) - 5):
            # Try different window sizes (6-10 points)
            for window in range(6, min(11, len(swings) - i + 1)):
                test_swings = swings[i:i + window]
                
                # Try to extract 5 waves from these swings
                wave_combinations = self._get_wave_combinations(test_swings, 5)
                
                for combo in wave_combinations:
                    waves = self._create_waves(combo, 'impulse')
                    
                    # Validate the impulse
                    valid, confidence, violations = self.rules.validate_impulse(waves)
                    
                    if confidence > 0.5:  # Minimum threshold
                        pattern = ElliottPattern(
                            waves=waves,
                            pattern_type='impulse',
                            degree=self._determine_degree(waves),
                            confidence=confidence,
                            start_time=waves[0].start.time,
                            end_time=waves[-1].end.time
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_corrective_patterns(self, swings: List[SwingPoint]) -> List[ElliottPattern]:
        """Detect corrective patterns (zigzag, flat, triangle)"""
        patterns = []
        
        # Zigzag (3 waves minimum)
        if len(swings) >= 4:
            for i in range(len(swings) - 3):
                test_swings = swings[i:i + 4]
                waves = self._create_waves_simple(test_swings[:4], ['A', 'B', 'C'])
                
                # Check zigzag characteristics
                if self._is_zigzag_pattern(waves):
                    valid, confidence = self.rules.validate_corrective(waves, 'zigzag')
                    if confidence > 0.5:
                        pattern = ElliottPattern(
                            waves=waves,
                            pattern_type='zigzag',
                            degree=self._determine_degree(waves),
                            confidence=confidence,
                            start_time=waves[0].start.time,
                            end_time=waves[-1].end.time
                        )
                        patterns.append(pattern)
        
        # Flat patterns
        if len(swings) >= 4:
            for i in range(len(swings) - 3):
                test_swings = swings[i:i + 4]
                waves = self._create_waves_simple(test_swings[:4], ['A', 'B', 'C'])
                
                # Check flat characteristics
                if self._is_flat_pattern(waves):
                    valid, confidence = self.rules.validate_corrective(waves, 'flat')
                    if confidence > 0.5:
                        pattern_type = 'expanded_flat' if waves[1].length > waves[0].length else 'flat'
                        pattern = ElliottPattern(
                            waves=waves,
                            pattern_type=pattern_type,
                            degree=self._determine_degree(waves),
                            confidence=confidence,
                            start_time=waves[0].start.time,
                            end_time=waves[-1].end.time
                        )
                        patterns.append(pattern)
        
        # Triangle patterns (5 waves)
        if len(swings) >= 6:
            for i in range(len(swings) - 5):
                test_swings = swings[i:i + 6]
                waves = self._create_waves_simple(test_swings[:6], ['A', 'B', 'C', 'D', 'E'])
                
                if self._is_triangle_pattern(waves):
                    valid, confidence = self.rules.validate_corrective(waves, 'triangle')
                    if confidence > 0.5:
                        pattern = ElliottPattern(
                            waves=waves,
                            pattern_type='triangle',
                            degree=self._determine_degree(waves),
                            confidence=confidence,
                            start_time=waves[0].start.time,
                            end_time=waves[-1].end.time
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_complex_patterns(self, swings: List[SwingPoint]) -> List[ElliottPattern]:
        """Detect complex corrective patterns (W-X-Y, W-X-Y-X-Z)"""
        patterns = []
        
        # W-X-Y pattern (minimum 7 swings)
        if len(swings) >= 7:
            for i in range(len(swings) - 6):
                # Look for double three pattern
                test_swings = swings[i:i + 7]
                
                # Try to identify W-X-Y structure
                if self._could_be_wxy(test_swings):
                    waves = self._create_complex_waves(test_swings, ['W', 'X', 'Y'])
                    if waves and len(waves) >= 3:
                        valid, confidence = self.rules.validate_corrective(waves, 'complex')
                        if confidence > 0.5:
                            pattern = ElliottPattern(
                                waves=waves,
                                pattern_type='WXY',
                                degree=self._determine_degree(waves),
                                confidence=confidence,
                                start_time=waves[0].start.time,
                                end_time=waves[-1].end.time
                            )
                            patterns.append(pattern)
        
        # W-X-Y-X-Z pattern (minimum 11 swings)
        if len(swings) >= 11:
            for i in range(len(swings) - 10):
                test_swings = swings[i:i + 11]
                
                if self._could_be_wxyxz(test_swings):
                    waves = self._create_complex_waves(test_swings, ['W', 'X', 'Y', 'X2', 'Z'])
                    if waves and len(waves) >= 5:
                        valid, confidence = self.rules.validate_corrective(waves, 'complex')
                        if confidence > 0.5:
                            pattern = ElliottPattern(
                                waves=waves,
                                pattern_type='WXYXZ',
                                degree=self._determine_degree(waves),
                                confidence=confidence * 0.9,  # Slightly lower for complex
                                start_time=waves[0].start.time,
                                end_time=waves[-1].end.time
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _get_wave_combinations(self, swings: List[SwingPoint], num_waves: int) -> List[List[SwingPoint]]:
        """Get possible combinations of swings that could form waves"""
        combinations = []
        
        # For 5 waves, we need 6 points
        if num_waves == 5 and len(swings) >= 6:
            # Simple case: consecutive points
            if len(swings) == 6:
                combinations.append(swings)
            else:
                # Try skipping some minor swings
                for skip_idx in range(1, len(swings) - 5):
                    combo = swings[:skip_idx] + swings[skip_idx + 1:]
                    if len(combo) >= 6:
                        combinations.append(combo[:6])
        
        return combinations
    
    def _create_waves(self, swings: List[SwingPoint], wave_type: str) -> List[Wave]:
        """Create wave objects from swing points"""
        waves = []
        
        for i in range(len(swings) - 1):
            wave_num = i + 1 if wave_type == 'impulse' else chr(65 + i)  # A, B, C...
            
            wave = Wave(
                start=swings[i],
                end=swings[i + 1],
                wave_number=wave_num,
                wave_type='impulse' if i % 2 == 0 else 'corrective',
                degree='Minor'
            )
            waves.append(wave)
        
        return waves
    
    def _create_waves_simple(self, swings: List[SwingPoint], labels: List[str]) -> List[Wave]:
        """Create waves with specific labels"""
        waves = []
        
        for i in range(min(len(labels), len(swings) - 1)):
            wave = Wave(
                start=swings[i],
                end=swings[i + 1],
                wave_number=labels[i],
                wave_type='corrective',
                degree='Minor'
            )
            waves.append(wave)
        
        return waves
    
    def _create_complex_waves(self, swings: List[SwingPoint], labels: List[str]) -> List[Wave]:
        """Create waves for complex patterns"""
        waves = []
        points_per_wave = len(swings) // len(labels)
        
        for i, label in enumerate(labels):
            start_idx = i * points_per_wave
            end_idx = min(start_idx + points_per_wave, len(swings) - 1)
            
            if end_idx < len(swings):
                wave = Wave(
                    start=swings[start_idx],
                    end=swings[end_idx],
                    wave_number=label,
                    wave_type='corrective',
                    degree='Minor'
                )
                waves.append(wave)
        
        return waves
    
    def _is_zigzag_pattern(self, waves: List[Wave]) -> bool:
        """Check if waves form a zigzag pattern"""
        if len(waves) < 3:
            return False
        
        # Wave B should retrace 50-78.6% of Wave A
        b_retrace = abs(waves[1].length / waves[0].length)
        return 0.3 <= b_retrace <= 0.8
    
    def _is_flat_pattern(self, waves: List[Wave]) -> bool:
        """Check if waves form a flat pattern"""
        if len(waves) < 3:
            return False
        
        # Wave B should retrace 80%+ of Wave A
        b_retrace = abs(waves[1].length / waves[0].length)
        return b_retrace >= 0.8
    
    def _is_triangle_pattern(self, waves: List[Wave]) -> bool:
        """Check if waves form a triangle pattern"""
        if len(waves) < 5:
            return False
        
        # Check for contracting pattern
        for i in range(1, len(waves)):
            if waves[i].length > waves[i-1].length * 1.2:
                return False
        
        return True
    
    def _could_be_wxy(self, swings: List[SwingPoint]) -> bool:
        """Check if swings could form W-X-Y pattern"""
        if len(swings) < 7:
            return False
        
        # Basic check: should have alternating larger moves
        return True  # Simplified for now
    
    def _could_be_wxyxz(self, swings: List[SwingPoint]) -> bool:
        """Check if swings could form W-X-Y-X-Z pattern"""
        if len(swings) < 11:
            return False
        
        # Basic check
        return True  # Simplified for now
    
    def _determine_degree(self, waves: List[Wave]) -> str:
        """Determine the degree of the wave pattern"""
        if not waves:
            return 'Minor'
        
        # Calculate total price move
        total_move = abs(waves[-1].end.price - waves[0].start.price)
        percent_move = (total_move / waves[0].start.price) * 100
        
        # Determine degree based on percentage move
        if percent_move > 100:
            return 'Cycle'
        elif percent_move > 50:
            return 'Primary'
        elif percent_move > 20:
            return 'Intermediate'
        elif percent_move > 10:
            return 'Minor'
        else:
            return 'Minute'
    
    def _filter_overlapping_patterns(self, patterns: List[ElliottPattern]) -> List[ElliottPattern]:
        """Remove overlapping patterns, keeping highest confidence"""
        if len(patterns) <= 1:
            return patterns
        
        filtered = []
        
        for pattern in patterns:
            overlap = False
            
            for existing in filtered:
                # Check time overlap
                start_overlap = max(pattern.start_time, existing.start_time)
                end_overlap = min(pattern.end_time, existing.end_time)
                
                if start_overlap < end_overlap:
                    # Calculate overlap percentage
                    pattern_duration = (pattern.end_time - pattern.start_time).total_seconds()
                    overlap_duration = (end_overlap - start_overlap).total_seconds()
                    
                    if overlap_duration / pattern_duration > 0.5:
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(pattern)
        
        return filtered

# ==================== Main Analyzer ====================

class ElliottWaveAnalyzer:
    """Main class for Elliott Wave analysis with market structure"""
    
    def __init__(self, symbol: str, period: str = '3mo', interval: str = '4h'):
        self.symbol = self._format_symbol(symbol)
        self.period = period
        self.interval = interval
        self.data = None
        self.swings = []
        self.patterns = []
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for yfinance"""
        # Crypto symbols
        if symbol.upper() in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE']:
            return f"{symbol}-USD"
        return symbol
    
    def download_data(self) -> pd.DataFrame:
        """Download historical data"""
        try:
            print(f"Downloading data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval=self.interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Add technical indicators
            self._add_indicators()
            
            print(f"Downloaded {len(self.data)} data points")
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    
    def _add_indicators(self):
        """Add technical indicators"""
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.data['EMA12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = self.data['EMA12'] - self.data['EMA26']
        self.data['Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['Signal']
        
        # Volume MA
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
    
    def find_patterns(self, min_swing_percent: float = 3.0, lookback: int = 10) -> List[ElliottPattern]:
        """Find Elliott Wave patterns using market structure"""
        if self.data is None:
            print("No data available. Please download first.")
            return []
        
        # Find major swings using market structure
        ms = MarketStructure()
        self.swings = ms.find_major_swings(self.data, min_swing_percent, lookback)
        
        print(f"\nFound {len(self.swings)} major swings:")
        
        # Show swing structure
        structure_counts = {'HH': 0, 'HL': 0, 'LL': 0, 'LH': 0}
        for swing in self.swings:
            if swing.structure:
                structure_counts[swing.structure] += 1
        
        print(f"Market Structure: HH={structure_counts['HH']}, HL={structure_counts['HL']}, "
              f"LL={structure_counts['LL']}, LH={structure_counts['LH']}")
        
        # Detect patterns
        detector = ElliottPatternDetector(self.data)
        self.patterns = detector.detect_all_patterns(self.swings)
        
        print(f"\nFound {len(self.patterns)} Elliott Wave patterns")
        
        # Show pattern breakdown
        pattern_types = {}
        for p in self.patterns:
            pattern_types[p.pattern_type] = pattern_types.get(p.pattern_type, 0) + 1
        
        if pattern_types:
            print("Pattern breakdown:")
            for ptype, count in sorted(pattern_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {ptype}: {count}")
        
        return self.patterns
    
    def plot_analysis(self, show_all_patterns: bool = False):
        """Plot the analysis with Elliott Wave patterns"""
        if self.data is None:
            print("No data to plot")
            return
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        
        # Main chart
        ax1 = fig.add_subplot(gs[0])
        
        # Plot candlesticks
        for idx, (index, row) in enumerate(self.data.iterrows()):
            color = 'green' if row['Close'] > row['Open'] else 'red'
            
            # Body
            height = abs(row['Close'] - row['Open'])
            bottom = min(row['Close'], row['Open'])
            rect = plt.Rectangle((idx - 0.3, bottom), 0.6, height,
                               facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
            
            # Wicks
            ax1.plot([idx, idx], [row['Low'], row['High']], 
                    color='black', linewidth=0.5)
        
        # Plot major swings with structure labels
        for i, swing in enumerate(self.swings):
            # Plot the swing point
            x_pos = swing.index
            if swing.type == 'high':
                ax1.plot(x_pos, swing.price, 'v', color='red', 
                        markersize=10, markeredgecolor='black', markeredgewidth=1)
                # Add structure label
                if swing.structure:
                    ax1.text(x_pos, swing.price * 1.01, swing.structure,
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', 
                                    alpha=0.7, edgecolor='black'))
            else:
                ax1.plot(x_pos, swing.price, '^', color='green',
                        markersize=10, markeredgecolor='black', markeredgewidth=1)
                # Add structure label
                if swing.structure:
                    ax1.text(x_pos, swing.price * 0.99, swing.structure,
                            ha='center', va='top', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='green',
                                    alpha=0.7, edgecolor='black'))
        
        # Plot Elliott Wave patterns
        if self.patterns:
            if show_all_patterns:
                # Show top 3 patterns
                for i, pattern in enumerate(self.patterns[:3]):
                    alpha = 0.8 - (i * 0.2)
                    self._plot_pattern(ax1, pattern, alpha)
            else:
                # Show best pattern
                self._plot_pattern(ax1, self.patterns[0], 0.8)
            
            # Add pattern info
            best_pattern = self.patterns[0]
            info_text = f"Pattern: {best_pattern.pattern_type.upper()}\n"
            info_text += f"Confidence: {best_pattern.confidence:.1%}\n"
            info_text += f"Degree: {best_pattern.degree}\n"
            info_text += f"Waves: {len(best_pattern.waves)}"
            
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Volume
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        colors = ['green' if self.data['Close'].iloc[i] > self.data['Open'].iloc[i] 
                 else 'red' for i in range(len(self.data))]
        ax2.bar(range(len(self.data)), self.data['Volume'], color=colors, alpha=0.5)
        ax2.plot(range(len(self.data)), self.data['Volume_MA'], 'b-', linewidth=2)
        ax2.set_ylabel('Volume')
        
        # RSI
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(range(len(self.data)), self.data['RSI'], 'purple', linewidth=2)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        
        # Formatting
        ax1.set_xlim(-1, len(self.data))
        ax1.set_title(f'{self.symbol} - Elliott Wave Analysis with Market Structure', 
                     fontsize=16, weight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_pattern(self, ax, pattern: ElliottPattern, alpha: float = 0.8):
        """Plot Elliott Wave pattern"""
        colors_map = {
            'impulse': ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71'],
            'corrective': ['#e74c3c', '#2ecc71', '#e74c3c'],
            'WXY': ['#3498db', '#f39c12', '#3498db'],
            'WXYXZ': ['#3498db', '#f39c12', '#3498db', '#f39c12', '#3498db']
        }
        
        # Get colors for pattern
        if pattern.pattern_type in colors_map:
            colors = colors_map[pattern.pattern_type]
        elif 'flat' in pattern.pattern_type or pattern.pattern_type == 'zigzag':
            colors = colors_map['corrective']
        else:
            colors = ['#95a5a6'] * len(pattern.waves)
        
        # Plot each wave
        for i, wave in enumerate(pattern.waves):
            color = colors[i % len(colors)]
            
            # Plot wave line
            x_start = wave.start.index
            x_end = wave.end.index
            ax.plot([x_start, x_end], [wave.start.price, wave.end.price],
                   color=color, linewidth=3, alpha=alpha, solid_capstyle='round')
            
            # Add wave label
            x_mid = (x_start + x_end) / 2
            y_mid = (wave.start.price + wave.end.price) / 2
            
            # Wave label
            ax.text(x_mid, y_mid, str(wave.wave_number),
                   fontsize=14, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.5', facecolor='yellow',
                            alpha=alpha, edgecolor='black', linewidth=2))
            
            # Add percentage move
            pct_text = f"{wave.percentage_move:+.1f}%"
            offset = wave.start.price * 0.02
            y_offset = y_mid + offset if wave.is_bullish else y_mid - offset
            
            ax.text(x_mid, y_offset, pct_text, fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=alpha*0.8))
    
    def get_trading_signals(self) -> Dict:
        """Generate trading signals based on patterns"""
        if not self.patterns:
            return {"status": "No patterns found"}
        
        best_pattern = self.patterns[0]
        current_price = self.data['Close'].iloc[-1]
        
        signals = {
            "pattern": best_pattern.pattern_type,
            "confidence": best_pattern.confidence,
            "current_price": current_price,
            "analysis": self._analyze_pattern_position(best_pattern),
            "recommendation": self._get_recommendation(best_pattern),
            "key_levels": self._calculate_key_levels(best_pattern)
        }
        
        return signals
    
    def _analyze_pattern_position(self, pattern: ElliottPattern) -> str:
        """Analyze where we are in the pattern"""
        if pattern.pattern_type == 'impulse':
            # Determine which wave we're likely in
            current_time = self.data.index[-1]
            
            # Simple check - could be enhanced
            if len(pattern.waves) >= 5:
                return "Impulse wave complete - expect correction"
            else:
                return f"In wave {len(pattern.waves) + 1} of impulse"
        
        elif pattern.pattern_type in ['zigzag', 'flat', 'expanded_flat']:
            return "In corrective phase - watch for completion"
        
        elif pattern.pattern_type in ['WXY', 'WXYXZ']:
            return "Complex correction - may extend further"
        
        return "Pattern in progress"
    
    def _get_recommendation(self, pattern: ElliottPattern) -> str:
        """Get trading recommendation"""
        if pattern.confidence < 0.6:
            return "Low confidence - wait for confirmation"
        
        if pattern.pattern_type == 'impulse' and len(pattern.waves) >= 5:
            return "Consider taking profits - correction likely"
        elif pattern.pattern_type == 'impulse' and len(pattern.waves) == 4:
            return "Prepare for final wave 5 - potential long opportunity"
        elif pattern.pattern_type in ['zigzag', 'flat'] and len(pattern.waves) >= 3:
            return "Correction may be ending - watch for reversal"
        
        return "Monitor pattern development"
    
    def _calculate_key_levels(self, pattern: ElliottPattern) -> Dict[str, float]:
        """Calculate key support and resistance levels"""
        prices = []
        for wave in pattern.waves:
            prices.extend([wave.start.price, wave.end.price])
        
        return {
            "support": min(prices),
            "resistance": max(prices),
            "mid_point": (min(prices) + max(prices)) / 2,
            "current": self.data['Close'].iloc[-1]
        }

# ==================== Example Usage ====================

def main():
    """Main function to demonstrate proper Elliott Wave analysis"""
    print("=" * 60)
    print("Elliott Wave Analysis with Market Structure")
    print("=" * 60)
    
    # Analyze Bitcoin
    analyzer = ElliottWaveAnalyzer("BTC", period="3mo", interval="4h")
    
    # Download data
    data = analyzer.download_data()
    
    if data is not None:
        # Find patterns with proper swing detection
        print("\nAnalyzing market structure and Elliott Waves...")
        patterns = analyzer.find_patterns(
            min_swing_percent=3.0,  # Minimum 3% move for a swing
            lookback=10  # Bars to confirm swing high/low
        )
        
        if patterns:
            print(f"\n✓ Analysis complete!")
            
            # Get trading signals
            signals = analyzer.get_trading_signals()
            print(f"\n📊 Trading Signals:")
            print(f"   Pattern: {signals['pattern']}")
            print(f"   Confidence: {signals['confidence']:.1%}")
            print(f"   Analysis: {signals['analysis']}")
            print(f"   Recommendation: {signals['recommendation']}")
            print(f"   Key Levels:")
            print(f"     • Support: ${signals['key_levels']['support']:,.2f}")
            print(f"     • Resistance: ${signals['key_levels']['resistance']:,.2f}")
            print(f"     • Current: ${signals['key_levels']['current']:,.2f}")
            
            # Show the chart
            print("\n📈 Displaying chart...")
            analyzer.plot_analysis(show_all_patterns=False)
        else:
            print("\n⚠️  No clear Elliott Wave patterns detected")
            print("   This could mean:")
            print("   • Market is in transition")
            print("   • Need to adjust parameters")
            print("   • Try different timeframe")

if __name__ == "__main__":
    main()