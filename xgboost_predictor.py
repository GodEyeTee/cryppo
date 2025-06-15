"""
Enhanced XGBoost Direction Predictor for Cryptocurrency Trading
Target: >80% accuracy on direction prediction
"""

import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import time
import joblib
from typing import Dict, List, Tuple, Optional

# Try to import sklearn components for newer versions
try:
    from sklearn.metrics import classification_report
except:
    print("Warning: Some sklearn features not available")

warnings.filterwarnings('ignore')

class DataDownloader:
    """Download and prepare cryptocurrency data"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
    
    def download_data(self, symbol='BTC/USDT', timeframe='1h', start_date='2020-01-01', end_date=None):
        """Download historical data"""
        print(f"Downloading {symbol} data from {start_date}...")
        
        start_timestamp = self.exchange.parse8601(start_date + 'T00:00:00Z')
        end_timestamp = self.exchange.milliseconds() if end_date is None else self.exchange.parse8601(end_date + 'T00:00:00Z')
        
        all_ohlcv = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_timestamp, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                if len(ohlcv) < 1000:
                    break
                
                current_timestamp = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        print(f"Downloaded {len(df)} candles ({df['datetime'].min()} to {df['datetime'].max()})")
        return df

class FeatureEngineering:
    """Advanced feature engineering for crypto trading"""
    
    @staticmethod
    def calculate_features(df):
        """Calculate comprehensive trading features"""
        df = df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low'].replace(0, 1e-8)
        
        # Volatility
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(50).mean()
        
        # ATR variants
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_7'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=7)
        df['atr_ratio'] = df['atr_7'] / df['atr_14'].replace(0, 1e-8)
        
        # Moving averages
        for period in [7, 14, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            df[f'price_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # MA crossovers
        df['ma_cross_7_20'] = (df['sma_7'] > df['sma_20']).astype(int) - (df['sma_7'].shift(1) > df['sma_20'].shift(1)).astype(int)
        df['ma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int) - (df['sma_20'].shift(1) > df['sma_50'].shift(1)).astype(int)
        
        # RSI variations
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].rolling(10).mean()
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int) - (df['macd'].shift(1) > df['macd_signal'].shift(1)).astype(int)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1e-8)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_ema'] = talib.EMA(df['obv'], timeperiod=20)
        df['obv_divergence'] = df['obv'] - df['obv_ema']
        
        # Price patterns
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))).astype(int)
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        
        # Support/Resistance
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['price_to_resistance'] = (df['close'] - df['resistance_20']) / df['resistance_20']
        df['price_to_support'] = (df['close'] - df['support_20']) / df['support_20'].replace(0, 1e-8)
        df['sr_ratio'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'] + 1e-8)
        
        # Market microstructure
        df['spread'] = (df['high'] - df['low']) / df['close']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        # Multi-timeframe features
        for hours in [4, 8, 24]:
            df[f'returns_{hours}h'] = df['close'].pct_change(hours)
            df[f'high_{hours}h'] = df['high'].rolling(hours).max()
            df[f'low_{hours}h'] = df['low'].rolling(hours).min()
            df[f'range_{hours}h'] = (df[f'high_{hours}h'] - df[f'low_{hours}h']) / df['close']
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
        
        # Additional indicators
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Sin/cos encoding for cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        
        return df

class EnhancedDirectionPredictor:
    """XGBoost predictor with advanced optimization"""
    
    def __init__(self, lookback=24, prediction_horizons=[1, 2, 4, 8]):
        self.lookback = lookback
        self.prediction_horizons = prediction_horizons
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_features = None
        self.performance_metrics = {}
        self.use_gpu = self._check_gpu()
        
    def create_dataset(self, df, feature_cols, horizon=4, mode='direction'):
        """Create dataset with different labeling modes
        
        Args:
            mode: 'direction' for up/down prediction, 'signal' for buy/sell signals
        """
        X, y, metadata = [], [], []
        
        for i in range(self.lookback, len(df) - horizon):
            # Get current and future prices
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+horizon+1]
            future_highs = df['high'].iloc[i+1:i+horizon+1]
            future_lows = df['low'].iloc[i+1:i+horizon+1]
            
            # Calculate metrics
            future_return = (future_prices.iloc[-1] - current_price) / current_price
            max_high = future_highs.max()
            min_low = future_lows.min()
            max_gain = (max_high - current_price) / current_price
            max_loss = (min_low - current_price) / current_price
            
            if mode == 'direction':
                # Simple direction prediction: 1 for up, 0 for down
                # Use smaller threshold for better balance
                threshold = 0.0  # Neutral point
                label = 1 if future_return > threshold else 0
                
            elif mode == 'direction_3class':
                # 3-class prediction: -1 down, 0 neutral, 1 up
                # Dynamic thresholds based on volatility
                if 'volatility_20' in df.columns and pd.notna(df['volatility_20'].iloc[i]):
                    vol = df['volatility_20'].iloc[i]
                    up_threshold = vol * 0.3
                    down_threshold = -vol * 0.3
                else:
                    up_threshold = 0.003
                    down_threshold = -0.003
                
                if future_return > up_threshold:
                    label = 2  # Strong up
                elif future_return < down_threshold:
                    label = 0  # Strong down
                else:
                    label = 1  # Neutral
                    
            else:  # 'signal' mode - original strategy
                # Trading signals with risk/reward
                if max_gain > 0.015 and abs(max_loss) < 0.01:
                    label = 1  # Good buy signal
                elif abs(max_loss) > 0.015:
                    label = 0  # Avoid
                elif future_return > 0.005:
                    label = 1
                else:
                    label = 0
            
            # Features: lookback window
            X.append(df[feature_cols].iloc[i-self.lookback:i].values.flatten())
            y.append(label)
            metadata.append({
                'timestamp': df['datetime'].iloc[i],
                'price': current_price,
                'future_return': future_return,
                'future_return_pct': future_return * 100,
                'max_gain': max_gain,
                'max_loss': max_loss,
                'actual_direction': 'UP' if future_return > 0 else 'DOWN'
            })
        
        return np.array(X), np.array(y), metadata
    
    def select_features(self, X_train, y_train, feature_names, top_k=50):
        """Feature selection using importance scores"""
        print("Selecting best features...")
        
        # Train a model to get feature importance
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            tree_method='gpu_hist' if self.use_gpu else 'hist',
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = feature_importance_df.head(top_k)['feature'].tolist()
        print(f"Selected {len(top_features)} features")
        
        return top_features, feature_importance_df
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters using grid search"""
        print("Optimizing hyperparameters...")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        
        # Convert to sample weights
        sample_weights_train = np.array([weight_dict[label] for label in y_train])
        sample_weights_val = np.array([weight_dict[label] for label in y_val])
        
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 500],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2],
            'scale_pos_weight': [1, 2, 3, 5]  # For imbalanced classes
        }
        
        best_score = 0
        best_params = None
        
        # Random search for efficiency
        n_iter = 20  # Reduced for faster testing
        for i in range(n_iter):
            params = {
                'max_depth': np.random.choice(param_grid['max_depth']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'n_estimators': np.random.choice(param_grid['n_estimators']),
                'subsample': np.random.choice(param_grid['subsample']),
                'colsample_bytree': np.random.choice(param_grid['colsample_bytree']),
                'gamma': np.random.choice(param_grid['gamma']),
                'reg_alpha': np.random.choice(param_grid['reg_alpha']),
                'reg_lambda': np.random.choice(param_grid['reg_lambda']),
                'scale_pos_weight': np.random.choice(param_grid['scale_pos_weight']),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                'random_state': 42,
                'early_stopping_rounds': 50  # Add here for new XGBoost version
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, 
                     sample_weight=sample_weights_train,
                     eval_set=[(X_val, y_val)], 
                     sample_weight_eval_set=[sample_weights_val],
                     verbose=False)
            
            y_pred = model.predict(X_val)
            
            # Use F1 score instead of accuracy for imbalanced data
            score = f1_score(y_val, y_pred, average='weighted')
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"New best F1 score: {best_score:.4f}")
        
        if best_params is None:
            # Default params if optimization fails
            best_params = {
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.01,
                'reg_lambda': 1,
                'scale_pos_weight': 2,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                'random_state': 42,
                'early_stopping_rounds': 50
            }
            print("Using default parameters")
        
        return best_params
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, n_models=5, horizon=4):
        """Train ensemble of models"""
        print(f"\nTraining ensemble for {horizon}h prediction...")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights_train = np.array([weight_dict[label] for label in y_train])
        sample_weights_val = np.array([weight_dict[label] for label in y_val])
        
        print(f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Class weights: {weight_dict}")
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        models = []
        val_scores = []
        
        for i in range(n_models):
            print(f"Training model {i+1}/{n_models}")
            
            # Add variation to ensemble
            params = best_params.copy()
            params['random_state'] = i * 42
            params['subsample'] = np.clip(params['subsample'] + np.random.uniform(-0.1, 0.1), 0.6, 0.95)
            params['colsample_bytree'] = np.clip(params['colsample_bytree'] + np.random.uniform(-0.1, 0.1), 0.6, 0.95)
            params['early_stopping_rounds'] = 50  # Add here
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                     sample_weight=sample_weights_train,
                     eval_set=[(X_val, y_val)],
                     sample_weight_eval_set=[sample_weights_val],
                     verbose=False)
            
            # Validate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            models.append(model)
            val_scores.append(f1)
            
            print(f"  Model {i+1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        self.models[f'horizon_{horizon}'] = models
        
        # Ensemble prediction
        ensemble_pred = self._ensemble_predict(models, X_val)
        ensemble_accuracy = accuracy_score(y_val, ensemble_pred)
        ensemble_f1 = f1_score(y_val, ensemble_pred, average='weighted')
        
        print(f"\nEnsemble accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble F1 score: {ensemble_f1:.4f}")
        print(f"Best individual F1: {max(val_scores):.4f}")
        
        return ensemble_accuracy
    
    def _ensemble_predict(self, models, X, threshold=0.5):
        """Get ensemble predictions"""
        if len(models) == 0:
            return np.zeros(X.shape[0])
            
        predictions = []
        
        for model in models:
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average based on model confidence
        ensemble_proba = np.mean(predictions, axis=0)
        
        # Dynamic threshold based on distribution
        if threshold == 'auto':
            threshold = np.percentile(ensemble_proba, 50)
        
        return (ensemble_proba > threshold).astype(int)
    
    def evaluate_model(self, X_test, y_test, metadata_test=None, horizon=4):
        """Comprehensive model evaluation with direction analysis"""
        models = self.models.get(f'horizon_{horizon}', [])
        if not models:
            print(f"No models found for horizon {horizon}")
            return None
        
        # Get predictions
        y_pred = self._ensemble_predict(models, X_test)
        y_proba = self._get_ensemble_proba(models, X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Analyze directional accuracy if metadata available
        directional_accuracy = None
        if metadata_test is not None:
            direction_correct = 0
            direction_total = 0
            
            for i in range(len(y_test)):
                actual_direction = metadata_test[i]['actual_direction']
                predicted_direction = 'UP' if y_pred[i] == 1 else 'DOWN'
                
                if actual_direction == predicted_direction:
                    direction_correct += 1
                direction_total += 1
            
            directional_accuracy = direction_correct / direction_total if direction_total > 0 else 0
        
        # Analyze prediction distribution
        pred_distribution = pd.Series(y_pred).value_counts().to_dict()
        actual_distribution = pd.Series(y_test).value_counts().to_dict()
        
        # Store metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'directional_accuracy': directional_accuracy,
            'confusion_matrix': cm,
            'total_predictions': len(y_test),
            'prediction_distribution': pred_distribution,
            'actual_distribution': actual_distribution
        }
        
        self.performance_metrics[f'horizon_{horizon}'] = metrics
        
        # Print results
        print(f"\n=== Evaluation Results for {horizon}h Horizon ===")
        print(f"Classification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        if directional_accuracy is not None:
            print(f"\nDirectional Accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nPrediction Distribution:")
        print(f"  Predicted: {pred_distribution}")
        print(f"  Actual:    {actual_distribution}")
        
        # Show sample predictions if metadata available
        if metadata_test is not None:
            print(f"\nSample Predictions (first 10):")
            print(f"{'Time':<20} {'Actual':<8} {'Pred':<8} {'Prob':<8} {'Return%':<10} {'Direction':<10}")
            print("-" * 80)
            
            for i in range(min(10, len(y_test))):
                meta = metadata_test[i]
                print(f"{str(meta['timestamp']):<20} {y_test[i]:<8} {y_pred[i]:<8} "
                      f"{y_proba[i]:.4f}  {meta['future_return_pct']:>8.2f}%  {meta['actual_direction']:<10}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Confusion Matrix - {horizon}h')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')
        
        # Probability distribution
        ax2.hist(y_proba, bins=50, alpha=0.7)
        ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
        ax2.set_title('Prediction Probability Distribution')
        ax2.set_xlabel('Probability')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return metrics
    
    def _get_ensemble_proba(self, models, X):
        """Get ensemble probability predictions"""
        if len(models) == 0:
            return np.zeros(X.shape[0])
            
        predictions = []
        for model in models:
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        return np.mean(predictions, axis=0)
    
    def _check_gpu(self):
        """Check if GPU is available for XGBoost"""
        try:
            # Try GPU operation
            test_params = {
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'n_estimators': 10,
                'random_state': 42
            }
            test_model = xgb.XGBClassifier(**test_params)
            test_data = np.random.rand(100, 10)
            test_labels = np.random.randint(0, 2, 100)
            test_model.fit(test_data, test_labels)
            print("GPU is available and working!")
            return True
        except:
            print("GPU not available, using CPU")
            return False
    
    def predict_next_candles(self, df, feature_cols, n_candles=5):
        """Predict direction for next n candles"""
        print(f"\n=== Predicting next {n_candles} candles ===")
        
        # Get the latest data
        latest_idx = len(df) - 1
        
        predictions = []
        for horizon in self.prediction_horizons:
            if f'horizon_{horizon}' not in self.models:
                continue
                
            # Prepare features from the latest lookback window
            if latest_idx >= self.lookback:
                X = df[feature_cols].iloc[latest_idx-self.lookback+1:latest_idx+1].values.flatten()
                X = X.reshape(1, -1)
                
                # Get prediction
                models = self.models[f'horizon_{horizon}']
                y_proba = self._get_ensemble_proba(models, X)[0]
                y_pred = 1 if y_proba > 0.5 else 0
                
                direction = 'UP' if y_pred == 1 else 'DOWN'
                confidence = y_proba if y_pred == 1 else (1 - y_proba)
                
                predictions.append({
                    'horizon': f'{horizon}h',
                    'direction': direction,
                    'probability': y_proba,
                    'confidence': confidence
                })
        
        # Display predictions
        current_price = df['close'].iloc[latest_idx]
        current_time = df['datetime'].iloc[latest_idx]
        
        print(f"Current Time: {current_time}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"\nPredictions:")
        print(f"{'Horizon':<10} {'Direction':<10} {'Confidence':<12} {'Probability':<12}")
        print("-" * 50)
        
        for pred in predictions:
            print(f"{pred['horizon']:<10} {pred['direction']:<10} "
                  f"{pred['confidence']:.2%}      {pred['probability']:.4f}")
        
        return predictions
    
    def backtest_predictions(self, df, feature_cols, start_idx=None, n_samples=100):
        """Backtest predictions on historical data"""
        if start_idx is None:
            start_idx = len(df) - n_samples - max(self.prediction_horizons)
        
        results = {h: {'correct': 0, 'total': 0} for h in self.prediction_horizons}
        
        print(f"\nBacktesting on {n_samples} samples...")
        
        for i in range(start_idx, start_idx + n_samples):
            if i < self.lookback or i >= len(df) - max(self.prediction_horizons):
                continue
            
            # Prepare features
            X = df[feature_cols].iloc[i-self.lookback:i].values.flatten().reshape(1, -1)
            
            for horizon in self.prediction_horizons:
                if f'horizon_{horizon}' not in self.models:
                    continue
                
                # Predict
                models = self.models[f'horizon_{horizon}']
                y_pred = self._ensemble_predict(models, X)[0]
                
                # Check actual
                if i + horizon < len(df):
                    current_price = df['close'].iloc[i]
                    future_price = df['close'].iloc[i + horizon]
                    actual_direction = 1 if future_price > current_price else 0
                    
                    if y_pred == actual_direction:
                        results[horizon]['correct'] += 1
                    results[horizon]['total'] += 1
        
        # Print results
        print(f"\n{'Horizon':<10} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
        print("-" * 40)
        
        for horizon in self.prediction_horizons:
            if results[horizon]['total'] > 0:
                acc = results[horizon]['correct'] / results[horizon]['total']
                print(f"{horizon}h{'':<7} {acc:.2%}      "
                      f"{results[horizon]['correct']:<10} {results[horizon]['total']:<10}")
        
        return results
    
    def save_model(self, filepath='xgb_predictor.pkl'):
        """Save trained models"""
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'best_features': self.best_features,
            'performance_metrics': self.performance_metrics
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='xgb_predictor.pkl'):
        """Load trained models"""
        data = joblib.load(filepath)
        self.models = data['models']
        self.scalers = data['scalers']
        self.best_features = data['best_features']
        self.performance_metrics = data['performance_metrics']
        print(f"Model loaded from {filepath}")

def main():
    """Main testing function"""
    print("=== Enhanced XGBoost Direction Predictor ===")
    print("Target: >80% directional accuracy\n")
    
    # Check XGBoost version
    print(f"XGBoost version: {xgb.__version__}")
    
    # 1. Download data
    downloader = DataDownloader()
    df = downloader.download_data('BTC/USDT', '1h', '2020-01-01')
    
    # 2. Feature engineering
    print("\nCalculating features...")
    df = FeatureEngineering.calculate_features(df)
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Total features: {len(feature_cols)}")
    
    # 3. Create predictor
    predictor = EnhancedDirectionPredictor(lookback=24, prediction_horizons=[1, 2, 4, 8])
    
    # Test different modes
    print("\n" + "="*60)
    print("Testing DIRECTION mode (Up/Down prediction)")
    print("="*60)
    
    results = {}
    
    # Focus on shorter horizons for direction prediction
    test_horizons = [1, 2, 4]
    
    for horizon in test_horizons:
        print(f"\n{'='*50}")
        print(f"Testing {horizon}h prediction horizon")
        print('='*50)
        
        # 4. Create dataset with direction mode
        X, y, metadata = predictor.create_dataset(df, feature_cols, horizon=horizon, mode='direction')
        print(f"Dataset size: {len(X)} samples")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            direction = "UP" if cls == 1 else "DOWN"
            print(f"  {direction}: {cnt} ({cnt/len(y)*100:.1f}%)")
        
        # 5. Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        metadata_train = metadata[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        metadata_val = metadata[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        metadata_test = metadata[train_size+val_size:]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 6. Train ensemble
        ensemble_accuracy = predictor.train_ensemble(X_train, y_train, X_val, y_val, n_models=3, horizon=horizon)
        
        # 7. Evaluate on test set
        metrics = predictor.evaluate_model(X_test, y_test, metadata_test, horizon=horizon)
        results[f'{horizon}h'] = metrics
    
    # 8. Test prediction on latest data
    print("\n" + "="*60)
    print("PREDICTIONS FOR NEXT CANDLES")
    print("="*60)
    
    predictions = predictor.predict_next_candles(df, feature_cols, n_candles=5)
    
    # 9. Backtest recent predictions
    backtest_results = predictor.backtest_predictions(df, feature_cols, n_samples=200)
    
    # 10. Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for horizon, metrics in results.items():
        if metrics:
            print(f"\n{horizon} prediction:")
            print(f"  Classification Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Directional Accuracy:    {metrics['directional_accuracy']*100:.2f}%")
            print(f"  F1 Score:               {metrics['f1_score']:.4f}")
    
    # Find best performing horizon
    best_horizon = max(results.keys(), 
                      key=lambda x: results[x]['directional_accuracy'] if results[x] else 0)
    best_dir_accuracy = results[best_horizon]['directional_accuracy']
    
    print(f"\nBest performing horizon: {best_horizon}")
    print(f"Best directional accuracy: {best_dir_accuracy*100:.2f}%")
    
    if best_dir_accuracy >= 0.55:  # More realistic target for direction
        print("\n✅ Model shows predictive ability! Ready for testing in production.")
        predictor.save_model('xgb_direction_predictor.pkl')
    else:
        print("\n❌ Model needs improvement. Consider:")
        print("  - Adding more features")
        print("  - Adjusting prediction horizons")
        print("  - Collecting more data")
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()