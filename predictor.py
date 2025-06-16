import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class CryptoDirectionPredictor:
    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT', 'DOGE/USDT'], 
                 lookback=300, forecast_steps=12, threshold=0.04):
        """
        Initialize the predictor
        
        Args:
            symbols: List of cryptocurrency pairs to trade
            lookback: Number of past candles to use for prediction (300)
            forecast_steps: Number of future candles to predict (12)
            threshold: Percentage threshold for up/down classification (4%)
        """
        self.symbols = symbols
        self.lookback = lookback
        self.forecast_steps = forecast_steps
        self.threshold = threshold
        self.exchange = ccxt.binance()
        self.scaler = StandardScaler()
        self.model = None
        
    def download_data(self, start_date='2019-01-01'):
        """Download historical data from exchange with rate limit handling"""
        print("Downloading data...")
        all_data = {}
        
        start_timestamp = self.exchange.parse8601(start_date + 'T00:00:00Z')
        end_timestamp = self.exchange.milliseconds()
        
        for symbol in self.symbols:
            print(f"Downloading {symbol}...")
            ohlcv_data = []
            current_timestamp = start_timestamp
            
            while current_timestamp < end_timestamp:
                try:
                    # Fetch 1000 candles at a time (Binance limit)
                    candles = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe='1h',
                        since=current_timestamp,
                        limit=1000
                    )
                    
                    if not candles:
                        break
                        
                    ohlcv_data.extend(candles)
                    current_timestamp = candles[-1][0] + 3600000  # Move to next hour
                    
                    # Rate limit handling
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    time.sleep(5)
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            all_data[symbol] = df
            print(f"Downloaded {len(df)} candles for {symbol}")
            
        return all_data
    
    def calculate_features(self, df):
        """Calculate technical indicators and features"""
        # Log transform prices to normalize across different price scales
        df['log_open'] = np.log(df['open'])
        df['log_high'] = np.log(df['high'])
        df['log_low'] = np.log(df['low'])
        df['log_close'] = np.log(df['close'])
        
        # Price returns (log returns for better statistical properties)
        df['returns'] = df['log_close'].diff()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI Overbought/Oversold flags
        df['rsi_ob'] = (df['rsi'] > 70).astype(int)
        df['rsi_os'] = (df['rsi'] < 30).astype(int)
        
        # Additional features for better prediction
        df['hl_ratio'] = df['log_high'] - df['log_low']  # Volatility measure
        df['co_ratio'] = df['log_close'] - df['log_open']  # Candle body
        
        # Drop original price columns (we use log-transformed versions)
        df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def create_labels(self, df):
        """Create labels for direction prediction"""
        # Calculate future returns (sum of next 12 hours)
        future_returns = df['returns'].rolling(window=self.forecast_steps).sum().shift(-self.forecast_steps)
        
        # Create labels: 0=down, 1=sideways, 2=up
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_returns < -self.threshold] = 0  # Down
        labels[(future_returns >= -self.threshold) & (future_returns <= self.threshold)] = 1  # Sideways
        labels[future_returns > self.threshold] = 2  # Up
        
        return labels
    
    def prepare_sequences(self, features, labels):
        """Prepare sequences for LSTM input"""
        X, y = [], []
        
        for i in range(self.lookback, len(features) - self.forecast_steps):
            X.append(features[i-self.lookback:i])
            y.append(labels[i])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, num_classes=3):
        """Build LSTM model architecture"""
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer with return sequences
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_dict):
        """Train the LSTM model on multiple symbols"""
        print("Preparing data for training...")
        
        all_X = []
        all_y = []
        
        # Process each symbol
        for symbol, df in data_dict.items():
            print(f"Processing {symbol}...")
            
            # Calculate features
            df_features = self.calculate_features(df.copy())
            
            # Create labels
            labels = self.create_labels(df_features)
            
            # Remove rows with NaN labels
            valid_idx = ~labels.isna()
            df_features = df_features[valid_idx]
            labels = labels[valid_idx]
            
            # Normalize features
            features_array = df_features.values
            
            # Create sequences
            X, y = self.prepare_sequences(features_array, labels.values)
            
            all_X.append(X)
            all_y.append(y)
            
        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        # Normalize features across all data
        n_samples, n_steps, n_features = X_combined.shape
        X_reshaped = X_combined.reshape(-1, n_features)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_combined = X_normalized.reshape(n_samples, n_steps, n_features)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y_combined, num_classes=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_categorical, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Build model
        self.model = self.build_model(input_shape=(n_steps, n_features))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Class distribution analysis
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nPrediction Distribution:")
        print(f"Down: {np.sum(y_pred_classes == 0)} predictions")
        print(f"Sideways: {np.sum(y_pred_classes == 1)} predictions")
        print(f"Up: {np.sum(y_pred_classes == 2)} predictions")
        
        return history
    
    def predict(self, symbol_data):
        """Make predictions on new data"""
        # Calculate features
        df_features = self.calculate_features(symbol_data.copy())
        
        # Get last lookback periods
        if len(df_features) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} periods of data")
        
        features = df_features.values[-self.lookback:]
        
        # Normalize
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_normalized = self.scaler.transform(features_reshaped)
        features = features_normalized.reshape(1, self.lookback, features.shape[-1])
        
        # Predict
        prediction = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Map to direction
        directions = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
        confidence = prediction[0][predicted_class]
        
        return directions[predicted_class], confidence, prediction[0]

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = CryptoDirectionPredictor(
        symbols=['BTC/USDT', 'ETH/USDT', 'DOGE/USDT'],
        lookback=300,
        forecast_steps=12,
        threshold=0.04
    )
    
    # Download data
    data = predictor.download_data(start_date='2019-01-01')
    
    # Train model
    history = predictor.train(data)
    
    # Save model
    predictor.model.save('crypto_direction_lstm_model.h5')
    np.save('scaler_params.npy', [predictor.scaler.mean_, predictor.scaler.scale_])
    
    # Example prediction on latest data
    print("\n--- Making prediction on latest BTC data ---")
    latest_btc = data['BTC/USDT'].tail(300)
    direction, confidence, probabilities = predictor.predict(latest_btc)
    
    print(f"Predicted direction for next 12 hours: {direction}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probabilities - Down: {probabilities[0]:.2%}, Sideways: {probabilities[1]:.2%}, Up: {probabilities[2]:.2%}")