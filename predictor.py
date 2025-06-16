import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU found: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
else:
    print("No GPU found, using CPU")

class CryptoDataset(Dataset):
    """PyTorch Dataset for cryptocurrency data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for direction prediction"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, num_classes=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(32, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Batch normalization
        out = self.batch_norm(out)
        
        # First FC layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.dropout1(out)
        
        # Second FC layer
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

class CryptoDirectionPredictor:
    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT', 'DOGE/USDT'], 
                 lookback=300, forecast_steps=12, threshold=0.04):
        """
        Initialize the predictor with PyTorch
        
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
        self.device = device
        
    def download_data(self, start_date='2019-01-01', batch_size=500):
        """Download historical data with memory-efficient batch processing"""
        print("Downloading data...")
        all_data = {}
        
        start_timestamp = self.exchange.parse8601(start_date + 'T00:00:00Z')
        end_timestamp = self.exchange.milliseconds()
        
        for symbol in self.symbols:
            print(f"Downloading {symbol}...")
            ohlcv_data = []
            current_timestamp = start_timestamp
            
            # Process in smaller batches to avoid memory issues
            batch_count = 0
            
            while current_timestamp < end_timestamp:
                try:
                    # Fetch smaller batches
                    candles = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe='1h',
                        since=current_timestamp,
                        limit=batch_size
                    )
                    
                    if not candles:
                        break
                    
                    ohlcv_data.extend(candles)
                    current_timestamp = candles[-1][0] + 3600000
                    
                    # Clear memory periodically
                    batch_count += 1
                    if batch_count % 10 == 0:
                        gc.collect()
                    
                    # Rate limit handling
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    time.sleep(5)
                    continue
            
            # Convert to DataFrame with memory optimization
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # Reduce memory usage by converting to float32
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype('float32')
            
            all_data[symbol] = df
            print(f"Downloaded {len(df)} candles for {symbol}")
            
            # Clear memory
            del ohlcv_data
            gc.collect()
            
        return all_data
    
    def calculate_features(self, df):
        """Calculate technical indicators with memory optimization"""
        # Keep close price as float64 for talib
        close_values = df['close'].values.astype('float64')
        
        # Now convert to float32 for memory efficiency
        df = df.astype('float32')
        
        # Log transform prices
        df['log_open'] = np.log(df['open'])
        df['log_high'] = np.log(df['high'])
        df['log_low'] = np.log(df['low'])
        df['log_close'] = np.log(df['close'])
        
        # Price returns
        df['returns'] = df['log_close'].diff()
        
        # RSI using talib (requires float64)
        df['rsi'] = talib.RSI(close_values, timeperiod=14).astype('float32')
        
        # MACD using talib (requires float64)
        macd, macd_signal, macd_hist = talib.MACD(close_values, 
                                                   fastperiod=12, 
                                                   slowperiod=26, 
                                                   signalperiod=9)
        df['macd'] = macd.astype('float32')
        df['macd_signal'] = macd_signal.astype('float32')
        df['macd_diff'] = macd_hist.astype('float32')
        
        # RSI Overbought/Oversold flags
        df['rsi_ob'] = (df['rsi'] > 70).astype('int8')
        df['rsi_os'] = (df['rsi'] < 30).astype('int8')
        
        # Additional features
        df['hl_ratio'] = df['log_high'] - df['log_low']
        df['co_ratio'] = df['log_close'] - df['log_open']
        
        # Drop original columns to save memory
        df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def create_labels(self, df):
        """Create labels for direction prediction"""
        # Calculate future returns
        future_returns = df['returns'].rolling(window=self.forecast_steps).sum().shift(-self.forecast_steps)
        
        # Create labels
        labels = pd.Series(index=df.index, dtype='int8')
        labels[future_returns < -self.threshold] = 0  # Down
        labels[(future_returns >= -self.threshold) & (future_returns <= self.threshold)] = 1  # Sideways
        labels[future_returns > self.threshold] = 2  # Up
        
        return labels
    
    def prepare_sequences_batch(self, features, labels, batch_size=10000):
        """Prepare sequences in batches to avoid memory issues"""
        n_samples = len(features) - self.lookback - self.forecast_steps
        
        # Process in batches
        X_list = []
        y_list = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = []
            y_batch = []
            
            for i in range(start_idx, end_idx):
                actual_idx = i + self.lookback
                X_batch.append(features[actual_idx-self.lookback:actual_idx])
                y_batch.append(labels[actual_idx])
            
            X_list.append(np.array(X_batch, dtype='float32'))
            y_list.append(np.array(y_batch, dtype='int8'))
            
            # Clear memory
            del X_batch, y_batch
            gc.collect()
        
        # Combine batches
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # Clear memory
        del X_list, y_list
        gc.collect()
        
        return X, y
    
    def train(self, data_dict, batch_size=32, max_samples_per_symbol=50000, epochs=100):
        """Train with PyTorch"""
        print("Preparing data for training...")
        
        all_X = []
        all_y = []
        
        # Process each symbol with memory limits
        for symbol, df in data_dict.items():
            print(f"Processing {symbol}...")
            
            # Limit data size if needed
            if len(df) > max_samples_per_symbol:
                df = df.tail(max_samples_per_symbol)
            
            # Calculate features
            df_features = self.calculate_features(df.copy())
            
            # Create labels
            labels = self.create_labels(df_features)
            
            # Remove NaN labels
            valid_idx = ~labels.isna()
            df_features = df_features[valid_idx]
            labels = labels[valid_idx]
            
            # Create sequences in batches
            X, y = self.prepare_sequences_batch(df_features.values, labels.values)
            
            all_X.append(X)
            all_y.append(y)
            
            # Clear memory
            del df, df_features, labels
            gc.collect()
        
        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        # Clear memory
        del all_X, all_y
        gc.collect()
        
        # Normalize features
        n_samples, n_steps, n_features = X_combined.shape
        X_reshaped = X_combined.reshape(-1, n_features)
        X_normalized = self.scaler.fit_transform(X_reshaped).astype('float32')
        X_combined = X_normalized.reshape(n_samples, n_steps, n_features)
        
        # Clear memory
        del X_reshaped, X_normalized
        gc.collect()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Clear combined data
        del X_combined, y_combined
        gc.collect()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Create PyTorch datasets
        train_dataset = CryptoDataset(X_train, y_train)
        test_dataset = CryptoDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=n_features,
            hidden_size=128,
            num_layers=3,
            num_classes=3,
            dropout=0.2
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        print("Training model...")
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = val_loss / len(test_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        self.model.eval()
        test_correct = 0
        test_total = 0
        predictions_dist = {0: 0, 1: 0, 2: 0}
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
                # Count predictions
                for pred in predicted.cpu().numpy():
                    predictions_dist[pred] += 1
        
        test_accuracy = 100 * test_correct / test_total
        print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
        
        print("\nPrediction Distribution:")
        print(f"Down: {predictions_dist[0]} predictions")
        print(f"Sideways: {predictions_dist[1]} predictions")
        print(f"Up: {predictions_dist[2]} predictions")
        
        # Clear memory
        del X_train, X_test, y_train, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {'final_accuracy': test_accuracy}
    
    def predict(self, symbol_data):
        """Make predictions with PyTorch model"""
        # Calculate features
        df_features = self.calculate_features(symbol_data.copy())
        
        # Get last lookback periods
        if len(df_features) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} periods of data")
        
        features = df_features.values[-self.lookback:].astype('float32')
        
        # Normalize
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_normalized = self.scaler.transform(features_reshaped).astype('float32')
        features = features_normalized.reshape(1, self.lookback, features.shape[-1])
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to direction
        directions = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
        
        return directions[predicted_class], confidence, probabilities[0].cpu().numpy()
    
    def save_model(self, model_path='crypto_lstm_model.pth', scaler_path='scaler_params.npy'):
        """Save PyTorch model and scaler"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_size': self.model.lstm.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                }
            }, model_path)
            np.save(scaler_path, [self.scaler.mean_, self.scaler.scale_])
            print(f"Model saved to {model_path}")
            print(f"Scaler parameters saved to {scaler_path}")
        else:
            print("No trained model to save")
    
    def load_model(self, model_path='crypto_lstm_model.pth', scaler_path='scaler_params.npy'):
        """Load PyTorch model and scaler"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['model_config']
            
            self.model = LSTMModel(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            scaler_params = np.load(scaler_path)
            self.scaler.mean_ = scaler_params[0]
            self.scaler.scale_ = scaler_params[1]
            
            print(f"Model loaded from {model_path}")
            print(f"Scaler parameters loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = CryptoDirectionPredictor(
        symbols=['BTC/USDT', 'ETH/USDT', 'DOGE/USDT'],
        lookback=300,
        forecast_steps=12,
        threshold=0.04
    )
    
    # Download data with smaller batches
    data = predictor.download_data(start_date='2019-01-01', batch_size=500)
    
    # Train model
    history = predictor.train(data, batch_size=32, max_samples_per_symbol=50000, epochs=100)
    
    # Save model
    predictor.save_model()
    
    # Clear memory before prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Example prediction
    print("\n--- Making prediction on latest BTC data ---")
    latest_btc = data['BTC/USDT'].tail(300)
    direction, confidence, probabilities = predictor.predict(latest_btc)
    
    print(f"Predicted direction for next 12 hours: {direction}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probabilities - Down: {probabilities[0]:.2%}, Sideways: {probabilities[1]:.2%}, Up: {probabilities[2]:.2%}")
    
    # Final cleanup
    del data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()