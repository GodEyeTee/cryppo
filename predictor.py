import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
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
    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOGE/USDT'], 
                 lookback=300, forecast_steps=12, threshold=0.02):
        """
        Initialize the predictor with PyTorch
        
        Args:
            symbols: List of cryptocurrency pairs to trade
            lookback: Number of past candles to use for prediction (300)
            forecast_steps: Number of future candles to predict (12)
            threshold: Percentage threshold for up/down classification (2% = 0.02)
                      - DOWN: < -2%
                      - SIDEWAYS: -2% to 2%
                      - UP: > 2%
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
        
        # Calculate class weights for imbalanced dataset
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_weights = []
        for i in range(3):
            if i in unique_classes:
                idx = np.where(unique_classes == i)[0][0]
                weight = len(y_train) / (3 * class_counts[idx])
                class_weights.append(weight)
            else:
                class_weights.append(1.0)
        
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        print("\nClass weights for balanced training:")
        print(f"  Down: {class_weights[0]:.3f}")
        print(f"  Sideways: {class_weights[1]:.3f}")
        print(f"  Up: {class_weights[2]:.3f}")
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        
        # Final evaluation with confusion matrix
        self.model.eval()
        test_correct = 0
        test_total = 0
        predictions_dist = {0: 0, 1: 0, 2: 0}
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
                # Store predictions and labels for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
                # Count predictions
                for pred in predicted.cpu().numpy():
                    predictions_dist[pred] += 1
        
        test_accuracy = 100 * test_correct / test_total
        print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
        
        print("\nPrediction Distribution:")
        print(f"Down: {predictions_dist[0]} predictions")
        print(f"Sideways: {predictions_dist[1]} predictions")
        print(f"Up: {predictions_dist[2]} predictions")
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Down', 'Sideways', 'Up'],
                    yticklabels=['Down', 'Sideways', 'Up'])
        plt.title('Confusion Matrix - Crypto Direction Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Calculate percentages for confusion matrix
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        print("\nConfusion Matrix (Percentages):")
        print(f"{'':>10} {'Down':>10} {'Sideways':>10} {'Up':>10}")
        labels = ['Down', 'Sideways', 'Up']
        for i, label in enumerate(labels):
            print(f"{label:>10} {cm_percent[i,0]:>10.1f}% {cm_percent[i,1]:>10.1f}% {cm_percent[i,2]:>10.1f}%")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=['Down', 'Sideways', 'Up'],
                                  digits=4))
        
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
        
        # Check if we have enough data after feature calculation
        if len(df_features) < self.lookback:
            print(f"Warning: After feature calculation, only {len(df_features)} periods available.")
            print(f"Need at least {self.lookback} periods. Try providing more raw data.")
            raise ValueError(f"Need at least {self.lookback} periods of data after feature calculation. Got {len(df_features)}")
        
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
    
    def predict_latest(self, symbol='BTC/USDT', hours_back=500):
        """
        Convenient method to predict latest price direction
        
        Args:
            symbol: Trading pair to predict
            hours_back: Number of hours of data to fetch (default 500 to ensure enough data)
        """
        print(f"Fetching latest {hours_back} hours of {symbol} data...")
        
        # Get recent data
        end_timestamp = self.exchange.milliseconds()
        start_timestamp = end_timestamp - (hours_back * 3600000)  # Convert hours to milliseconds
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe='1h',
                since=start_timestamp,
                limit=hours_back
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Make prediction
            direction, confidence, probabilities = self.predict(df)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'probabilities': {
                    'down': probabilities[0],
                    'sideways': probabilities[1],
                    'up': probabilities[2]
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
            return None
    
    def analyze_predictions(self, test_data_dict=None):
        """
        Analyze model predictions on test data
        
        Args:
            test_data_dict: Dictionary of test data for each symbol
        """
        if self.model is None:
            print("No trained model found. Please train or load a model first.")
            return
        
        if test_data_dict is None:
            # Use last 1000 hours from each symbol for testing
            test_data_dict = {}
            for symbol in self.symbols:
                print(f"Fetching test data for {symbol}...")
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    test_data_dict[symbol] = df
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
        
        # Analyze predictions for each symbol
        results = {}
        for symbol, df in test_data_dict.items():
            try:
                # Get multiple predictions by sliding window
                predictions = []
                for i in range(len(df) - self.lookback - 50):
                    window_data = df.iloc[i:i+self.lookback+50]
                    direction, confidence, probs = self.predict(window_data)
                    predictions.append({
                        'direction': direction,
                        'confidence': confidence,
                        'probs': probs
                    })
                
                results[symbol] = predictions
                
                # Summary statistics
                if predictions:
                    dirs = [p['direction'] for p in predictions]
                    print(f"\n{symbol} Predictions Summary:")
                    print(f"  Total predictions: {len(dirs)}")
                    print(f"  UP: {dirs.count('UP')} ({dirs.count('UP')/len(dirs)*100:.1f}%)")
                    print(f"  SIDEWAYS: {dirs.count('SIDEWAYS')} ({dirs.count('SIDEWAYS')/len(dirs)*100:.1f}%)")
                    print(f"  DOWN: {dirs.count('DOWN')} ({dirs.count('DOWN')/len(dirs)*100:.1f}%)")
                    print(f"  Average confidence: {np.mean([p['confidence'] for p in predictions]):.2%}")
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        return results
    
    def find_optimal_threshold(self, data_dict, thresholds=[0.01, 0.015, 0.02, 0.025, 0.03]):
        """
        Find optimal threshold for classification
        
        Args:
            data_dict: Training data
            thresholds: List of thresholds to test
        """
        print("Finding optimal threshold...")
        results = {}
        
        for threshold in thresholds:
            print(f"\nTesting threshold: {threshold*100:.1f}%")
            
            # Count distribution for this threshold
            all_labels = []
            
            for symbol, df in data_dict.items():
                df_features = self.calculate_features(df.copy())
                
                # Calculate future returns
                future_returns = df_features['returns'].rolling(
                    window=self.forecast_steps
                ).sum().shift(-self.forecast_steps)
                
                # Create labels with current threshold
                labels = pd.Series(index=df_features.index, dtype='int8')
                labels[future_returns < -threshold] = 0  # Down
                labels[(future_returns >= -threshold) & (future_returns <= threshold)] = 1  # Sideways
                labels[future_returns > threshold] = 2  # Up
                
                # Remove NaN
                valid_labels = labels[~labels.isna()]
                all_labels.extend(valid_labels.values)
            
            # Calculate distribution
            unique, counts = np.unique(all_labels, return_counts=True)
            total = len(all_labels)
            
            distribution = {}
            for u, c in zip(unique, counts):
                label = ['Down', 'Sideways', 'Up'][u]
                distribution[label] = c / total * 100
            
            results[threshold] = distribution
            
            print(f"  Down: {distribution.get('Down', 0):.1f}%")
            print(f"  Sideways: {distribution.get('Sideways', 0):.1f}%")
            print(f"  Up: {distribution.get('Up', 0):.1f}%")
            
            # Calculate balance score (lower is better)
            balance_score = np.std([distribution.get('Down', 0), 
                                   distribution.get('Up', 0)])
            print(f"  Balance score: {balance_score:.2f}")
        
        return results
    
    def predict_with_multi_threshold(self, symbol_data, thresholds=[0.01, 0.015, 0.02]):
        """
        Make predictions using multiple thresholds for comparison
        """
        df_features = self.calculate_features(symbol_data.copy())
        
        if len(df_features) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} periods of data")
        
        features = df_features.values[-self.lookback:].astype('float32')
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_normalized = self.scaler.transform(features_reshaped).astype('float32')
        features_tensor = torch.FloatTensor(features_normalized.reshape(1, self.lookback, features.shape[-1])).to(self.device)
        
        # Get raw predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Interpret with different thresholds
        results = {}
        for threshold in thresholds:
            # Simple interpretation based on probability differences
            prob_diff_up = probabilities[2] - probabilities[1]
            prob_diff_down = probabilities[0] - probabilities[1]
            
            if prob_diff_up > threshold * 2:  # Adjust sensitivity
                direction = 'UP'
            elif prob_diff_down > threshold * 2:
                direction = 'DOWN'
            else:
                direction = 'SIDEWAYS'
            
            results[threshold] = {
                'direction': direction,
                'probabilities': probabilities,
                'confidence': max(probabilities)
            }
        
        return results
    
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
    # Initialize predictor with 5 symbols and 2% threshold
    predictor = CryptoDirectionPredictor(
        symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOGE/USDT'],
        lookback=300,
        forecast_steps=12,
        threshold=0.02  # Start with 2%
    )
    
    # Download data with smaller batches
    data = predictor.download_data(start_date='2019-01-01', batch_size=500)
    
    # Find optimal threshold
    print("\n" + "="*60)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*60)
    threshold_results = predictor.find_optimal_threshold(
        data, 
        thresholds=[0.01, 0.015, 0.02, 0.025, 0.03]
    )
    
    # Train model with class weights
    print("\n" + "="*60)
    print("TRAINING MODEL WITH CLASS WEIGHTS")
    print("="*60)
    history = predictor.train(data, batch_size=32, max_samples_per_symbol=50000, epochs=100)
    
    # Save model
    predictor.save_model()
    
    # Clear memory before prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Example predictions with multiple thresholds
    print("\n" + "="*60)
    print("PREDICTIONS WITH MULTIPLE THRESHOLDS")
    print("="*60)
    
    latest_btc = data['BTC/USDT'].tail(400)
    try:
        # Standard prediction
        direction, confidence, probabilities = predictor.predict(latest_btc)
        print(f"\nStandard prediction (threshold={predictor.threshold*100:.1f}%):")
        print(f"Direction: {direction} (Confidence: {confidence:.2%})")
        print(f"Probabilities - Down: {probabilities[0]:.2%}, Sideways: {probabilities[1]:.2%}, Up: {probabilities[2]:.2%}")
        
        # Multi-threshold predictions
        multi_results = predictor.predict_with_multi_threshold(
            latest_btc, 
            thresholds=[0.01, 0.015, 0.02]
        )
        
        print("\nPredictions with different thresholds:")
        for threshold, result in multi_results.items():
            print(f"  {threshold*100:.1f}%: {result['direction']} (confidence: {result['confidence']:.2%})")
            
    except Exception as e:
        print(f"Prediction error: {e}")
    
    # Analyze predictions for all symbols
    print("\n" + "="*60)
    print("ANALYZING PREDICTIONS FOR ALL SYMBOLS")
    print("="*60)
    analysis_results = predictor.analyze_predictions()
    
    # Final cleanup
    del data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()