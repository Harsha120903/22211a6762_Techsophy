import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

# Try importing TensorFlow for LSTM, with fallback to z-score
try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
except ImportError:
    tf = None
    logging.warning("TensorFlow not available; using z-score method only")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockAnomalyDetector:
    """Class to monitor stock prices and detect anomalies in real-time."""
    
    def __init__(self, config_path=None):
        """
        Initialize with configuration from a JSON file or hardcoded config.
        
        Args:
            config_path (str): Path to configuration file (optional).
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {str(e)}")
                self.config = {}
        else:
            self.config = {}
        
        # Hardcoded config as fallback
        default_config = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'period': '1d',
            'interval': '1m',
            'window': 5,
            'z_threshold': 1.2,
            'percent_change_threshold': 0.3,
            'market_open_ignore_minutes': 10,
            'poll_interval': 15,
            'log_file': 'alerts_log.txt',
            'json_output': 'anomalies.json',
            'detection_method': 'z_score',
            'lstm_params': {'lookback': 10, 'prediction_threshold': 0.003, 'epochs': 20, 'batch_size': 32},
            'max_iterations': 3
        }
        
        # Merge provided config with default, prioritizing provided values
        self.config = {**default_config, **self.config}
        
        self.symbols = self.config.get('symbols', [])
        self.period = self.config.get('period', '1d')
        self.interval = self.config.get('interval', '1m')
        self.window = self.config.get('window', 5)
        self.z_threshold = self.config.get('z_score_threshold', 1.2) or self.config.get('z_threshold', 1.2)
        self.percent_change_threshold = self.config.get('percent_change_threshold', 0.3)
        self.market_open_ignore_minutes = self.config.get('market_open_ignore_minutes', 10)
        self.poll_interval = self.config.get('poll_interval_seconds', 15) or self.config.get('poll_interval', 15)
        self.log_file = self.config.get('log_file', 'alerts_log.txt')
        self.json_output = self.config.get('output_file', 'anomalies.json') or self.config.get('json_output', 'anomalies.json')
        self.detection_method = self.config.get('detection_method', 'z_score') if tf else 'z_score'
        self.lstm_params = self.config.get('lstm_params', {})
        self.max_iterations = self.config.get('max_iterations', 3)
        self.anomalies = []
        self.last_data = {}
        self.scaler = MinMaxScaler() if tf else None
        self.lstm_model = None
    
    def fetch_stock_data(self, symbol):
        """
        Fetch stock price data from yfinance.
        
        Args:
            symbol (str): Stock ticker symbol.
        
        Returns:
            pd.Series: Closing prices or None if failed.
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=self.period, interval=self.interval, prepost=False)
            if data.empty:
                logger.error(f"No data retrieved for {symbol}")
                return None
            logger.info(f"Fetched {len(data)} data points for {symbol}")
            return data['Close']
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def validate_and_clean_data(self, data):
        """
        Validate and clean stock price data.
        
        Args:
            data (pd.Series): Raw stock price data.
        
        Returns:
            pd.Series: Cleaned data or None if invalid.
        """
        if data is None or len(data) < self.window:
            logger.error(f"Invalid or insufficient data: {len(data) if data is not None else 0} points")
            return None
        
        data = data.fillna(method='ffill').dropna()
        if len(data) < self.window:
            logger.error(f"Data too short after cleaning: {len(data)} points")
            return None
        
        logger.info(f"Cleaned data: {len(data)} points")
        return data
    
    def is_market_open_period(self, timestamp):
        """
        Check if timestamp is within market open ignore period.
        
        Args:
            timestamp (pd.Timestamp): Timestamp to check.
        
        Returns:
            bool: True if within ignore period.
        """
        try:
            timestamp = pd.Timestamp(timestamp)
            market_open = timestamp.replace(hour=9, minute=30)
            return timestamp < market_open + timedelta(minutes=self.market_open_ignore_minutes)
        except Exception as e:
            logger.error(f"Error checking market open period: {str(e)}")
            return False
    
    def compute_features(self, data):
        """
        Compute features: moving average and percentage change.
        
        Args:
            data (pd.Series): Stock price data.
        
        Returns:
            tuple: (moving average, percentage change)
        """
        moving_avg = data.rolling(window=self.window).mean()
        percent_change = data.pct_change() * 100
        logger.info(f"Data stats - Mean: {data.mean():.2f}, Std: {data.std():.2f}, Max % change: {percent_change.abs().max():.2f}%")
        return moving_avg, percent_change
    
    def prepare_lstm_data(self, data):
        """
        Prepare data for LSTM model.
        
        Args:
            data (pd.Series): Stock price data.
        
        Returns:
            tuple: (X, y, scaler) for training or (X, scaler) for prediction.
        """
        lookback = self.lstm_params.get('lookback', 10)
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X = []
        y = []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, self.scaler if len(y) > 0 else (X, self.scaler)
    
    def build_lstm_model(self):
        """
        Build and compile LSTM model.
        
        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential()
        model.add(Input(shape=(self.lstm_params.get('lookback', 10), 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def detect_anomalies_z_score(self, data):
        """
        Detect anomalies using z-scores and percentage change.
        
        Args:
            data (pd.Series): Stock price data.
        
        Returns:
            pd.DataFrame: Anomalous price points with Z-Score.
        """
        try:
            df = pd.DataFrame({'Close': data})
            df['MA'] = df['Close'].rolling(window=self.window).mean()
            df['Std'] = df['Close'].rolling(window=self.window).std()
            df['Z-Score'] = (df['Close'] - df['MA']) / df['Std'].replace(0, np.nan)
            df['Percent_Change'] = df['Close'].pct_change() * 100
            df = df.dropna()
            
            if df.empty or df['Std'].isna().all():
                logger.warning("Invalid data or zero standard deviation; skipping anomaly detection")
                return pd.DataFrame(columns=['Close', 'Z-Score'])
            
            anomalies = df[
                (df['Z-Score'].abs() > self.z_threshold) & 
                (df['Percent_Change'].abs() > self.percent_change_threshold)
            ]
            anomalies = anomalies[~anomalies.index.to_series().apply(self.is_market_open_period)]
            logger.info(f"Z-score anomalies detected: {len(anomalies)}")
            return anomalies[['Close', 'Z-Score']]
        except Exception as e:
            logger.error(f"Error in z-score anomaly detection: {str(e)}")
            return pd.DataFrame(columns=['Close', 'Z-Score'])
    
    def detect_anomalies_lstm(self, data, symbol):
        """
        Detect anomalies using LSTM predictions.
        
        Args:
            data (pd.Series): Stock price data.
            symbol (str): Stock ticker symbol.
        
        Returns:
            pd.DataFrame: Anomalous price points with Z-Score placeholder.
        """
        if not tf:
            logger.warning("TensorFlow not available; cannot use LSTM")
            return pd.DataFrame(columns=['Close', 'Z-Score'])
        
        lookback = self.lstm_params.get('lookback', 10)
        if len(data) < lookback + 1:
            logger.warning(f"Not enough data for LSTM for {symbol}: {len(data)} points")
            return pd.DataFrame(columns=['Close', 'Z-Score'])
        
        X, y, scaler = self.prepare_lstm_data(data)
        if len(y) == 0:
            logger.warning(f"Insufficient data for LSTM training for {symbol}")
            return pd.DataFrame(columns=['Close', 'Z-Score'])
        
        # Train or load model
        if symbol not in self.last_data or self.lstm_model is None:
            self.lstm_model = self.build_lstm_model()
            self.lstm_model.fit(
                X, y,
                epochs=self.lstm_params.get('epochs', 20),
                batch_size=self.lstm_params.get('batch_size', 32),
                verbose=0
            )
        
        # Predict
        predictions = self.lstm_model.predict(X, verbose=0)
        predictions = scaler.inverse_transform(predictions).flatten()
        actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        
        # Calculate errors
        errors = np.abs((actual - predictions) / actual)
        threshold = self.lstm_params.get('prediction_threshold', 0.003)
        logger.info(f"Max LSTM prediction error for {symbol}: {np.max(errors):.4f}")
        anomaly_indices = data.index[lookback:][errors > threshold]
        anomalies = pd.DataFrame({'Close': data.loc[anomaly_indices], 'Z-Score': errors[errors > threshold]})
        
        anomalies = anomalies[~anomalies.index.to_series().apply(self.is_market_open_period)]
        logger.info(f"LSTM anomalies detected: {len(anomalies)}")
        return anomalies
    
    def detect_anomalies(self, data, symbol):
        """
        Detect anomalies based on configured method.
        
        Args:
            data (pd.Series): Stock price data.
            symbol (str): Stock ticker symbol.
        
        Returns:
            pd.DataFrame: Anomalous price points with Close and Z-Score columns.
        """
        try:
            self.compute_features(data)  # Log data stats for debugging
            if self.detection_method == 'lstm':
                return self.detect_anomalies_lstm(data, symbol)
            else:
                return self.detect_anomalies_z_score(data)
        except Exception as e:
            logger.error(f"Error in anomaly detection for {symbol}: {str(e)}")
            return pd.DataFrame(columns=['Close', 'Z-Score'])
    
    def generate_alert(self, symbol, row):
        """
        Generate and save anomaly alert.
        
        Args:
            symbol (str): Stock ticker symbol.
            row (pd.Series): Anomaly data with Close and Z-Score.
        """
        timestamp = row.name.strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': round(row['Close'], 2),
            'z_score': round(row['Z-Score'], 2),
            'message': f"Anomaly detected for {symbol} at {timestamp}: Price = {row['Close']:.2f}, Z-Score = {row['Z-Score']:.2f}"
        }
        logger.info(f"[ALERT] {alert['message']}")
        with open(self.log_file, 'a') as f:
            f.write(alert['message'] + '\n')
        self.anomalies.append(alert)
    
    def save_anomalies(self):
        """
        Save anomalies to a JSON file or print if file output is not supported.
        """
        try:
            with open(self.json_output, 'w') as f:
                json.dump(self.anomalies, f, indent=4)
            logger.info(f"Anomalies saved to {self.json_output}")
        except Exception as e:
            logger.info("File output not supported; printing anomalies to console")
            print(json.dumps(self.anomalies, indent=4))
    
    def plot_data(self, symbol, data, anomalies):
        """
        Plot data with anomalies using Matplotlib.
        
        Args:
            symbol (str): Stock ticker symbol.
            data (pd.Series): Stock price data.
            anomalies (pd.DataFrame): Anomalous price points.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(data.index, data, label='Price', color='blue')
        if not anomalies.empty:
            plt.scatter(anomalies.index, anomalies['Close'], label='Anomaly', color='red')
        plt.title(f"{symbol} Price with Anomalies")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def monitor_stocks(self):
        """
        Monitor stocks in real-time with a polling loop.
        """
        print("📈 Real-time Stock Price Anomaly Detection")
        iteration = 0
        while self.max_iterations is None or iteration < self.max_iterations:
            logger.info(f"Polling iteration {iteration + 1}/{self.max_iterations or '∞'}")
            try:
                for symbol in self.symbols:
                    logger.info(f"Processing {symbol}")
                    
                    # Fetch data
                    raw_data = self.fetch_stock_data(symbol)
                    if raw_data is None:
                        continue
                    
                    # Incremental update for subsequent iterations
                    if iteration > 0 and symbol in self.last_data:
                        last_timestamp = self.last_data[symbol].index[-1]
                        # Fetch only new data
                        raw_data = raw_data[raw_data.index > last_timestamp]
                    
                    if symbol in self.last_data and not raw_data.empty:
                        data = pd.concat([self.last_data[symbol], raw_data])
                    else:
                        data = raw_data
                    
                    data = self.validate_and_clean_data(data)
                    if data is None:
                        continue
                    
                    self.last_data[symbol] = data
                    
                    # Detect anomalies
                    anomalies = self.detect_anomalies(data, symbol)
                    if anomalies.empty:
                        logger.info(f"No anomalies detected for {symbol}")
                    else:
                        for idx, row in anomalies.iterrows():
                            self.generate_alert(symbol, row)
                        self.plot_data(symbol, data, anomalies)
                
                # Save anomalies
                self.save_anomalies()
                
                # Wait before next poll
                if self.max_iterations is None or iteration < self.max_iterations - 1:
                    logger.info(f"Waiting {self.poll_interval} seconds before next poll")
                    time.sleep(self.poll_interval)
                iteration += 1
            except Exception as e:
                logger.error(f"Error in polling loop: {str(e)}")
                break

def main():
    """Entry point for the application."""
    detector = StockAnomalyDetector('config.json')
    detector.monitor_stocks()

if __name__ == "__main__":
    main()