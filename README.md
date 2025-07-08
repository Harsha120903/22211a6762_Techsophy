# 📈 Real-Time Stock Price Anomaly Detector

This project is a real-time system that monitors stock prices and detects anomalies using **statistical methods (Z-score)** and optionally **deep learning (LSTM)**. It fetches live stock data using the `yfinance` API and alerts when unusual price movements are detected.

---

## 🚀 Features

- ✅ Real-time stock monitoring (1-minute intervals)
- 📊 Anomaly detection using:
  - **Z-score thresholding**
  - **Percentage change filters**
  - **LSTM-based prediction (if TensorFlow is available)**
- 🧠 Automatically selects Z-score if TensorFlow is unavailable
- 🧹 Data cleaning and validation pipeline
- 📉 Plots anomalies on price curves using `matplotlib`
- 🧾 Logs alerts to `alerts_log.txt`
- 💾 Saves anomaly data in JSON format (`anomalies.json`)
- 🔧 Configurable via `config.json`

---

## 🛠 Technologies Used

| Area              | Tech Stack                          |
|-------------------|--------------------------------------|
| Language          | Python 3.x                           |
| API               | yfinance                             |
| Data Analysis     | pandas, numpy                        |
| Visualization     | matplotlib                           |
| Logging & Output  | logging, JSON, file I/O              |
| Deep Learning     | TensorFlow, LSTM (optional)          |
| Preprocessing     | scikit-learn (MinMaxScaler)          |

---

## 📂 Folder Structure

