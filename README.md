# 📈 Real-Time Stock Price Anomaly Detector

This project is part of the **Techsophy Coding Assessment**. It is a Python-based real-time anomaly detection system for stock prices using live financial data. The system fetches stock prices at regular intervals and uses statistical methods to detect abnormal price movements (anomalies).

---

## 🎯 Objective

To design a real-time application that:
- Monitors live stock prices
- Detects unusual price changes using statistical thresholds
- Alerts the user via console and logs
- Visualizes the anomalies in price charts

---

## 🧠 Features

- ✅ Real-time monitoring of multiple stock tickers
- 📊 Statistical anomaly detection using:
  - Moving Average (MA)
  - Standard Deviation (STD)
  - Z-score thresholding
- 🔔 Console alerts for anomalies
- 🧾 Logs alerts to a text file (`alerts_log.txt`)
- 📈 Anomaly plots using `matplotlib`
- 🧹 Clean and modular code (easy to extend to LSTM in future)

---

## 🧰 Tech Stack

| Component       | Library       |
|----------------|---------------|
| Data Source     | `yfinance`    |
| Data Analysis   | `pandas`, `numpy` |
| Visualization   | `matplotlib`  |
| Logging         | Python file I/O |
| Language        | Python 3.x    |





