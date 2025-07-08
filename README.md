techsophy/
├── main.py # Main entry point
├── config.json # Configuration (symbols, thresholds, etc.)
├── alerts_log.txt # Alert logs
├── anomalies.json # Output anomalies
├── README.md # Project description
└── requirements.txt # Python dependencies

yaml
Copy
Edit

---

## ⚙️ Configuration (`config.json`)

Customize stock tickers, thresholds, polling interval, and detection method:

```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "period": "1d",
  "interval": "1m",
  "window": 5,
  "z_threshold": 1.2,
  "percent_change_threshold": 0.3,
  "detection_method": "z_score",
  "poll_interval": 15,
  "max_iterations": 3
}
🧪 How to Run
1️⃣ Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Create Config File
Edit or create a file named config.json based on the sample above.

3️⃣ Run the Application
bash
Copy
Edit
python main.py
🔬 Detection Logic
📌 Z-Score Method
An anomaly is detected if:

Z-score > z_threshold

% price change > percent_change_threshold

Not within the first N minutes after market opens (configurable)

🧠 LSTM (Optional)
If TensorFlow is installed:

Trains an LSTM on past price windows

Predicts next price

Flags as anomaly if prediction error exceeds a threshold

📊 Output
🔴 Alerts printed in console and saved to alerts_log.txt

📁 Anomalies saved to anomalies.json

📈 Visual plot of price and anomalies shown using matplotlib

⚠️ Notes
If TensorFlow is not available, it gracefully falls back to Z-score method

Internet connection is required for real-time stock data

You can increase max_iterations or set it to null for infinite loop

👤 Author
Harsha Vardhan Reddy
Techsophy Placement Assessment – July 2025

🪪 License
This project is confidential and intended for assessment purposes only.
Do not redistribute without permission.

yaml
Copy
Edit

---

### ✅ Next Steps:
- Save this as `README.md` inside your project folder.
- Replace `"Harsha Vardhan Reddy"` with your actual name if needed.
- Push changes to GitHub:
  ```bash
  git add README.md
  git commit -m "Add detailed README"
  git push
Let me know if you want help generating the config.json or requirements.txt as well.








You said:
give me  full fledged readme file


ChatGPT said:
Here is a complete, professional, and submission-ready README.md file for your Techsophy Real-Time Stock Price Anomaly Detector project. It includes everything: intro, features, setup, usage, architecture, and interview-ready language.

✅ Save the following content as README.md in your techsophy project folder:
markdown
Copy
Edit
# 📈 Real-Time Stock Price Anomaly Detector

A Python application that monitors real-time stock prices using public APIs and detects unusual or abnormal price patterns using statistical anomaly detection methods. This project is built as part of the Techsophy coding assessment.

---

## 🎯 Objective

To detect anomalies (abnormal stock price movements) in real-time using a modular and extensible Python system. Alerts are generated when a stock's price deviates significantly from its recent average.

---

## 🧠 Features

- ✅ Real-time stock monitoring (1-minute intervals)
- 📊 Anomaly detection using:
  - Moving Average
  - Z-score Thresholding
- 📈 Plots live stock price data and highlights anomalies
- 🧾 Logs alerts to `alerts_log.txt`
- 💡 Easily extendable to include LSTM/AI-based detection (optional)
- 🧹 Includes data validation and error handling
- ⚙️ Modular code structure: fetch → validate → analyze → alert → visualize

---

## 🧰 Technologies Used

| Purpose        | Technology            |
|----------------|------------------------|
| Data Fetching  | `yfinance`             |
| Processing     | `pandas`, `numpy`      |
| Visualization  | `matplotlib`           |
| Logging        | Python file I/O        |
| Optional (DL)  | `tensorflow`, `sklearn`|

---

## 🗂️ Folder Structure

techsophy/
├── main.py # Main application file
├── requirements.txt # Python dependencies
├── alerts_log.txt # Log of anomaly alerts
├── README.md # Project overview

yaml
Copy
Edit

---

## ⚙️ How It Works

1. **Fetches** live 1-minute stock data using `yfinance`
2. **Cleans** the data (removes nulls)
3. **Calculates** moving average and standard deviation (rolling window)
4. **Computes Z-score** to identify outliers (price deviations)
5. **Flags anomalies** where Z-score exceeds threshold
6. **Logs and displays** the alerts and plots

---

## 🧪 Anomaly Detection Logic

An anomaly is triggered when:
- Z-score > `Z_THRESHOLD` (e.g., 2.5)
- Stock price has deviated from its mean by significant magnitude

Mathematically:
```python
z_score = (current_price - moving_average) / std_deviation
🔧 Configuration
You can adjust the following in main.py:

python
Copy
Edit
TICKERS = ['AAPL', 'MSFT']
INTERVAL = 60             # time (seconds) between checks
WINDOW = 20               # moving average window
Z_THRESHOLD = 2.5         # Z-score threshold for alert
📥 Installation & Setup
✅ 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Harsha120903/22211a6762_Techsophy.git
cd 22211a6762_Techsophy
✅ 2. Create Virtual Environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
✅ 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ 4. Run the App
bash
Copy
Edit
python main.py
📊 Sample Output
Console:

sql
Copy
Edit
📈 Real-time Stock Price Anomaly Detection
[ALERT] AAPL | Price: 192.33 | Z: 2.71
Log File (alerts_log.txt):

yaml
Copy
Edit
2025-07-08 10:35:04 | AAPL | Price: 192.33 | Z: 2.71
Plot:
Shows live price curve with anomalies in red.

🚀 Future Enhancements
✅ Save plots as PNG files

✅ Email/SMS alert integration

✅ Add LSTM-based deep learning anomaly detection

✅ Store time series data for long-term trend analysis

✅ Deploy on cloud (AWS Lambda / GCP)

🙋 Author
Harsha Vardhan Reddy
Dept. of CSE - BVRIT
GitHub: Harsha120903

🛡️ License
This project is developed as part of Techsophy Placement Assessment and is confidential.