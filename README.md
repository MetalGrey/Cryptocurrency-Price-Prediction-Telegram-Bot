# 📈 Crypto Price Prediction Bot

A Telegram bot that predicts the price movement of a selected cryptocurrency (e.g., **SOLUSDT**) using an LSTM neural network. It sends updates every hour with a probability-based forecast and a graph visualization.  

---

## 📌 Features

- 📊 **LSTM Model** trained on historical OHLCV data and technical indicators (RSI, MACD, EMA, CCI, ADX)
- 🤖 **Telegram Bot** interface to receive predictions interactively
- 🕒 **Automated hourly predictions** using APScheduler
- 🖼️ **Visualization** of model predictions (growth/drop) on real price charts
- 🧠 Built using **Keras**, **PyBit**, **Pandas**, and **ta-lib**

---

## 🚀 How It Works

1. Loads the latest 1000 candles from Bybit via `pybit`
2. Adds technical indicators to the data
3. Trains an LSTM model to classify whether price will go **up** or **down** in the next 3 hours
4. Sends:
   - A text forecast (with probability)
   - A plotted chart image with prediction points

---

## 🤖 Bot Commands

| Command      | Description |
|--------------|-------------|
| `/start`     | Initializes the bot and saves your chat ID for hourly updates |
| `/predict`   | Sends a fresh prediction based on the latest market data |

---

## 🧠 Model Overview

- **Model Type:** LSTM (Long Short-Term Memory)
- **Input Data:** Scaled close prices, RSI, MACD, EMA12, EMA26, volume
- **Label:** Binary (1 = growth in 3 hours, 0 = drop or flat)
- **Layers:** LSTM + Dropout + BatchNorm + Dense
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 35
- **Batch Size:** 32

---

## 🛠️ Tech Stack

- Python 3.10+
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/)
- [Matplotlib](https://matplotlib.org/)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [PyBit](https://github.com/verata-veratae/pybit)
- [APScheduler](https://apscheduler.readthedocs.io/en/stable/)

---

## 📷 Example Output


  ![photo_2025-05-07_20-56-28](https://github.com/user-attachments/assets/37bf575b-c41c-48cd-8668-30cd1732bfbf)

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/MetalGrey/Cryptocurrency-Price-Prediction-Telegram-Bot.git
cd crypto-predict-bot

# Install dependencies
pip install -r requirements.txt

---

## 📋 Setup Requirements

Before running the bot, make sure to:

1. **Insert your Telegram bot token**  
   In the `server.py` file, replace this line:
   ```python
   TOKEN = 'YOUR_BOT_TOKEN_HERE'
