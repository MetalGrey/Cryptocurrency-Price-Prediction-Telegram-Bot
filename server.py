# ============ ABOUT ME ============
# Project Name: Cryptocurrency Price Prediction Bot
# Author: MetalGrey (Sergey)
# Description: This bot predicts the price movements (growth or drop) of a given cryptocurrency 
# (e.g., SOLUSDT) using LSTM neural networks and provides updates via Telegram.
# The bot predicts price changes every hour and sends notifications to users.
# Technologies: Python, Keras, Telegram API, PyBit, Matplotlib, Pandas, Scikit-learn
# Purpose: The bot helps users make informed decisions about cryptocurrency investments.
# Last Updated: 07.05.2025

# ============ COMMANDS DESCRIPTION ============
# /start Command:
# Description: This command is used to start the interaction with the bot. 
# It sends a welcome message to the user and stores their chat ID for future use.
# Trigger: /start
# Response: "Hello! I'll send you the price prediction for {symbol} every hour ⏰📈."

# /predict Command:
# Description: This command triggers the bot to send the latest price prediction for the given cryptocurrency.
# It will send a prediction of whether the price of the cryptocurrency will increase or decrease in the next few hours.
# Trigger: /predict
# Response: A message indicating whether the price will go up or down along with the model's prediction accuracy.



from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from pybit.unified_trading import HTTP
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
import asyncio


# ============ Settings ⚙️ ============
TOKEN = ''
CHAT_ID = None 
symbol = 'SOLUSDT'

# ============ Logs 📜 ============
logging.basicConfig(level=logging.INFO)

# ============ SCHEDULE 📅 ============
async def send_prediction_to(chat_id, symbol):
    await app.bot.send_message(chat_id=CHAT_ID, text=f"\n⏳ Please wait a moment...")
    
    session = HTTP()
    data = session.get_kline(
        category="linear",
        symbol=symbol,
        interval="60",
        limit=1000
    )

    #DataFrame
    df = pd.DataFrame(data['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df.sort_values('timestamp')
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    
    df.dropna(inplace=True)  # Removing rows with NaN values 🧹

    features = ['close', 'rsi', 'macd', 'ema_12', 'ema_26', 'volume']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    window_size = 60
    
    #Creating a binary target variable (will there be growth in 3 hours?) 📈
    future_offset = 3  # how many hours ahead ⏰
    X, y = [], []

    for i in range(window_size, len(scaled_data) - future_offset):
        X.append(scaled_data[i - window_size:i])
        future_price = scaled_data[i + future_offset][0]
        current_price = scaled_data[i][0]
        label = 1 if future_price > current_price else 0  # Growth or not? 📊⬆
        y.append(label)

    X, y = np.array(X), np.array(y)

    # ============ MODEL 📊 ============
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Sgmoid

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]


    model.fit(X_train, y_train, epochs=35, batch_size=32, validation_data=(X_test, y_test))


    # Prediction: Growth or not? 📊⬆
    last_input = X[-1].reshape(1, X.shape[1], X.shape[2])
    probability_of_growth = model.predict(last_input)[0][0]

    if probability_of_growth > 0.5:
        print(f"📈 Price increase expected for {symbol} in {future_offset} hours! Probability: {probability_of_growth:.2f}")
        output = f"📈 Price increase expected for {symbol} in {future_offset} hours! Probability: {probability_of_growth:.2f}"
    else:
        print(f"📉 Price drop or sideways movement expected for {symbol}. Growth probability: {probability_of_growth:.2f}")
        output = f"📉 Price drop or sideways movement expected for {symbol}. Growth probability: {probability_of_growth:.2f}"

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten() 
    price_start_index = split + window_size + future_offset
    actual_prices = df['close'].values[price_start_index : price_start_index + len(y_test)]
    plt.figure(figsize=(15, 6))
    plt.plot(actual_prices, label=f'{symbol} Price (Close) 💰', color='black')


   # Mark model predictions 📍
    growth_indices = np.where(y_pred == 1)[0]
    fall_indices = np.where(y_pred == 0)[0]

    plt.scatter(growth_indices, actual_prices[growth_indices], marker='^', color='green', label='Prediction: Growth 📈', alpha=0.6)
    plt.scatter(fall_indices, actual_prices[fall_indices], marker='v', color='red', label='Prediction: Drop/Sideways 📉', alpha=0.6)

    plt.title(f"Model Predictions for {symbol} Growth (+{future_offset}h) on Test Data 📊")
    plt.xlabel("Time Steps ⏳")
    plt.ylabel(f"{symbol} Price 💵")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot.png', format='png', dpi=300)
    
    await app.bot.send_photo(
    chat_id=CHAT_ID,
    photo="plot.png",  #Path to file 💾
    caption="📊 Model Prediction Chart"
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=f"\n {output}")

async def send_prediction(CHAT_ID, symbol):
    if CHAT_ID is not None:
        await send_prediction_to(CHAT_ID, symbol)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_prediction_to(update.effective_chat.id, symbol)

# ============ Commands ============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hello! I'll send you the price prediction for {symbol} every hour ⏰📈.")
    global CHAT_ID
    CHAT_ID = update.effective_chat.id  # Save ID for distribution 💾📤

if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    
    #/predict and /start commadns
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))

    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: asyncio.run(send_prediction(CHAT_ID, symbol)), 'interval', minutes=60)
    scheduler.start()
    app.run_polling()