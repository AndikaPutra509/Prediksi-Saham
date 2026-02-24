# Prediksi Saham
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
    Bidirectional, Attention, Input
)
from tensorflow.keras.models import Model

data = yf.download("BBCA.JK", start="2018-01-01", end="2026-02-24")
data = data[['Open','High','Low','Close','Volume']]
data.tail(10)

close = data['Close'].squeeze()
high = data['High'].squeeze()
low = data['Low'].squeeze()
volume = data['Volume'].squeeze()

data['RSI'] = ta.momentum.RSIIndicator(close).rsi()

data['MA20'] = close.rolling(20).mean()
data['MA50'] = close.rolling(50).mean()

data['MACD'] = ta.trend.MACD(close).macd()

bb = ta.volatility.BollingerBands(close)
data['BB_high'] = bb.bollinger_hband()
data['BB_low'] = bb.bollinger_lband()

data['ATR'] = ta.volatility.AverageTrueRange(
    high, low, close).average_true_range()

data['OBV'] = ta.volume.OnBalanceVolumeIndicator(
    close, volume).on_balance_volume()

data = data.dropna()

data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

data = data.dropna()

X = []
y = []

window = 60

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(data['Target'].iloc[i])

X = np.array(X)
y = np.array(y)

split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

features = [
    'Open','High','Low','Close','Volume',
    'RSI','MA20','MA50','MACD','BB_high','BB_low','ATR','OBV'
]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(data[features])

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
x = MaxPooling1D(pool_size=2)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

attention = Attention()([x,x])

x = LSTM(32)(attention)

x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

loss, acc = model.evaluate(X_test,y_test)

print(loss, acc)

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_test,y_pred))

import matplotlib.pyplot as plt

plt.plot(y_test,label='Actual')
plt.plot(y_pred,label='Prediction')

plt.legend()
plt.show()
