import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Load stock data
data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')['Adj Close']

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
data_normalized = scaler.fit_transform(np.array(data).reshape(-1, 1))

# Prepare data
X, y = [], []
for i in range(len(data_normalized) - 60):
    X.append(data_normalized[i:i+60, 0])
    y.append(data_normalized[i+60, 0])

X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Create LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, batch_size=1, epochs=1)

# Predict future prices
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting the results can be done using libraries like Matplotlib
