import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load electricity consumption data
data = pd.read_csv('electricity_consumption_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.resample('D').sum()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data
X, y = [], []
for i in range(len(data_normalized) - 30):
    X.append(data_normalized[i:i+30, 0])
    y.append(data_normalized[i+30, 0])

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

# Predict future consumption
predicted_consumption = model.predict(X)
predicted_consumption = scaler.inverse_transform(predicted_consumption)

# Plotting the results can be done using libraries like Matplotlib
