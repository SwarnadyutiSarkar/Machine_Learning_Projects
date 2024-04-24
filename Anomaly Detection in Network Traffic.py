import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load and preprocess network traffic data
data = pd.read_csv('network_traffic_data.csv')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Train Isolation Forest model
clf = IsolationForest(contamination=0.01)
clf.fit(data_scaled)

# Predict anomalies
predictions = clf.predict(data_scaled)
anomalies = data[predictions == -1]
