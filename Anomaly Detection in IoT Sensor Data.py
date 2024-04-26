import pandas as pd
from sklearn.ensemble import IsolationForest

# Load and preprocess IoT sensor data
data = pd.read_csv('iot_sensor_data.csv')

# Train Isolation Forest model
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(data)

# Predict anomalies
anomalies = clf.predict(data)

# Print detected anomalies
print(f'Detected anomalies: {anomalies}')
