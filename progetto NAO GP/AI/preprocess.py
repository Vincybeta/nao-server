import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Carica il dataset
data = pd.read_csv('AI/telemetry_data.csv')

# Seleziona le feature e il target
features = data[['speed', 'acceleration', 'steering_angle', 'brake_pressure', 'throttle_input']].values
labels = data['feedback'].values

# Codifica le etichette in valori numerici
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Suddividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Normalizza le feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Preprocessing completato e scaler salvato!")
