import numpy as np
import joblib
from tensorflow import keras

# Carica tutto
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model = keras.models.load_model('feedback_model.h5')

# Esempio di nuovi dati
new_data = np.array([[0.5, 1.2, 0.7, 120, 3, -2, 0.1, 0.2, 115, 2.8, -1.5, 0.1, 0.3, 0.05, 0.02, 0.01, 0.03, 0.5, 20, 90]])

# Normalizza
new_data_scaled = scaler.transform(new_data)

# Predici
prediction = model.predict(new_data_scaled)
predicted_class = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_class])[0]

print(f'Feedback Predetto: {predicted_label}')
