import joblib
import numpy as np
from tensorflow import keras

# Carica il modello addestrato
model = keras.models.load_model('feedback_model.h5')

# Carica scaler e encoder salvati durante il preprocessing
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Nuovo esempio di dati (speed, acceleration, steering_angle, brake_pressure, throttle_input)
new_data = np.array([[120, 3.5, 10, 0.2, 0.8]])  # ← Modifica questi valori per testare diversi casi

# Applica lo stesso scaler del training
new_data_scaled = scaler.transform(new_data)

# Fai la previsione
prediction = model.predict(new_data_scaled)
predicted_class = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_class])[0]

print("✅ Classe Predetta:", predicted_label)
