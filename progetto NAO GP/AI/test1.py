import joblib
import numpy as np
from tensorflow import keras

# Carica il modello addestrato
model = keras.models.load_model('feedback_model.h5')

# Carica lo scaler e l'encoder salvati
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Nuovi dati di esempio
new_data = np.array([[120, 3.5, 10, 0.2, 0.8]])  # Cambia questi valori per testare

# Normalizza i dati con lo stesso scaler usato nel training
new_data_scaled = scaler.transform(new_data)

# Fai la previsione
prediction = model.predict(new_data_scaled)
predicted_class = np.argmax(prediction)

# Decodifica il valore predetto
predicted_label = encoder.inverse_transform([predicted_class])[0]

print("Classe Predetta:", predicted_label)
