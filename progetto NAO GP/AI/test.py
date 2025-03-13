from tensorflow import keras
import numpy as np

model = keras.models.load_model('feedback_model.h5')

def predict_feedback(new_data):
    # new_data deve essere un array NumPy con la forma (1, numero_di_feature)
    new_data = scaler.transform(new_data)  # Ricorda di applicare lo stesso scaling usato in training
    prediction = model.predict(new_data)
    feedback_index = np.argmax(prediction)
    # Ritorna il feedback in forma testuale usando l'encoder inverso
    return encoder.inverse_transform([feedback_index])[0]