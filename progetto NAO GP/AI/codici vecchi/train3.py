import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib

# Carica i dati
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Carica encoder
encoder = joblib.load('encoder.pkl')
num_classes = len(encoder.classes_)

# Costruisci il modello
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compila
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestra
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Valuta
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Accuratezza sul test: {test_accuracy:.2f}')

# Salva il modello
model.save('feedback_model.h5')
print("Modello salvato!")
