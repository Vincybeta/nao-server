import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np

# Carica i dati preprocessati
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Determina il numero di classi (feedback unici)
encoder = joblib.load('encoder.pkl')
num_classes = len(encoder.classes_)

# Crea il modello
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestra il modello
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Valuta il modello sui dati di test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', test_accuracy)

# Salva il modello
model.save('feedback_model.h5')

print("Modello salvato come 'feedback_model.h5'")
