import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tf2onnx
import tensorflow as tf
import os
import pickle
import json

# === 1. Carica il dataset ===
data1 = pd.read_csv('Telemetry_Giro_010_noTime.csv')
data2 = pd.read_csv('Telemetry_Giro_003_noTime.csv')
data = pd.concat([data1, data2], ignore_index=True)

# === 2. Prepara Input (X) e Output (y) ===
X = data[[ 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ', 
           'LocalVelX', 'LocalVelY', 'LocalVelZ', 'AccLat', 'AccLong', 'AccVert',
           'Yaw', 'Pitch', 'Roll', 'DistFromIdeal']].values

y = data[['Steering', 'Throttle', 'Brake']].values

# === 3. Normalizza ===
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

# === 4. Salvataggio scaler ===
os.makedirs("model", exist_ok=True)

def save_standard_scaler_to_json(scaler, filename):
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(scaler_data, f, indent=4)

save_standard_scaler_to_json(scaler_x, "model/standard_scaler_x.json")
save_standard_scaler_to_json(scaler_y, "model/standard_scaler_y.json")

with open("model/scaler_x.pkl", "wb") as f:
    pickle.dump(scaler_x, f)
with open("model/scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# === 5. Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Crea il modello migliorato ===
model = keras.Sequential([
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),

    layers.Dense(32, activation='relu'),
    layers.Dense(3)  # Uscita: Steering, Throttle, Brake
])

# === 7. Compila ===
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

# === 8. Callback migliorati ===
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# === 9. Allena ===
history = model.fit(
    X_train, y_train, 
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# === 10. Valuta ===
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# === 11. Grafico ===
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.title('Andamento della Loss durante l\'allenamento')
plt.savefig("model/loss_plot.png")
plt.show()

# === 12. Predizioni e denormalizzazione ===
y_pred = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# === 13. Conversione in ONNX ===
spec = (tf.TensorSpec((None, X_train.shape[1]), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

onnx_path = "model/expert_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Modello salvato in 'model/expert_model.h5' e 'model/expert_model.onnx'")
print("Scaler salvato in 'model/scaler_x.pkl' e 'model/scaler_y.pkl'")
