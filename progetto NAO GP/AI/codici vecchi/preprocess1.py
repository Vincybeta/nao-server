import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Carica il dataset
data = pd.read_csv('Telemetry_Giro_010_modificato.csv')

# Crea nuova colonna "Feedback" basata su semplici regole
def generate_feedback(row):
    if row['Curva'] > 30 and row['Speed'] > 50:
        return "Rallenta in curva"
    elif row['Brake'] > 0.7:
        return "Frenata brusca"
    elif abs(row['Steering']) > 0.5 and row['Speed'] > 70:
        return "Sterza troppo a velocità alta"
    else:
        return "Guida corretta"

data['Feedback'] = data.apply(generate_feedback, axis=1)

# Seleziona le feature (input) e il target (output)
feature_columns = ['PosX', 'PosY', 'PosZ', 
                   'VelX', 'VelY', 'VelZ',
                   'Steering', 'Brake',
                   'LocalVelX', 'LocalVelY', 'LocalVelZ',
                   'AccLat', 'AccLong', 'AccVert',
                   'Yaw', 'Pitch', 'Roll',
                   'DistFromIdeal', 'Curva', 'LapTime']

X = data[feature_columns].values
y = data['Feedback'].values

# Codifica le etichette dei feedback
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Dividi in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalizza le feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salva tutto
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Preprocessing completato e file salvati!")
