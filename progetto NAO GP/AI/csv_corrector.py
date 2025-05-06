import pandas as pd

# Carica il file come testo normale
with open('Telemetry_Giro_010.csv', 'r') as file:
    lines = file.readlines()

# Dividi ogni riga usando il PUNTO SOLO UNA VOLTA
dati = [line.strip().split('.') for line in lines]

# Crea un DataFrame
df = pd.DataFrame(dati)

# Ora unisci i pezzi giusti: prendi solo le prime 22 colonne
df = df.iloc[:, :22]

# Dai i nomi corretti alle colonne
df.columns = [
    'Time', 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ',
    'Steering', 'Throttle', 'Brake', 'LocalVelX', 'LocalVelY', 'LocalVelZ',
    'AccLat', 'AccLong', 'AccVert', 'Yaw', 'Pitch', 'Roll',
    'DistFromIdeal', 'Curva', 'LapTime'
]

# Salva il nuovo file corretto
df.to_csv('Telemetry_Giro_010_corretto.csv', index=False)

print("File corretto salvato come 'Telemetry_Giro_010_corretto.csv'")
