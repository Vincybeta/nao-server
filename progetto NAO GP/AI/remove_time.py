import pandas as pd

# Carica il CSV
file_path = "Telemetry_Giro_010_fixed.csv"
data = pd.read_csv(file_path)

# Rimuove la prima colonna (assumendo che sia "Time")
data = data.drop(columns=data.columns[0])

# Salva il nuovo CSV
data.to_csv("Telemetry_Giro_010_noTime.csv", index=False)

print("Colonna 'Time' rimossa e nuovo file salvato come 'Telemetry_Giro_003_noTime.csv'")
