import pandas as pd
import numpy as np

# Imposta un seed per la riproducibilità
np.random.seed(42)

# Numero di campioni
num_samples = 1000

# Genera dati casuali realistici
speed = np.random.uniform(0, 150, num_samples)  # Velocità tra 0 e 150 km/h
acceleration = np.random.uniform(-5, 5, num_samples)  # Accelerazione tra -5 e 5 m/s^2
steering_angle = np.random.uniform(-30, 30, num_samples)  # Angolo sterzata tra -30° e 30°
brake_pressure = np.random.uniform(0, 1, num_samples)  # Pressione freno tra 0 e 1
throttle_input = np.random.uniform(0, 1, num_samples)  # Input acceleratore tra 0 e 1

# Genera feedback basato su condizioni simulate
def classify_feedback(speed, acceleration, steering_angle, brake_pressure, throttle_input):
    if speed > 120 and throttle_input > 0.8:
        return "pericoloso"
    elif brake_pressure > 0.7 and speed < 20:
        return "cautela"
    elif acceleration > 4 and throttle_input > 0.7:
        return "aggressivo"
    else:
        return "normale"

feedback = [classify_feedback(s, a, sa, bp, ti) for s, a, sa, bp, ti in zip(speed, acceleration, steering_angle, brake_pressure, throttle_input)]

# Creazione del DataFrame
data = pd.DataFrame({
    'speed': speed,
    'acceleration': acceleration,
    'steering_angle': steering_angle,
    'brake_pressure': brake_pressure,
    'throttle_input': throttle_input,
    'feedback': feedback
})

# Salva il dataset in un file CSV
data.to_csv("telemetry_data.csv", index=False)

print("Dataset creato e salvato come 'telemetry_data.csv'")
