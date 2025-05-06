import paho.mqtt.client as mqtt

# Callback che viene chiamata al momento della connessione al broker.
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connesso al broker MQTT con successo!")
        # Sottoscrizione al topic che Unity usa per pubblicare
        client.subscribe("M2MQTT_Unity/test", qos=2)
    else:
        print("Connessione fallita con codice", rc)

# Callback per la ricezione dei messaggi
def on_message(client, userdata, msg):
    # Decodifica del payload (assumendo che sia in UTF-8)
    message = msg.payload.decode("utf-8")
    print(f"Messaggio ricevuto dal topic '{msg.topic}': {message}")

# Istanza del client MQTT
client = mqtt.Client()

# Imposta i callback
client.on_connect = on_connect
client.on_message = on_message

# Parametri di connessione:
# Inserisci qui l'indirizzo del broker MQTT e la porta.
# Se il broker si trova sulla stessa macchina, puoi usare "localhost" e la porta standard 1883.
broker_address = "broker.emqx.io"  # O sostituisci con l'indirizzo del tuo broker
broker_port = 1883

# Connessione al broker MQTT
client.connect(broker_address, broker_port, keepalive=60)

client.publish("M2MQTT_Unity/test", "Ciao da Python!", qos=1)

# Avvio del loop: questo metodo blocca l'esecuzione e gestisce tutte le operazioni di rete.
client.loop_forever()

