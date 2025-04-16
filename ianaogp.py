import ollama

messages = [
    {
        "role": "system",
        "content": (
            "Sei un ingegnere ITALIANO di pista il cui compito è rispondere alle domande di un pilota alla guida di un kart che può parlare. "
            "In input ti darò diversi dati: ultimo tempo sul giro, miglior tempo sul giro, numero di giri fatti. "
            "In base ai dati, rispondi alle domande che ti faccio. "
            "Dammi risposte molto brevi, secche e precise del tipo: il tuo miglior giro è stato ...; il tuo ultimo tempo è stato ecc."
            "Rispondi con massimo 5 parole. Non usare frasi lunghe. Nessuna spiegazione."
        )
    }
]

def run(inp):
    messages.append({
        "role": "user",
        "content": inp
    })

    response = ollama.chat(
        model='phi3',
        messages=messages,
        options={"num_predict": 15}  # Limite token super basso
    )

    response_content = response['message']['content'].strip()
    print(response_content)

    messages.append({
        "role": "assistant",
        "content": response_content
    })

x = ""
while x != "bye":
    x = input("Tu: ")
    run(x)
