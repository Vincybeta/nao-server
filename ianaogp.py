import ollama

response = ollama.chat(model='phi3',messages=[
    {
        "role":"system",
        "content": "sei un ingeniere di pista il cui compito è quello di rispondere alle domande di un pilota alla guida di un kart che può parlare "

    },
    {
        "role":"user",
        "content": "come faccio a migliorare le mie curve quando il tracciato è bagnato"
    },
    ])
print(response['message']['content'].strip())