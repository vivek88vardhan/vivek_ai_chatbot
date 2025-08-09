import requests

def query_ollama(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama2",
        "prompt": prompt
    })
    return response.json()["response"]
