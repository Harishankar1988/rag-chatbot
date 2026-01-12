import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

def ollama_generate(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"
