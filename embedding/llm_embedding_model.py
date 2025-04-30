import numpy as np
import requests

# Configuration for the Ollama model API
OLLAMA_HOST = "http://192.168.0.225:11434"  # Adjust the port if necessary
MODEL_NAME = "nomic-embed-text"  # Replace with your actual model name

def get_llm_embedding(text):
    """
    Calls the Ollama model API to get an embedding for the input text.
    Assumes the API endpoint is available at {OLLAMA_HOST}/embed and that it expects
    a JSON payload with "model" and "text" keys, returning a JSON with an "embedding" key.
    """
    endpoint = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()  # Raise an error for bad status codes.
    data = response.json()
    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError("No 'embedding' key found in the response.")
    return np.array(embedding)

