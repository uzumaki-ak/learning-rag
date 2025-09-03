import requests
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# !Embedding model functions



def get_embedding(text, model="models/embedding-001"):
    """
    Get embedding for text using Gemini API
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # payload structure for Gemini embedding API
    payload = {
        "model": model,
        "taskType": "RETRIEVAL_DOCUMENT",
        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    
    # Gemini returns embedding in this structure 
    response_data = response.json()
    embedding = response_data["embedding"]["values"]
    
    return np.array(embedding)