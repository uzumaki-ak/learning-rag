import requests
import numpy as np
import faiss

with open("unstructured-data.txt", "r", encoding="utf-8") as file:
    d1 = file.read()

# ! Split text into chunks of max 100 words

def chunk_text(text, max_words=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks  
chunks = chunk_text(d1)
print(f"total chunks: {len(chunks)}")
print(chunks[0])
print(len(chunks[1]))



# !Embedding model


GEMINI_API_KEY = ""

def get_embedding(text, model="models/embedding-001"):
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    #  payload structure for Gemini embedding API
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
    
    #  Gemini returns embedding in this structure 
    response_data = response.json()
    embedding = response_data["embedding"]["values"]
    
    return np.array(embedding)

# Testing the output of the embedding function 
test_embedding = get_embedding(chunks[0])
print(test_embedding.shape)
print(len(test_embedding))


print("First 10 values:", test_embedding[:10])
print("Embedding vector:", test_embedding)

# ! Create FAISS index and add embeddings
dimension = test_embedding.shape[0]
# to store all info in faiss we willl creare inddex for test-emb to store in faiss 
index = faiss.IndexFlatL2(dimension)
chunk_mapping = []

# noqw wht we have to do i to convert entire chunk into their respective embeddings and store into the faiss index 
# tis will  give all the chunks one buy one 
for  chunk in chunks:
    # now call gett emb func and try to pass all thec hunks one by one 
    emb = get_embedding(chunk)
    print(emb)
    # now we have to add this emb to the faiss index and when adding we have to convert it into numpy array 
    index.add(np.array([emb]).astype('float32'))
# now we have to maintain the mapping of chunk to their respective embeddings append to the chunk mapping list 
    chunk_mapping.append(chunk)
    # now storing the db in the loal if not did this its stored in the ram 
    faiss.write_index(index, "faiss_index.index")