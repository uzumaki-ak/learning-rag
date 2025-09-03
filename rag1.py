# import requests
# import numpy as np
# import faiss

# with open("unstructured-data.txt", "r", encoding="utf-8") as file:
#     d1 = file.read()

# # ! Split text into chunks of max 100 words

# def chunk_text(text, max_words=100):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words):
#         chunks.append(" ".join(words[i:i+max_words]))
#     return chunks  
# chunks = chunk_text(d1)
# print(f"total chunks: {len(chunks)}")
# print(chunks[0])
# print(len(chunks[1]))



# # !Embedding model


# GEMINI_API_KEY = ""

# def get_embedding(text, model="models/embedding-001"):
#     url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={GEMINI_API_KEY}"
    
#     headers = {
#         "Content-Type": "application/json"
#     }
    
#     #  payload structure for Gemini embedding API
#     payload = {
#         "model": model,
#         "taskType": "RETRIEVAL_DOCUMENT",
#         "content": {
#             "parts": [
#                 {
#                     "text": text
#                 }
#             ]
#         }
#     }
    
#     response = requests.post(url, headers=headers, json=payload)
    
#     if response.status_code != 200:
#         raise Exception(f"Error {response.status_code}: {response.text}")
    
#     #  Gemini returns embedding in this structure 
#     response_data = response.json()
#     embedding = response_data["embedding"]["values"]
    
#     return np.array(embedding)

# # Testing the output of the embedding function 
# test_embedding = get_embedding(chunks[0])
# print(test_embedding.shape)
# print(len(test_embedding))


# print("First 10 values:", test_embedding[:10])
# print("Embedding vector:", test_embedding)

# # ! Create FAISS index and add embeddings
# dimension = test_embedding.shape[0]
# # to store all info in faiss we willl creare inddex for test-emb to store in faiss 
# index = faiss.IndexFlatL2(dimension)
# chunk_mapping = []

# # noqw wht we have to do i to convert entire chunk into their respective embeddings and store into the faiss index 
# # tis will  give all the chunks one buy one 
# for  chunk in chunks:
#     # now call gett emb func and try to pass all thec hunks one by one 
#     emb = get_embedding(chunk)
#     print(emb)
#     # now we have to add this emb to the faiss index and when adding we have to convert it into numpy array 
#     index.add(np.array([emb]).astype('float32'))
# # now we have to maintain the mapping of chunk to their respective embeddings append to the chunk mapping list 
#     chunk_mapping.append(chunk)
#     # now storing the db in the loal if not did this its stored in the ram 
#     faiss.write_index(index, "faiss_index.index")

#     # performing a test search 
# def retrirve_top_k(query,k=3):
#         # we took query as input and k as the number of top similar chunks we want to retrieve and passed itot the get emb func to convert it into embedding
#         query_emb = get_embedding(query)
#         # now we have to search this emb in the faiss index
#         distance , indices = index.search(np.array([query_emb]).astype('float32'), k)
#         # now we have to return the top k chunks based on the indices we got from the search 
#         return [chunk_mapping[i] for i in indices[0]]
#     #   now we will prepare a prompt for the llm to generate the answer based on the retrieved chunks 
# def build_prompt(context_chunks,query):
#         context = "\n\n".join(context_chunks)
#         return f"""You are a helpful assistant. Use the following context to answer the question.
#     context:
#     {context}
#     Question: 
#     {query}
#     Answer:"""
#     # now we will call the retrive top k func to get the top 3 chunks   
    
# def generate_ans(prompt,model="models/gemini-1.5-flash"):
#         url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateText?key={GEMINI_API_KEY}"
#         headers = {
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": model,
#             "prompt": {
#                 "text": prompt
#             },
#             "temperature": 0.2,
#             "maxOutputTokens": 1024
#         }
#         response = requests.post(url, headers=headers, json=payload)
#         if response.status_code != 200:
#             raise Exception(f"Error {response.status_code}: {response.text}")
#         response_data = response.json()
#         return response_data["candidates"][0]["output"]
# query = "tell me who is uzumaki-ak ?"
# top_k_chunks = retrirve_top_k(query, k=3)
# print("\n===Retrieved Chunks===\n")
# print(prompt[:500]+"..." if len(prompt) > 500 else prompt)


# //ORIGINAL CODE 
import requests
import numpy as np
import faiss
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

#now performing a test search
def retrirve_top_k(query,k=3):
    # we took query as input and k as the number of top similar chunks we want to retrieve and passed itot the get emb func to convert it into embedding
    query_emb = get_embedding(query)
    # now we have to search this emb in the faiss index
    distance , indices = index.search(np.array([query_emb]).astype('float32'), k)
    # now we have to return the top k chunks based on the indices we got from the search 
    return [chunk_mapping[i] for i in indices[0]]

#   now we will prepare a prompt for the llm to generate the answer based on the retrieved chunks 
def build_prompt(context_chunks,query):
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.
context:
{context}
Question: 
{query}
Answer:"""

# now we will call the retrive top k func to get the top 3 chunks   
def generate_ans(prompt,model="models/gemini-1.5-flash "):
    
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")
    response_data = response.json()
    return response_data["candidates"][0]["content"]["parts"][0]["text"]

# ACTUALLY CALL THE FUNCTIONS TO GET THE ANSWER
query = "tell me who is uzumaki-ak ?"
# this will return the top 3 chunks 
top_k_chunks = retrirve_top_k(query, k=3)
print("\n=== RETRIEVED CHUNKS ===")

# Display each retrieved chunk individually for clarity 
for i, chunk in enumerate(top_k_chunks):
    print(f"Chunk {i+1}: {chunk[:100]}...")  # Show first 100 chars

# it returns the prompt 
prompt = build_prompt(top_k_chunks, query)
print("\n=== GENERATED PROMPT ===")
# Display the prompt clearly 
print(prompt[:500] + "..." if len(prompt) > 500 else prompt)  # Show first 500 chars

# now we will call the generate ans func to get the final answer from the llm
answer = generate_ans(prompt)
print("\n=== FINAL ANSWER ===")
print(answer)