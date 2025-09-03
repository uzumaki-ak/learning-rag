with open("unstructured-data.txt", "r", encoding="utf-8") as file:
    d1 = file.read()

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
