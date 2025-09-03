# Prompt building functions

def build_prompt(context_chunks, query):
    """
    Build prompt for LLM with context and query
    """
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.
context:
{context}
Question: 
{query}
Answer:"""