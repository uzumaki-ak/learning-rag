# import numpy as np
# import faiss
# import pickle
# import os
# from utils.chunking import chunk_text
# from utils.embedding import get_embedding
# from utils.prompt import build_prompt
# from utils.completion import generate_ans

# # Load data
# with open("data/unstructured-data.txt", "r", encoding="utf-8") as file:
#     d1 = file.read()

# # ! Split text into chunks of max 100 words
# chunks = chunk_text(d1)
# print(f"total chunks: {len(chunks)}")
# print(chunks[0])
# print(len(chunks[1]))

# # Testing the output of the embedding function 
# test_embedding = get_embedding(chunks[0])
# print(test_embedding.shape)
# print(len(test_embedding))

# print("First 10 values:", test_embedding[:10])
# print("Embedding vector:", test_embedding)

# # ! Create FAISS index and add embeddings
# dimension = test_embedding.shape[0]
# # to store all info in faiss we will create index for test-emb to store in faiss 
# index = faiss.IndexFlatL2(dimension)
# chunk_mapping = []

# # Check if FAISS index already exists
# faiss_index_path = "faiss_store/index.faiss"
# mapping_path = "faiss_store/chunk_mapping.pkl"

# if os.path.exists(faiss_index_path) and os.path.exists(mapping_path):
#     print("Loading existing FAISS index...")
#     index = faiss.read_index(faiss_index_path)
#     with open(mapping_path, 'rb') as f:
#         chunk_mapping = pickle.load(f)
# else:
#     print("Creating new FAISS index...")
#     # Create directory if it doesn't exist
#     os.makedirs("faiss_store", exist_ok=True)
    
#     # Convert all chunks into their respective embeddings and store into the faiss index
#     for chunk in chunks:
#         # Get embedding for each chunk
#         emb = get_embedding(chunk)
#         print(emb)
#         # Add this emb to the faiss index (convert to numpy array)
#         index.add(np.array([emb]).astype('float32'))
#         # Maintain the mapping of chunk to their respective embeddings
#         chunk_mapping.append(chunk)
    
#     # Store the index and mapping
#     faiss.write_index(index, faiss_index_path)
#     with open(mapping_path, 'wb') as f:
#         pickle.dump(chunk_mapping, f)

# # Function to retrieve top k chunks
# def retrieve_top_k(query, k=3):
#     """
#     Retrieve top k similar chunks for a query
#     """
#     # Convert query to embedding
#     query_emb = get_embedding(query)
#     # Search this emb in the faiss index
#     distance, indices = index.search(np.array([query_emb]).astype('float32'), k)
#     # Return the top k chunks based on the indices from search 
#     return [chunk_mapping[i] for i in indices[0]]

# # ACTUALLY CALL THE FUNCTIONS TO GET THE ANSWER
# query = "tell me who is uzumaki-ak ?"
# # This will return the top 3 chunks 
# top_k_chunks = retrieve_top_k(query, k=3)
# print("\n=== RETRIEVED CHUNKS ===")

# # Display each retrieved chunk individually for clarity 
# for i, chunk in enumerate(top_k_chunks):
#     print(f"Chunk {i+1}: {chunk[:100]}...")  # Show first 100 chars

# # Build the prompt
# prompt = build_prompt(top_k_chunks, query)
# print("\n=== GENERATED PROMPT ===")
# # Display the prompt clearly 
# print(prompt[:500] + "..." if len(prompt) > 500 else prompt)  # Show first 500 chars

# # Generate the final answer from the LLM
# answer = generate_ans(prompt)
# print("\n=== FINAL ANSWER ===")
# print(answer)

import numpy as np
import faiss
import pickle
import os
import streamlit as st
from utils.chunking import chunk_text
from utils.embedding import get_embedding
from utils.prompt import build_prompt
from utils.completion import generate_ans
import shutil




# Streamlit page configuration
st.set_page_config(
    page_title="RAG System - Query Your Documents",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🔍 RAG Document Query System</h1>', unsafe_allow_html=True)
st.write("Ask questions about your documents using Retrieval Augmented Generation (RAG) technology.")

# Initialize session state
if 'faiss_initialized' not in st.session_state:
    st.session_state.faiss_initialized = False
    st.session_state.initialization_error = None

def cleanup_faiss_files():
    """Remove corrupted FAISS files"""
    faiss_dir = "faiss_store"
    if os.path.exists(faiss_dir):
        shutil.rmtree(faiss_dir)
    os.makedirs(faiss_dir, exist_ok=True)

def initialize_faiss():
    """Initialize or load FAISS index with proper error handling"""
    try:
        # Clean up any corrupted files first
        cleanup_faiss_files()
        
        # Load data
        with open("data/unstructured-data.txt", "r", encoding="utf-8") as file:
            d1 = file.read()

        # Split text into chunks
        chunks = chunk_text(d1)
        
        # Get dimension from first chunk
        test_embedding = get_embedding(chunks[0])
        dimension = test_embedding.shape[0]
        index = faiss.IndexFlatL2(dimension)
        chunk_mapping = []

        # Progress bar for embedding generation
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Convert all chunks into embeddings
        for i, chunk in enumerate(chunks):
            status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
            progress_bar.progress((i + 1) / len(chunks))
            
            emb = get_embedding(chunk)
            index.add(np.array([emb]).astype('float32'))
            chunk_mapping.append(chunk)
        
        # Store the index and mapping
        faiss_index_path = "faiss_store/index.faiss"
        mapping_path = "faiss_store/chunk_mapping.pkl"
        
        faiss.write_index(index, faiss_index_path)
        with open(mapping_path, 'wb') as f:
            pickle.dump(chunk_mapping, f)
        
        progress_bar.empty()
        status_text.empty()
        
        # Store in session state
        st.session_state.index = index
        st.session_state.chunk_mapping = chunk_mapping
        st.session_state.faiss_initialized = True
        st.session_state.total_chunks = len(chunk_mapping)
        st.session_state.initialization_error = None
        
        st.success("✅ FAISS index created successfully!")
        return True
        
    except Exception as e:
        st.session_state.initialization_error = str(e)
        st.error(f"❌ Error initializing FAISS: {str(e)}")
        return False

def retrieve_top_k(query, k=3):
    """Retrieve top k similar chunks for a query"""
    if not st.session_state.faiss_initialized:
        return [], []
    
    try:
        # Convert query to embedding
        query_emb = get_embedding(query)
        # Search in the faiss index
        distance, indices = st.session_state.index.search(
            np.array([query_emb]).astype('float32'), k
        )
        # Return the top k chunks
        return [st.session_state.chunk_mapping[i] for i in indices[0]], distance[0]
    
    except Exception as e:
        st.error(f"❌ Error during retrieval: {str(e)}")
        return [], []

# Show initialization status
if st.session_state.initialization_error:
    st.markdown(f'<div class="error-box">❌ Initialization Error: {st.session_state.initialization_error}</div>', unsafe_allow_html=True)

# Initialize FAISS if not already done
if not st.session_state.faiss_initialized:
    with st.spinner("Initializing RAG system..."):
        if initialize_faiss():
            st.rerun()

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ System Information")
    if st.session_state.faiss_initialized:
        st.success("✅ FAISS initialized")
        st.write(f"**Total chunks:** {st.session_state.total_chunks}")
        st.write(f"**Index dimension:** {st.session_state.index.d}")
    else:
        st.warning("⚠️ System not initialized")
    
    st.header("⚙️ Settings")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)

# Main query interface
if st.session_state.faiss_initialized:
    st.markdown('<h2>💬 Ask a Question</h2>', unsafe_allow_html=True)

    query = st.text_input(
        "Enter your question about the document:",
        placeholder="e.g., Who is uzumaki-ak? What technologies do they use?",
        help="Ask anything about the content of your document",
        key="query_input"
    )

    if st.button("🔍 Search and Generate Answer", type="primary"):
        if query:
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with st.spinner("🔍 Searching for relevant content..."):
                    # Retrieve relevant chunks
                    top_k_chunks, distances = retrieve_top_k(query, top_k)
                    
                    if top_k_chunks:
                        st.markdown('<h3>📄 Retrieved Context</h3>', unsafe_allow_html=True)
                        
                        for i, (chunk, distance) in enumerate(zip(top_k_chunks, distances)):
                            with st.expander(f"Chunk {i+1} (Distance: {distance:.4f})", expanded=i==0):
                                st.write(chunk)
            with col2:
                with st.spinner("🤖 Generating answer..."):
                    # Build prompt and generate answer
                    prompt = build_prompt(top_k_chunks, query)
                    answer = generate_ans(prompt)
                    
                    st.markdown('<h3>💡 Generated Answer</h3>', unsafe_allow_html=True)
                    st.info(answer)
                    
                    # Show prompt for debugging
                    with st.expander("🔧 View generated prompt"):
                        st.code(prompt)
        else:
            st.warning("⚠️ Please enter a question first!")
else:
    st.error("❌ System not properly initialized. Please check the error above.")

# System actions
st.markdown("---")
if st.button("🔄 Force Reinitialize FAISS Index"):
    cleanup_faiss_files()
    if initialize_faiss():
        st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit, FAISS, and Gemini API | RAG Document Query System")