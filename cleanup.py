import os
import shutil

def cleanup_faiss():
    """Remove corrupted FAISS files"""
    faiss_dir = "faiss_store"
    
    if os.path.exists(faiss_dir):
        print("Removing corrupted FAISS files...")
        shutil.rmtree(faiss_dir)
        print("✅ FAISS files removed successfully!")
    else:
        print("FAISS directory doesn't exist.")
    
    # Recreate the directory
    os.makedirs(faiss_dir, exist_ok=True)
    print("✅ FAISS directory recreated!")

if __name__ == "__main__":
    cleanup_faiss()