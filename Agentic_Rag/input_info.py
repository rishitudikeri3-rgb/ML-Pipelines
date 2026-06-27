import os
import argparse
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(directory_path: str):
    """Load all text documents from the specified directory."""
    print(f"Loading documents from: {directory_path}")
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s).")
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def generate_embeddings_and_save(chunks, output_dir: str, embedding_provider: str = "huggingface"):
    """Generate vector embeddings and save the vector store locally."""
    print(f"Generating embeddings using provider: {embedding_provider}...")
    
    if embedding_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            print("\n[!] Error: 'langchain-huggingface' or 'sentence-transformers' is not installed.")
            print("Please run: pip install langchain-huggingface sentence-transformers\n")
            return False
        
        # Use a small, efficient local model (approx 90MB)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Initializing local HuggingFace model: {model_name}...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
    elif embedding_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            print("\n[!] Error: 'langchain-openai' is not installed.")
            print("Please run: pip install langchain-openai\n")
            return False
        
        if not os.environ.get("OPENAI_API_KEY"):
            print("[!] Warning: OPENAI_API_KEY environment variable is not set. The API call might fail.")
        
        print("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {embedding_provider}")

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("\n[!] Error: 'faiss-cpu' is not installed.")
        print("Please run: pip install faiss-cpu\n")
        return False

    print("Building vector store index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index locally
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving FAISS index to: {output_dir}")
    vector_store.save_local(output_dir)
    print("Vector store saved successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process document folder into vector embeddings.")
    parser.add_argument("--input_dir", type=str, default="input_data", help="Directory containing source text files")
    parser.add_argument("--output_dir", type=str, default="vector_embeddings", help="Directory to save the FAISS vector store")
    parser.add_argument("--provider", type=str, choices=["huggingface", "openai"], default="huggingface", 
                        help="Embedding provider (huggingface or openai)")
    parser.add_argument("--chunk_size", type=int, default=500, help="Text chunk size for splitting")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap size")
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input_dir)
    output_path = os.path.abspath(args.output_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: Input directory '{input_path}' does not exist.")
        return
        
    docs = load_documents(input_path)
    if not docs:
        print("No documents loaded. Exiting.")
        return
        
    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    success = generate_embeddings_and_save(chunks, output_path, embedding_provider=args.provider)
    if success:
        print("Process completed successfully.")
    else:
        print("Process failed due to missing dependencies.")

if __name__ == "__main__":
    main()
