import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "vector_embeddings")

def initialize_vector_store():
    """Load the saved FAISS vector database."""
    print("\n--- [Step 1] Initializing Vector Store ---")
    if not os.path.exists(DB_DIR):
        print(f"[!] Error: Vector database directory '{DB_DIR}' not found.")
        print("Please run 'python input_info.py' first to generate embeddings.")
        sys.exit(1)
        
    print("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Loading FAISS database index...")
    # allow_dangerous_deserialization=True is required to load locally saved pickle files
    vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Vector database successfully loaded!")
    return vector_store

def setup_llm():
    """Setup the language model based on environment variables or user input."""
    print("\n--- [Step 2] Setting up Language Model (LLM) ---")
    
    # Try to detect API keys in the environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    
    if openai_key:
        print("[+] Detected OPENAI_API_KEY in environment. Using OpenAI Chat Model.")
        return get_openai_llm()
    elif google_key:
        print("[+] Detected GOOGLE_API_KEY in environment. Using Google GenAI Chat Model.")
        return get_google_llm()
        
    # If no keys are in the environment, prompt the user
    print("No environment API keys detected (OPENAI_API_KEY or GOOGLE_API_KEY).")
    print("How would you like to run the chatbot?")
    print("  1. OpenAI (Requires key)")
    print("  2. Google Gemini (Requires key)")
    print("  3. Retrieval-Only Debug Mode (No key required; shows search scores and prompt)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        key = input("Enter your OpenAI API Key: ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return get_openai_llm()
    elif choice == "2":
        key = input("Enter your Google API Key: ").strip()
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            return get_google_llm()
            
    print("\n[i] Starting in Retrieval-Only Debug Mode.")
    return "retrieval_only"

def get_openai_llm():
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except ImportError:
        print("[!] Error: 'langchain-openai' is not installed.")
        print("Please install it with: pip install langchain-openai")
        print("Falling back to Retrieval-Only Debug Mode.")
        return "retrieval_only"

def get_google_llm():
    try:
        from langchain_google_genai import ChatGoogleGenAI
        return ChatGoogleGenAI(model="gemini-2.5-flash", temperature=0)
    except ImportError:
        print("[!] Error: 'langchain-google-genai' is not installed.")
        print("Please install it with: pip install langchain-google-genai")
        print("Falling back to Retrieval-Only Debug Mode.")
        return "retrieval_only"

def run_chat_loop(vector_store, llm):
    """Interactive loop for chatbot queries and explanations."""
    print("\n=========================================================")
    print("      AetherFlow HR Assistant Chatbot Initialized        ")
    print("=========================================================")
    print("Ask me anything about company policies, rules, and regulations.")
    print("Type 'exit', 'quit', or 'q' to end the chat.")
    print("=========================================================\n")
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not query:
            continue
            
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        # STEP A: Retrieve relevant chunks and display distance scores
        print("\n--- [Step A] Searching Vector Database (FAISS) ---")
        # similarity_search_with_score returns list of (Document, score)
        # Score is L2 distance: lower is more similar (0.0 is identical)
        results = vector_store.similarity_search_with_score(query, k=3)
        
        print(f"Retrieved {len(results)} chunks of text from index.")
        for idx, (doc, score) in enumerate(results):
            snippet = doc.page_content.replace('\n', ' ')[:80] + "..."
            filename = os.path.basename(doc.metadata.get('source', 'unknown'))
            print(f"  Chunk {idx+1}: L2 Distance Score={score:.4f} | Src={filename}")
            print(f"    Snippet: \"{snippet}\"\n")
            
        # STEP B: Construct context and final prompt
        context_parts = []
        for doc, _ in results:
            context_parts.append(doc.page_content)
        context = "\n\n---\n\n".join(context_parts)
        
        system_instruction = (
            "You are AetherFlow Technologies' helpful HR Assistant.\n"
            "Answer the user's question using ONLY the provided corporate context.\n"
            "If the context does not contain the answer, politely state that you do not know.\n"
            "Do not make up facts or extrapolate beyond the provided text.\n"
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        print("--- [Step B] Prompt Construction ---")
        print("System Instruction:")
        print(f"  > {system_instruction.replace(chr(10), chr(10)+'  > ')}")
        print("Formatted Context Fed to LLM:")
        # Show a formatted block of the context sent to the LLM
        for i, text in enumerate(context_parts):
            first_line = text.split('\n')[0]
            print(f"  [Chunk {i+1} Start: '{first_line[:40]}...']\n  {text[:150].replace(chr(10), chr(10)+'  ')}...")
            print("  [Chunk End]\n")
            
        # STEP C: Generate response via LLM or print instructions
        print("--- [Step C] Querying LLM ---")
        if llm == "retrieval_only":
            print("[Retrieval-Only Mode] Retrieved context successfully.")
            print("To receive real AI-generated answers, restart this script and enter your API Key,")
            print("or run 'pip install langchain-openai' and set the OPENAI_API_KEY environment variable.")
        else:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_prompt)
                ]
                print("Sending prompt to LLM...")
                response = llm.invoke(messages)
                print("\n=========================================")
                print(f"HR Assistant: {response.content}")
                print("=========================================")
            except Exception as e:
                print(f"[!] Error calling LLM API: {e}")
                print("Falling back to retrieval-only info.")

def main():
    vector_store = initialize_vector_store()
    llm = setup_llm()
    run_chat_loop(vector_store, llm)

if __name__ == "__main__":
    main()
