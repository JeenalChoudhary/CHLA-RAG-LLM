import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import logging
import json # This import is correctly included
import langid
from llama_index.llms.ollama import Ollama
import nltk
import argparse
import shutil

# ---- Configuration and Constraints ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SCRIPT_DIRECTORY will be the path to the 'backend' directory
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# PROJECT_ROOT is the parent directory of SCRIPT_DIRECTORY (e.g., CHLA-RAG-LLM-Capstone-Project-2025)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)

# Now, define paths relative to PROJECT_ROOT or SCRIPT_DIRECTORY as per your structure
# PDF_DIRECTORY is inside 'backend/data/docs'
PDF_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "data", "docs")

# DB_PATH and CACHE_DIR are also inside 'backend/data/'
DB_PATH = os.path.join(SCRIPT_DIRECTORY, "data", "db")
CACHE_DIR = os.path.join(SCRIPT_DIRECTORY, "data", "cache")
PROCESSED_DOCS_CACHE = os.path.join(CACHE_DIR, "processed_docs.json") # This path is correct relative to CACHE_DIR

COLLECTION_NAME = "example_health_docs"
EMBEDDING_MODEL_NAMES = {"english_medical":"NeuML/pubmedbert-base-embeddings"}
                # "multilingual":"paraphrase-multilingual-mpnet-base-v2"}
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L4-v2'

SENTENCES_PER_CHUNK = 6
STRIDE = 2
INITIAL_RETRIEVAL_COUNT = 50
FINAL_CONTEXT_COUNT = 8

_embedding_models = {}
_llm_models = {}
_reranker_model = None
_chroma_client = None # Global for the ChromaDB client
_collection = None    # Global for the ChromaDB collection

# ---- Setup NLTK ----
try:
    nltk.data.find('tokenizer/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
    logging.info("NLTK 'punkt' downloaded.")

# --- Add a function to initialize models and client once ---
def initialize_backend_components():
    global _embedding_models, _llm_models, _reranker_model, _chroma_client, _collection

    logging.info("Initializing RAG backend components...")

    # Load embedding models
    if "english_medical" in EMBEDDING_MODEL_NAMES and EMBEDDING_MODEL_NAMES["english_medical"] not in _embedding_models:
        logging.info(f"Loading English medical embedding model: {EMBEDDING_MODEL_NAMES['english_medical']}")
        _embedding_models["english_medical"] = SentenceTransformer(EMBEDDING_MODEL_NAMES["english_medical"])
    # if "multilingual" in EMBEDDING_MODEL_NAMES and EMBEDDING_MODEL_NAMES["multilingual"] not in _embedding_models:
    #     logging.info(f"Loading multilingual embedding model: {EMBEDDING_MODEL_NAMES['multilingual']}")
    #     _embedding_models["multilingual"] = SentenceTransformer(EMBEDDING_MODEL_NAMES["multilingual"])

    # Load reranker model
    if _reranker_model is None:
        logging.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)

    # Load LLM model (Ollama)
    if 'default_llm' not in _llm_models: # Using 'default_llm' as a key
        logging.info("Loading Ollama LLM model: gemma3:1b-it-qat")
        _llm_models['default_llm'] = Ollama(model="gemma3:1b-it-qat", request_timeout=300.0) # Adjust timeout as needed

    # Initialize ChromaDB client and collection
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        logging.error(f"Database not found or is empty at '{DB_PATH}'")
        logging.info("Please run 'python backend/backend_rag.py --rebuild' to create the database.")
        _collection = None # Ensure collection is None if DB is not ready
        _chroma_client = None
        return # Exit early if DB is not there

    if _chroma_client is None:
        logging.info(f"Connecting to ChromaDB at {DB_PATH}")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    if _collection is None:
        try:
            _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
            logging.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded. Document count: {_collection.count()}")
        except Exception as e:
            logging.error(f"Error loading ChromaDB collection '{COLLECTION_NAME}': {e}")
            logging.info("Please ensure the database has been built. Run 'python backend/backend_rag.py --rebuild'.")
            _collection = None # Ensure collection is None if there's an error


# ---- Document Processing Functions ----
def load_documents(directory):
    documents = []
    processed_files = load_processed_docs_cache()
    if not os.path.exists(directory):
        logging.warning(f"Document directory '{directory}' not found. No documents will be loaded.")
        return []

    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            if filepath in processed_files:
                logging.info(f"Skipping already processed file: {filename}")
                continue

            try:
                doc_content = []
                with fitz.open(filepath) as doc:
                    for page in doc:
                        text = page.get_text()
                        doc_content.append(text)
                
                # Simple check for empty content
                if not any(doc_content):
                    logging.warning(f"File {filename} contained no readable text. Skipping.")
                    continue
                
                documents.append({"id": filepath, "content": "\n".join(doc_content)})
                logging.info(f"Loaded content from {filename}")
            except Exception as e:
                logging.error(f"Error loading PDF {filename}: {e}", exc_info=True)
    return documents

def classify_and_chunk_documents(documents):
    all_chunks = []
    # english_docs = []
    # other_docs = []

    for doc in documents:
        text = doc['content']
        filename = os.path.basename(doc['id'])

        # Basic language detection (can be improved)
        # lang, _ = langid.classify(text[:1000]) # Classify based on first 1000 chars for speed
        # if lang != 'en':
        #     logging.info(f"Classified '{filename}' as non-English ({lang}).")
        #     # For simplicity, if not English, just add to other_docs
        #     other_docs.append({"content": text, "metadata": {"filename": filename, "language": lang}})
        #     continue

        # Process as English (default if no other language model)
        logging.info(f"Processing '{filename}' as English.")
        sentences = nltk.sent_tokenize(text)
        
        # Simple chunking by sentences
        for i in range(0, len(sentences), STRIDE):
            chunk_sentences = sentences[i:i + SENTENCES_PER_CHUNK]
            if not chunk_sentences:
                continue
            chunk_content = " ".join(chunk_sentences)
            
            all_chunks.append({
                "content": chunk_content,
                "metadata": {"filename": filename, "language": "en"}
            })

    # Saving processed documents to cache after chunking (not before)
    # This ensures that even if chunking fails for some reason, the file isn't marked as fully processed
    processed_files = load_processed_docs_cache()
    for doc in documents:
        processed_files.add(doc['id'])
    save_processed_docs_cache(processed_files)

    # Returning all_chunks (English only based on current config)
    # and an empty list for other_docs as multilingual is commented out
    return all_chunks, [] # No other_docs currently processed


def embed_and_store(docs, embedding_model, collection, model_type='english'):
    if not docs:
        return
    logging.info(f"Embedding and storing {len(docs)} documents with {model_type} model...")
    batch_size = 32 # Can adjust based on memory/model
    total_batches = (len(docs) + batch_size - 1) // batch_size
    for i in range(0, len(docs), batch_size):
        current_batch_num = (i // batch_size) + 1
        logging.info(f"Processing batch {current_batch_num}/{total_batches}...")
        batch = docs[i:i + batch_size]
        contents = [doc['content'] for doc in batch]
        metadatas = [doc['metadata'] for doc in batch]
        # Add a unique ID for each chunk based on original doc ID and chunk index
        ids = [f"{os.path.basename(doc['metadata']['filename'])}_chunk_{i+j}" for j, doc in enumerate(batch)]
        
        # Ensure metadata is JSON serializable
        for meta in metadatas:
            if 'filename' in meta:
                meta['filename'] = str(meta['filename']) # Ensure filename is string
            if 'language' in meta:
                meta['language'] = str(meta['language']) # Ensure language is string
            meta['model'] = model_type # Add model type to metadata

        try:
            embeddings = embedding_model.encode(contents, show_progress_bar=False, convert_to_tensor=False).tolist()
            collection.add(embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids)
        except Exception as e:
            logging.error(f"Error during embedding or storing batch {current_batch_num}: {e}", exc_info=True)
            logging.error(f"Problematic batch contents (first item): {contents[0] if contents else 'N/A'}")


def retrieve_context(query, chat_history, collection):
    """
    Retrieves relevant document chunks based on the query and chat history.
    """
    if collection is None:
        logging.error("ChromaDB collection is not initialized. Cannot retrieve context.")
        return [], []

    # Combine query with recent chat history for better context in retrieval
    combined_query = query
    if chat_history:
        # Use only the last few turns for context to avoid too much noise
        recent_history = " ".join([msg['text'] for msg in chat_history[-3:]])
        combined_query = f"{recent_history} {query}"
    
    logging.info(f"Combined query for retrieval: {combined_query}")

    # Use the appropriate embedding model for the query
    embedding_model = _embedding_models["english_medical"] # Default to English
    # If you enable multilingual, you might add logic here to choose model based on query lang
    
    query_embedding = embedding_model.encode(combined_query, convert_to_tensor=False).tolist()

    # Step 1: Initial retrieval
    retrieved_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=INITIAL_RETRIEVAL_COUNT,
        include=['documents', 'metadatas', 'distances']
    )

    if not retrieved_results or not retrieved_results['documents']:
        logging.info("No documents retrieved from ChromaDB.")
        return [], []

    retrieved_chunks = []
    for i in range(len(retrieved_results['documents'][0])):
        retrieved_chunks.append({
            "content": retrieved_results['documents'][0][i],
            "metadata": retrieved_results['metadatas'][0][i],
            "distance": retrieved_results['distances'][0][i]
        })
    logging.info(f"Retrieved {len(retrieved_chunks)} initial chunks.")

    # Step 2: Re-ranking
    if _reranker_model:
        query_passage_pairs = [[combined_query, chunk['content']] for chunk in retrieved_chunks]
        rerank_scores = _reranker_model.predict(query_passage_pairs).tolist()
        
        for i, score in enumerate(rerank_scores):
            retrieved_chunks[i]['rerank_score'] = score
        
        # Sort by rerank score (higher score is better) and select top FINAL_CONTEXT_COUNT
        reranked_chunks = sorted(retrieved_chunks, key=lambda x: x.get('rerank_score', -1), reverse=True)[:FINAL_CONTEXT_COUNT]
        logging.info(f"Selected {len(reranked_chunks)} chunks after re-ranking.")
    else:
        # If no reranker, just take the top N by initial distance
        reranked_chunks = sorted(retrieved_chunks, key=lambda x: x['distance'])[:FINAL_CONTEXT_COUNT]
        logging.info(f"Selected {len(reranked_chunks)} chunks by distance (no reranker).")

    # Extract unique sources
    sources = set()
    for chunk in reranked_chunks:
        if 'filename' in chunk['metadata']:
            sources.add(chunk['metadata']['filename'])
    
    return reranked_chunks, sorted(list(sources))

# ---- Cache Management ----
def load_processed_docs_cache():
    if os.path.exists(PROCESSED_DOCS_CACHE):
        with open(PROCESSED_DOCS_CACHE, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_docs_cache(processed_files):
    os.makedirs(os.path.dirname(PROCESSED_DOCS_CACHE), exist_ok=True)
    with open(PROCESSED_DOCS_CACHE, 'w') as f:
        json.dump(list(processed_files), f)

# --- Modify handle_query to stream ---
# Renaming for clarity and adding `collection` as a parameter
def handle_query_stream(query: str, chat_history: list) -> (str, list):
    """
    Handles a user query by retrieving context and generating an answer,
    streaming the response.
    """
    global _collection # Use the globally initialized collection

    if _collection is None:
        logging.error("ChromaDB collection is not initialized. Cannot handle query.")
        yield 'error', "The knowledge base is not ready. Please ensure the database is built and the backend is initialized."
        yield 'sources', []
        return

    logging.info(f"Handling query: '{query}' with history length: {len(chat_history)}")

    try:
        # Step 1: Retrieve context
        context_docs, sources = retrieve_context(query, chat_history, _collection) # Use the global _collection

        if not context_docs:
            yield 'text', "I couldn't find relevant information in my knowledge base for your question. Please try rephrasing or asking a different question."
            yield 'sources', []
            return

        # Step 2: Generate answer using LLM
        llm = _llm_models['default_llm'] # Get the pre-loaded LLM

        # Build prompt for LLM
        context_str = "\n\n".join([doc['content'] for doc in context_docs])
        chat_history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "USER" if msg["isUser"] else "ASSISTANT"
                chat_history_str += f"{role}: {msg['text']}\n"

        system_message = (
            "You are a compassionate, empathetic, and knowledgeable medical assistant for Children's Hospital Los Angeles (CHLA). "
            "Your purpose is to provide health education information to patient families, explaining complex medical concepts clearly, concisely, "
            "and in an easy-to-understand manner, ideally at an 8th-grade reading level. "
            "Prioritize patient safety and well-being. Always adhere to the following rules:"
        )

        behavioral_guardrails = (
            "- Always be empathetic and reassuring in your tone.\n"
            "- If you cannot find information in the provided context, state that you do not have enough information to answer.\n"
            "- Never provide medical advice, diagnosis, or treatment. Always remind the user to consult with a qualified healthcare professional for personal medical concerns.\n"
            "- Do not make up information or hallucinate. Stick strictly to the provided context.\n"
            "- Maintain strict patient privacy and confidentiality (though no PII will be in the docs).\n"
            "- If asked about sensitive topics (e.g., self-harm), provide general helpful resources (like a crisis line) but reiterate that you are not a substitute for professional help.\n"
            "- Keep answers concise and to the point, avoiding jargon where possible.\n"
            "- If a question is outside the scope of general health education or CHLA resources, gently steer the conversation back to appropriate topics."
        )

        answering_rules = (
            "- Answer the question based solely on the provided context.\n"
            "- If the context does not contain the answer, state that you cannot answer based on the provided information.\n"
            "- Summarize the relevant parts of the context to answer the user's query directly.\n"
            "- Format your answer clearly and logically.\n"
            "- Do not include the sources directly in your answer. The sources will be provided separately by the system."
        )

        prompt = (
            f"SYSTEM: {system_message}\n"
            f"BEHAVIORAL_GUARDRAILS: {behavioral_guardrails}\n"
            f"ANSWERING_RULES: {answering_rules}\n\n"
            f"CHAT HISTORY:\n{chat_history_str}\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER: {query}\n"
            "ASSISTANT:"
        )
        logging.info(f"Prompt sent to LLM:\n{prompt}")

        response_chunks = llm.stream_complete(prompt)
        full_response_content = ""
        for chunk in response_chunks:
            full_response_content += chunk.text
            yield 'text', chunk.text # Stream text chunks

        logging.info(f"Generated full response from LLM (first 200 chars): {full_response_content[:200]}...")
        yield 'sources', list(sources) # Send sources after all text has streamed

    except Exception as e:
        logging.error(f"Error in handle_query_stream: {e}", exc_info=True)
        yield 'error', f"An error occurred while generating the response: {str(e)}"
        yield 'sources', []

# --- Update the main execution block to support API setup and rebuild ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHLA RAG Backend.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the ChromaDB from documents in the 'docs' directory.")
    args = parser.parse_args()

    # When running the script directly, always initialize components
    initialize_backend_components()

    if args.rebuild:
        logging.info("Rebuild flag detected. Starting database rebuild process...")
        # Clean up old database and cache
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            logging.info(f"Removed existing database at {DB_PATH}")
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            logging.info(f"Removed existing cache at {CACHE_DIR}")
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load documents and embed
        docs_to_process = load_documents(PDF_DIRECTORY)
        if docs_to_process:
            # Re-initialize client and collection specifically for rebuild process
            # This ensures we get a fresh collection even if _collection was None
            client_rebuild = chromadb.PersistentClient(path=DB_PATH)
            collection_rebuild = client_rebuild.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            logging.info(f"Initial collection count for rebuild: {collection_rebuild.count()}")
            if collection_rebuild.count() == 0:
                logging.info(f"Ingesting {len(docs_to_process)} documents into ChromaDB...")
                english_docs, other_docs = classify_and_chunk_documents(docs_to_process)
                embed_and_store(english_docs, _embedding_models["english_medical"], collection_rebuild)
                # If multilingual was enabled:
                # embed_and_store(other_docs, _embedding_models["multilingual"], collection_rebuild, model_type='multilingual')
                logging.info(f"Ingestion complete. Total documents in DB: {collection_rebuild.count()}")
            else:
                logging.info(f"Database already populated with {collection_rebuild.count()} documents. Skipping ingestion during rebuild.")
        else:
            logging.warning("No documents were loaded from the 'docs' directory. Database might be empty after rebuild attempt.")
    
    # After potential rebuild, ensure the global collection is set up for subsequent calls
    # This might re-run parts of initialize_backend_components, but it ensures _collection is current
    initialize_backend_components()

    logging.info("Backend script finished. To run the API, execute 'python backend/app.py'.")
