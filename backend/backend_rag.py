import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import logging
import json
import nltk
import argparse
import shutil
from llama_index.llms.ollama import Ollama

# ---- Configuration and Constraints ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "data")
PDF_DIRECTORY = os.path.join(DATA_DIRECTORY, "docs")
DB_PATH = os.path.join(DATA_DIRECTORY, "db")
CACHE_DIR = os.path.join(DATA_DIRECTORY, "cache")
PROCESSED_DOCS_CACHE = os.path.join(CACHE_DIR, "processed_docs.json")

COLLECTION_NAME = "example_health_docs"
EMBEDDING_MODEL_NAMES = "NeuML/pubmedbert-base-embeddings"
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L4-v2'
LLM_MODEL_NAME = "gemma3:1b-it-qat"
SENTENCES_PER_CHUNK = 6
STRIDE = 2
INITIAL_RETRIEVAL_COUNT = 50
FINAL_CONTEXT_COUNT = 10

_embedding_models = None
_llm_models = None
_reranker_model = None
_chroma_client = None
_collection = None

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    if _embedding_models is None:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAMES}")
        _embedding_models = SentenceTransformer(EMBEDDING_MODEL_NAMES, device=device)
    if _reranker_model is None:
        logging.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=device)
    if _llm_models is None: 
        logging.info(f"Loading Ollama LLM model: {LLM_MODEL_NAME}")
        _llm_models = Ollama(model=LLM_MODEL_NAME, request_timeout=300.0)
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        logging.error(f"Database not found or is empty at '{DB_PATH}'")
        logging.info("Please run 'python backend/backend_rag.py --rebuild' to create the database.")
        return
    if _chroma_client is None:
        logging.info(f"Connecting to ChromaDB at: {DB_PATH}")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
        logging.info(f"Sucessfully loaded ChromaDB collection '{COLLECTION_NAME}'. Document count: {_collection.count()}")
    except Exception as e:
        logging.error(f"Failed to load ChromaDB collection '{COLLECTION_NAME}': {e}")
        logging.info("Please ensure the database has been built. Run 'python backend/backend_rag.py --rebuild'.")
        _collection = None

def clean_pdf_text(text: str) -> str:
    text = re.sub(r"Children's\s+Hospital\s+LOS ANGELES", "", text, flags=re.IGNORECASE)
    text = re.sub(r".*4650 Sunset Blvd\., Los Angeles, CA 90027.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}.*LLC.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Disclaimer:.*|This information is not intended as a substitute for professional medical care.*|This information is intended for general knowledge.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    text = re.sub(r" {2,}", " ", text)
    return text

# ---- Document Processing Functions ----
def load_and_process_docs(directory):
    all_chunks = []
    processed_pdf_count = 0
    if not os.path.exists(directory):
        logging.error(f"PDF directory not found: {directory}")
        return all_chunks
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    logging.info(f"Found {len(pdf_files)} PDF files in '{directory}'.")
    for filename in pdf_files:
        path = os.path.join(directory, filename)
        logging.info(f"Processing PDF: '{path}'")
        try:
            doc = fitz.open(path)
            full_text = "".join(page.get_text() for page in doc)
            doc.close()
            cleaned_text = clean_pdf_text(full_text)
            if not cleaned_text:
                logging.warning(f"No content left in {filename} after cleaning. Skipping file.")
                continue
            sentences = nltk.sent_tokenize(cleaned_text)
            logging.info(f"Extracting {len(sentences)} sentences from {filename}.")
            if len(sentences) < SENTENCES_PER_CHUNK:
                if sentences:
                    all_chunks.append({
                        "content": " ".join(sentences),
                        "metadata": {"source": filename, "start_sentence_index":0}
                    })
                    logging.info(f"Document too short, creating a single chunk")
            else:            
                initial_chunk_count = len(all_chunks)
                for i in range(0, len(sentences) - SENTENCES_PER_CHUNK + 1, STRIDE):
                    chunk_text = " ".join(sentences[i:i + SENTENCES_PER_CHUNK])
                    all_chunks.append({
                        "content": chunk_text,
                        "metadata": {"source": filename, "start_sentence_index": i}
                    })
                if len(sentences) % STRIDE != 0:
                    last_chunk_start = len(sentences) - SENTENCES_PER_CHUNK
                    if last_chunk_start > 0 and (len(all_chunks) == initial_chunk_count or last_chunk_start > all_chunks[-1]['metadata']['start_sentence_index']):
                        chunk_text = " ".join(sentences[last_chunk_start:])
                        all_chunks.append({
                            "content": chunk_text,
                            "metadata": {"source": filename, "start_sentence_index": last_chunk_start}
                        })
                logging.info(f"Created {len(all_chunks) - initial_chunk_count} chunks for {filename}.")
            processed_pdf_count += 1
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}", exc_info=True)
    logging.info(f"---- Document Processing Complete ----")
    logging.info(f"Successfully processed {len(os.listdir(directory))} PDFs into {len(all_chunks)} chunks.")
    return all_chunks

def generate_hypothetical_document(query: str, conversation_history: str = "") -> str:
    history_prompt_section = ""
    if conversation_history:
        history_prompt_section = f"""
        Here is the recent conversation history. Use it to understand the full context of the user's latest query.
        ---
        {conversation_history}
        ---
        """
    prompt = f""" You are a medical writer creating a hypothetical document for a database search.
        {history_prompt_section}
        Your task is to generate an ideal, detailed paragraph that directly answers and expands upon the user's question: '{query}'.

        **Instructions:**
        - Use the conversation history to tailor the paragraph. For example, if the history is about "a newborn baby" and the query is "feeding tubes", the hypothetical document should be about **feeding tubes specifically for newborns.**
        - Elaborate on the answer by explaining the 'how' and 'why', including details about related procedures, or what a caregiver should know.
        - The final paragraph must be self-contained, dense, and cohesive. Do not use lists or add introductory or concluding phrases.

        HYPOTHETICAL DOCUMENT:
        """
    try:
        response = _llm_models.complete(prompt)
        logging.info(f"Successfully generated hypothetical document for query '{query}': {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error generating hypothetical document: {e}")
        return query

def retrieve_context(query, chat_history):
    if _collection is None:
        logging.error("ChromaDB collection is not initialized. Cannot retrieve context.")
        return [], []
    hypothetical_document = generate_hypothetical_document(query, chat_history)
    query_embedding = _embedding_models.encode(hypothetical_document).tolist()
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=INITIAL_RETRIEVAL_COUNT
    )
    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    if not retrieved_docs:
        logging.warning("No documents found in the initial retrieval.")
        return [], []
    rerank_pairs = [[query, doc] for doc in retrieved_docs]
    scores = _reranker_model.predict(rerank_pairs)
    scored_docs = sorted(zip(scores, retrieved_docs, retrieved_metadata), key=lambda x: x[0], reverse=True)
    top_docs = scored_docs[:FINAL_CONTEXT_COUNT]
    final_context = [doc for score, doc, meta in top_docs]
    final_metadata = [meta for score, doc, meta in top_docs]
    sources = sorted(list(set(meta['source'] for meta in final_metadata)))
    logging.info(f"Reranked and selected top {len(final_context)} documents for query: '{query}'")
    return final_context, sources

def generate_answer_stream(query: str, context_docs: list, conversation_history: str):
    context = "\n\n---\n\n".join(context_docs)
    history_prompt = ""
    if conversation_history:
        history_prompt = f"""
        Here is the recent conversation history. Use it to understand the full context of the user's latest query.
        ---
        {conversation_history}
        ---
        """
    prompt_template = f"""
        You are an expert medical educator and assistant at Children's Hospital Los Angeles. Your purpose is to provide clear, comprehensive, and reassuring answers to patient families who have an average 8th-grade reading level, based **STRICTLY** on the provided context.
        
        {history_prompt}
        
        ---
        **BEHAVIORAL GUARDRAILS (Follow these always):**
        - **NEVER** break character. You are an assistant from CHLA, not a generic AI.
        - **NEVER** mention that you are an AI, a language model, or a chatbot.
        - **NEVER** add a medical disclaimer. Your role is educational, not advisory.
        - **NEVER** provide external website links or suggest resources not mentioned in the context.
        - **NEVER** praise the user's question (e.g., do not say "That's a great question"). A reassuring tone is better (e.g., "That's an important question...").
        ---
        
        **ANSWERING RULES:**
        1. **Synthesize a direct, complete answer** from the context. Do not ask for clarification.
        2. **Tone:** For serious topics, begin with a short, reassuring sentence (e.g., "It's wise to prepare for that. Based on the provided information...").
        3. **Completeness:** Never give a simple "yes" or "no". Always explain the "how" and "why" using details from the context.
        4. **Missing Info:** If the context does not contain the answer, state that clearly and simply: "I could not find specific information on that topic in the provided documents." Do not try to answer from general knowledge.
        
        ***IMPORTANT: You must write your entire answer in English.***'
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
    """
    logging.info(f"Sending final prompt to LLM to query: '{query}'")
    response_iter = _llm_models.stream_complete(prompt_template)
    for token in response_iter:
        yield token.delta

def handle_query_stream(query: str, chat_history: list):
    global _collection
    if _collection is None:
        logging.error(f"Cannot handle query because ChromaDB collection is not initialized.")
        yield 'error', "The knowledge base is not ready. Please contact support."
        yield 'sources', []
        return
    logging.info(f"Handling streamed query: {query}")
    history_str = "\n".join([f"{'User' if msg.get('isUser') else 'Assistant'}:{msg.get('text')}" for msg in chat_history])
    try:
        context_docs, sources = retrieve_context(query, history_str)
        if not context_docs:
            yield 'text', "I couldn't find relevant information for your question in my knowledge base. Please try rephrasing your query."
            yield 'sources', []
            return
        answer_generator = generate_answer_stream(query, context_docs, history_str)
        for token in answer_generator:
            yield 'text', token
        yield 'sources', sources
    except Exception as e:
        logging.error(f"An error occurred while handling the query: {e}", exc_info=True)
        yield 'error', f"An internal error occurred while generating the response."
        yield 'sources', []

# --- Update the main execution block to support API setup and rebuild ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend script for the CHLA RAG chatbot. Run with --rebuild to populate the database.")
    parser.add_argument("--rebuild", action="store_true", help="RForce a complete rebuild of the database and all cached files from the PDFs in the data/docs directory.")
    args = parser.parse_args()
    if args.rebuild:
        logging.info("Rebuild flag detected. Starting database rebuild process...")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            logging.info(f"Removed existing database at {DB_PATH}")
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            logging.info(f"Removed existing cache at {CACHE_DIR}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        chunks = load_and_process_docs(PDF_DIRECTORY)
        if not chunks:
            logging.error(f"No chunks were created from the PDFs. Aborting database build.")
        else:
            initialize_backend_components()
            logging.info("Initializing components for database rebuild...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model_build = SentenceTransformer(EMBEDDING_MODEL_NAMES, device=device)
            client_build = chromadb.PersistentClient(path=DB_PATH)
            collection_build = client_build.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
            batch_size = 128
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            logging.info(f"Embedding {len(chunks)} chunks in {total_batches} batches...")
            for i in range(0, len(chunks), batch_size):
                batch_num = (i // batch_size) + 1
                logging.info(f"Processing batch {batch_num}/{total_batches}...")
                batch = chunks[i:i + batch_size]
                contents = [doc['content'] for doc in batch]
                metadatas = [doc['metadata'] for doc in batch]
                ids = [f"{doc['metadata']['source']}_chunk_{doc['metadata']['start_sentence_index']}" for doc in batch]
                embeddings = embedding_model_build.encode(contents, show_progress_bar=False).tolist()
                collection_build.add(embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids)
            logging.info(f"---- Rebuild complete! ----")
            logging.info(f"Database created at '{DB_PATH}' with {collection_build.count()} documents.")
            logging.info("You can now start the Flask API server using 'python app.py'.")
    else:
        print("Script finished. This script is intended to be run with the '--rebuild' flag to set up the database.")
        print("To run the web application, execute the 'python (or py) app.py' command from the backend folder.")