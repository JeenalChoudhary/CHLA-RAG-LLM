import os
import re
import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import logging
import json
import nltk
import time
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
EMBEDDING_MODEL_NAMES = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L4-v2'
LLM_MODEL_NAME = "gemma3:1b-it-qat"
SENTENCES_PER_CHUNK = 6
STRIDE = 2
INITIAL_RETRIEVAL_COUNT = 20
FINAL_CONTEXT_COUNT = 10
RERANKER_SCORE_THRESHOLD = 0.3

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
    nltk.download('punkt_tab', quiet=True)
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
        _llm_models = Ollama(model=LLM_MODEL_NAME, base_url="http://ollama:11434", request_timeout=600.0)
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        logging.error(f"Database not found or is empty at '{DB_PATH}'")
        logging.info("Please run 'python backend/backend_rag.py --rebuild' to create the database.")
        return
    if _chroma_client is None:
        logging.info(f"Connecting to ChromaDB at: {DB_PATH}")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, is_persistent=True, anonymized_telemetry=False))
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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text

# ---- Document Processing Functions ----
def load_and_process_docs(directory):
    all_chunks = []
    if not os.path.exists(directory):
        logging.error(f"PDF directory not found: {directory}")
        return all_chunks
    
    for filename in os.listdir(directory):
        if not filename.endswith(".pdf"):
            continue
        
        path = os.path.join(directory, filename)
        logging.info(f"Processing PDF: '{filename}'")
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
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}", exc_info=True)
    logging.info(f"---- Document Processing Complete ----")
    logging.info(f"Successfully processed {len(os.listdir(directory))} PDFs into {len(all_chunks)} chunks.")
    return all_chunks

def generate_hypothetical_document(query: str, conversation_history: str) -> list:
    history_prompt = ""
    if conversation_history:
        history_prompt = f"""
        <conversation_history>
        {conversation_history}
        </conversation_history>
        """
    prompt = f"""
        You are an expert query analyst. Your task is to refine a user's query based on conversation history and then break it down into 3-5 specific, self-contained questions for a medical knowledge base search.

        {history_prompt}

        **Instructions:**
        1.  **Analyze the Query:** If the query is a follow-up (e.g., "what about for kids?"), use the history to create a clear, standalone query first.
        2.  **Generate Varied Questions:** From the full query, create a JSON list of questions that explore different angles.

        **User Query:** '{query}'

        Generate a JSON list of questions based on your analysis. For example: ['question 1', 'question 2', 'question 3']

        **JSON list of questions:**
    """
    response = _llm_models.complete(prompt)
    response_text = response.text.strip()
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    try:
        sub_questions = json.loads(response_text)
        logging.info(f"Successfully generated sub-questions: {sub_questions}")
        return sub_questions
    except json.JSONDecodeError:
        logging.error(f"Failed to decode the sub questions: {response.text}")
        return [query]

def retrieve_context(query: str, conversation_history: str = "", n_results: int = FINAL_CONTEXT_COUNT):
    logging.info(f"Retrieving context for query: '{query}'")
    sub_queries = generate_hypothetical_document(query, conversation_history=conversation_history)
    all_queries = [query] + sub_queries
    retrieved_doc_set = {}
    logging.info(f"Performing multi-query retrieval for {all_queries}")
    for sub_q in all_queries:
        query_embedding = _embedding_models.encode(sub_q).tolist()
        results = _collection.query(query_embeddings=[query_embedding], n_results=INITIAL_RETRIEVAL_COUNT)
        for i, doc_content in enumerate(results['documents'][0]):
            if doc_content not in retrieved_doc_set:
                retrieved_doc_set[doc_content] = results['metadatas'][0][i]
        if not retrieved_doc_set:
            logging.warning(f"Multi-query retrieval returned no relevant documents for reranking.")
            return [], []
    retrieved_docs = list(retrieved_doc_set.keys())
    retrieved_metadata = list(retrieved_doc_set.values())
    logging.info(f"Retrieved {len(retrieved_docs)} documents for reranking.")
    rerank_pairs = [[query, doc] for doc in retrieved_docs]
    scores = _reranker_model.predict(rerank_pairs, show_progress_bar=False)
    sorted_docs = sorted(zip(scores, retrieved_docs, retrieved_metadata), key=lambda x: x[0], reverse=True)
    if not sorted_docs or sorted_docs[0][0] < RERANKER_SCORE_THRESHOLD:
        logging.warning(f"Reranker returned no sufficiently relevant documents. The top relevance score was {sorted_docs[0][0] if sorted_docs else 'N/A'}. Aborting generation.")
        return [], []
    top_reranked_docs = sorted_docs[:n_results]
    final_docs = [doc for score, doc, meta in top_reranked_docs]
    final_metadata = [meta for score, doc, meta in top_reranked_docs]
    sources = sorted(list(set(meta['source'] for meta in final_metadata)))
    logging.info(f"Reranked and selected top {len(final_docs)} documents.")
    logging.info(f"Retrieved final context from sources: {sources}")
    return final_docs, sources
   
def generate_answer_stream(query: str, context_docs: list, conversation_history: str):
    context = "\n\n---\n\n".join(context_docs)
    history_prompt = ""
    if conversation_history:
        history_prompt = f"""
        <conversation_history>
        {conversation_history}
        </conversation_history>
        """
    prompt_template = f"""
    **Persona:** You are an expert medical educator at Children's Hospital Los Angeles. Your mission is to provide helpful, reassuring, and empathetic guidance to worried parents and patients. Your language MUST be simple and clear.
   
    **--- YOUR MOST IMPORTANT RULES ---**
    1. **USE ONLY THE PROVIDED CONTEXT.** Your entire response MUST be generated using ONLY the information from the "CONTEXT" section below. Do not use any outside knowledge.
    2. **NO EXTERNAL INFORMATION OR LINKS.** Under NO circumstances will you suggest external organizations, provide URLs, or mention websites (e.g., "chla.org").
    3. **NO DISCLAIMERS.** Under NO circumstances will you include a disclaimer about not being a medical professional or that the information is not medical advice.
    4. **NO UNSUPPORTIVE OR BIZARRE ADVICE.** Do not invent information or stitch together text in a way that creates nonsensical, unhelpful, or dangerous advice. All information must be grounded and coherent.
    5. **STICK TO THE PERSONA.** You are a medical educator at CHLA. Do not break character.
    6. **HANDLE EMPTY CONTEXT.** If the "CONTEXT" section is empty or irrelevant, you MUST ONLY reply with the exact phrase: "I could not find any information related to your question in the documents I have access to."
   
    **Instructions:**
    1. Analyze the user's "Question" and the provided "CONTEXT".
    2. Synthesize the information from the CONTEXT into a clear answer. Use lists and bolding to make the information easy to digest.
    3. If the user's query is too broad (e.g., "asthma", "breathing problems"), provide a general overview and then suggest 2-3 specific follow-up questions to guide the user.
   
    ---
    CONTEXT:
    {context}
    ---
    {history_prompt}
    Question: {query}
   
    Answer:
    """
    logging.info(f"Sending final prompt to LLM to query: '{query}'")
    response_iter = _llm_models.stream_complete(prompt_template)
    for token in response_iter:
        yield token.delta

def deduplicate_context(documents: list, similarity_threshold: float = 0.92) -> list:
    if not documents:
        return []
    logging.info(f"Starting de-duplication on {len(documents)} documents with {similarity_threshold}...")
    embeddings = _embedding_models.encode(documents, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    unique_docs = []
    is_duplicate = [False] * len(documents)
    for i in range(len(documents)):
        if not is_duplicate[i]:
            unique_docs.append(documents[i])
            for j in range(i + 1, len(documents)):
                if cosine_scores[i][j] > similarity_threshold:
                    is_duplicate[j] = True
    logging.info(f"De-duplication complete. Reduced from {len(documents)} to {len(unique_docs)} documents.")
    return unique_docs

def parse_and_clean_output(text: str) -> str:
    logging.info("Applying post-processing parser to final output.")
    text = re.sub(r"https?://\S+|www\.\S+|\S+\.(com|org|net)\S*", "", text, flags=re.IGNORECASE)
    disclaimer_patterns = [r"this information is for information purposes only.*", r"consult a medical professional.*", r"this is not medical advice.*", r"important note:.*", r"disclaimer:.*"]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\*\*?(important resources|additional resources|resources)\*\*?:?", "", text, flags=re.IGNORECASE)
    return text.strip()

def handle_query_stream(query: str, chat_history: list):
    global _collection
    if _collection is None:
        logging.error(f"Cannot handle query because ChromaDB collection is not initialized.")
        yield 'error', "The knowledge base is not ready. Please contact support."
        yield 'sources', []
        return
    logging.info(f"Handling streamed query: {query}")
    cleaned_history = []
    for msg in chat_history:
        role = 'User' if msg.get('isUser') else 'Assistant'
        text = msg.get('text', '')
        if not msg.get('isUser'):
            text = text.split('**Sources:**')[0].strip()
        if text:
            cleaned_history.append(f"{role}: {text}")
    history_str = "\n".join(cleaned_history)
    try:
        context_docs, sources = retrieve_context(query, history_str)
        if not context_docs:
            yield 'text', "I couldn't find relevant information for your question in my knowledge base. Please try rephrasing your query."
            yield 'sources', []
            return
        deduplicated_docs = deduplicate_context(context_docs)
        answer_generator = generate_answer_stream(query, deduplicated_docs, history_str)
        full_response_text = "".join([token for token in answer_generator])
        cleaned_response = parse_and_clean_output(full_response_text)
        for word in cleaned_response.split():
            yield 'text', word + " "
            time.sleep(0.02)
        yield 'sources', sources
    except Exception as e:
        logging.error(f"An error occurred while handling the query: {e}", exc_info=True)
        yield 'error', f"An internal error occurred while generating the response."
        yield 'sources', []

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
            logging.info("Initializing components for database rebuild...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model_build = SentenceTransformer(EMBEDDING_MODEL_NAMES, device=device)
            client_build = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, is_persistent=True, anonymized_telemetry=False))
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