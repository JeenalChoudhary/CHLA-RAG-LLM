import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import logging
import json
import langid
from llama_index.llms.ollama import Ollama
import nltk
import argparse
import shutil

# ---- Configuration and Constraints ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PDF_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "docs")
DB_PATH = os.path.join(SCRIPT_DIRECTORY, "db")
CACHE_DIR = os.path.join(SCRIPT_DIRECTORY, "cache")
PROCESSED_DOCS_CACHE = os.path.join(CACHE_DIR, "processed_docs.json")
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

# ---- Setup NLTK ----
try:
    nltk.data.find('tokenizer/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' model...")
    nltk.download('punkt')
    logging.info("NLTK 'punkt' model downloaded successfully.")
    
# ---- Text extraction and cleaning ----
def clean_pdf_text(text):
    text = re.sub(r"Children's\s+Hospital\s+LOS ANGELES", "", text, flags=re.IGNORECASE)
    text = re.sub(r".*4650 Sunset Blvd\., Los Angeles, CA 90027.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}.*LLC.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Disclaimer:.*|This information is not intended as a substitute for professional medical care.*|This information is intended for general knowledge.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text

def detect_language(text):
    logging.info("Language detection is disabled. Defaulting to English ('en').")
    return "en"
    # try:
    #     lang_code, confidence = langid.classify(text)
    #     logging.info(f"Detected language: {lang_code} with {confidence} confidence.")
    #     return lang_code
    # except Exception as e:
    #     logging.error(f"Failed to detect language: {e}. Defaulting to English ('en').")
    #     return "en"
        
def load_and_process_pdfs(directory):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(PROCESSED_DOCS_CACHE):
        logging.info(f"Loading processes documents from cache: {PROCESSED_DOCS_CACHE}")
        with open(PROCESSED_DOCS_CACHE, 'r', encoding='utf-8') as f:
            logging.info(f"Successfully loaded {len(json.load(f))} documents from cache.")
            return json.load(f)

    logging.info("No cache was found. Starting the PDF processing from scratch.")
    documents = []
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        return documents
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            path = os.path.join(directory, filename)
            logging.info(f"Processing PDF: {filename}")
            try:
                doc = fitz.open(path)
                full_text = "".join(page.get_text() for page in doc)
                cleaned_text = clean_pdf_text(full_text)
                if not cleaned_text:
                    logging.warning(f"No content left in {filename} after cleaning. Skipping this PDF.")
                    continue
                detected_language = detect_language(cleaned_text[:500])
                logging.info(f"Detected language: {detected_language} for {filename}")
                sentences = nltk.sent_tokenize(cleaned_text)
                for i in range(0, len(sentences) - SENTENCES_PER_CHUNK + 1, STRIDE):
                    chunk = " ".join(sentences[i:i + SENTENCES_PER_CHUNK])
                    documents.append({
                        "content":chunk,
                        "metadata":{"source": filename, "language": detected_language, "start_sentence_index": i}
                    })
                if (len(sentences) % STRIDE != 0) and (len(sentences) > SENTENCES_PER_CHUNK):
                    last_chunk_start = len(sentences) - SENTENCES_PER_CHUNK
                    if not documents or last_chunk_start > documents[-1]['metadata']['start_sentence_index']:
                        chunk = " ".join(sentences[last_chunk_start:])
                        documents.append({
                        "content":chunk,
                        "metadata":{"source": filename, "language": detected_language, "start_sentence_index": last_chunk_start}
                        })
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    
    logging.info(f"Successfully processed {len(os.listdir(directory))} PDFs into {len(documents)} chunks.")
    with open(PROCESSED_DOCS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    logging.info(f"Saved processed chunks to cache: {PROCESSED_DOCS_CACHE}")
    return documents

#---- Vector database setup ----
def get_embedding_model(model_key="english_medical"):
    if model_key not in _embedding_models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = EMBEDDING_MODEL_NAMES[model_key]
        logging.info(f"Loading embedding model: '{model_name}' ({model_key}) on device {device}")
        _embedding_models[model_key] = SentenceTransformer(model_name, device=device)
    return _embedding_models[model_key]

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading reranker model '{RERANKER_MODEL_NAME}' on device {device}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=device)
    return _reranker_model

def get_llm(model_name, temperature=0.25, timeout=300.0):
    cache_key = (model_name, temperature)
    if cache_key not in _llm_models:
        logging.info(f"Loading LLM model: '{model_name}' with temperature={temperature} for the first time.")
        _llm_models[cache_key] = Ollama(model=model_name, temperature=temperature, request_timeout=timeout)
    return _llm_models[cache_key]

def generate_hypothetical_document(query, conversation_history="",model_name="gemma3:1b-it-qat"):
    query_lower = query.lower()
    word_count = len(query_lower.split())
    is_question = any(query_lower.startswith(q) for q in ['how', 'what', 'when', 'why', 'can', 'do', 'is']) or '?' in query_lower
    history_prompt_section = ""
    if conversation_history:
        history_prompt_section = f"""
        Here is the recent conversation history. Use it to understand the full context of the user's latest query.
        ---
        {conversation_history}
        ---
        """
    if word_count < 4 and not is_question:
        prompt = f"""
        You are a medical writer creating a hypothetical document for a database search.
        {history_prompt_section}
        Your task is to generate a single, comprehensive paragraph about the user's query: '{query}'.
        
        **Instructions:** Use the conversation history to tailor the paragraph. For example, if the history is about "a newborn baby" and the user's query is "feeding tubes", the hypothetical document should be about **feeding tubes specifically for newborns.**
        
        The paragraph must be authoritative and detailed, including:
        1. A clear definition of the main subject.
        2. Its primary purpose and indications.
        3. Relevant technical details and types.
        4. Key management considerations and potential complications and risks.
        
        Write only the self-contained paragraph. DO NOT add any introductory text.
        
        HYPOTHETICAL ANSWER:
        """
        logging.info(f"Generating definitional document for simpler terms: '{query}'.")
    else:
        prompt = f"""
        You are a medical writer creating a hypothetical document for a database search.
        {history_prompt_section}
        Your task is to generate an ideal, detailed paragraph that directly and hypothetical answers and/or extends the user's question: '{query}'.
        
        **Instructions:** Use the conversation history to understand and resolve the user's query. For example, if the history is about "g-tubes" and the query is "how do I clean it?", the hypothetical document should be a detailed answer or extension about **the specific procedure for cleaning a g-tube.**
        
        Elaborate on the answer by explaining the 'how' and 'why', including details about related procedures, or what a caregiver should know.
        The final paragraph must be self-contained, dense, and cohesive. Do not use lists or add introductory or concluding phrases.
        
        HYPOTHETICAL DOCUMENT:
        """
    try:
        llm = get_llm(model_name)
        response = llm.complete(prompt)
        logging.info(f"Successfully generated hypothetical document. Hypothetical answer: {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error generating hypothetical document: {e}")
        return query

#---- Retrieval and Generation ----
def retrieve_context(query, collection, language, conversation_history="", n_results=FINAL_CONTEXT_COUNT):
    model_key = "english_medical"
    where_filter = {"$and": [{"language": {"$eq": "en"}}, {"model": {"$eq": "english_medical"}}]}
    embedding_model = get_embedding_model(model_key)
    logging.info(f"Retrieving context for {language} query: '{query}' using '{model_key}' model.")
    hypothetical_document = generate_hypothetical_document(query, conversation_history)
    query_embedding = embedding_model.encode(hypothetical_document).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=INITIAL_RETRIEVAL_COUNT, where=where_filter)
    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    if not retrieved_docs:
        logging.warning(f"No documents found for language {language}. Retrieval failed.")
        return "", []
    logging.info(f"Retrieved {len(retrieved_docs)} candidate documents for reranking.")
    reranker = get_reranker_model()
    rerank_pairs = [[query, doc] for doc in retrieved_docs]
    scores = reranker.predict(rerank_pairs, show_progress_bar=True)
    scored_docs = list(zip(scores, retrieved_docs, retrieved_metadata))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_reranked_docs = scored_docs[:n_results]
    final_docs = [doc for score, doc, meta in top_reranked_docs]
    final_metadata = [meta for score, doc, meta in top_reranked_docs]
    context = "\n\n---\n\n".join(final_docs)
    sources = sorted(list(set(meta['source'] for meta in final_metadata)))
    logging.info(f"Reranked and selected top {len(final_docs)} documents.")
    logging.info(f"Retrieved final context from sources: {list(sources)}")
    return context, sources

def generate_answer(query, context, model_name, conversation_history=""):
    friendly_language_name = "English"
    history_prompt = ""
    if conversation_history:
        history_prompt = f"""
        Here is the recent conversation history. Use it to understand the context of the user's latest question, if relevant:
        ----
        {conversation_history}
        ----
        """
    prompt_template = f"""
    You are an expert medical educator and assistant at Children's Hospital Los Angeles. Your purpose is to provide clear, comprehensive, and reassuring answers to patient families who have an average 8th-grade reading level, based STRICTLY on the provided context.
    
    {history_prompt}
    
    ---
    **BEHAVIORAL GUARDRAILS (Follow these always):**
    - **NEVER** break character. You are an assistant from CHLA, not a generic AI.
    - **NEVER** mention that you are an AI, a language model, or a chatbot.
    - **NEVER** add a disclaimer of any kind.
    - **NEVER** provide external website links or suggest resources not provided in the context.
    - **NEVER** praise the user's question (e.g., do not say "That's a great question"). You can use a reassuring tone (e.g., "That's an important question...") for serious topics as instructed below.
    ---
    
    **ANSWERING RULES:**
    
    **RULE 0: Handle Vague/Short Queries ONLY.**
    - **CONDITION:** Check if the user's query, '{query}', has **FEWER THAN 4 words**.
    - **ACTION:** Provide a 1-2 paragraph overview based on the context, then append the exact phrase: "This is a general overview. To give you more specific information, could you tell me more about your question? For example, you could ask about:" and list three follow-up topics. **STOP** and do not follow other rules.
    
    **RULE 1: Answer All Specific Questions Directly.**
    - **CONDITION:** If the user's query has **4 OR MORE words**.
    - **ACTION:** Synthesize a direct, complete answer from the context. Do not ask for clarification. Follow the content rules below.
    
    **CONTENT RULES (For Rule 1):**
    - **Tone:** For serious topics, begin with a short, reassuring sentence (e.g., "It's wise to prepare for that. Based on the provided information, here are the steps...").
    - **Procedures:** If the context describes a procedure, you MUST list ALL steps in order.
    - **Completeness:** Never give a simple "yes" or "no" answer. Always explain the "how" and "why" using details from the context.
    - **Missing Info:** If the context doesn't have the answer, state that clearly and simply: "I could not find specific information on that topic in the provided documents."
    
    ***IMPORTANT: You must write your entire answer in {friendly_language_name}.***
    
    Context:
    {context}
    
    Question: {query}
    
    Answer in {friendly_language_name}:
    """
    logging.info(f"Sending prompt to Ollama model via LlamaIndex: {model_name}")
    llm = get_llm(model_name, temperature=0)
    response_iter = llm.stream_complete(prompt_template)
    for token in response_iter:
        yield token.delta

def handle_query(query, collection, model_name, conversation_history=[]):
    logging.info(f"Handling specific query with RAG: '{query}'")
    query_lang_code = detect_language(query)
    context, sources = retrieve_context(query, collection, query_lang_code)
    if not context.strip():
        logging.warning("Retrieval returned empty context. The model will likely be unable to answer.")
        def empty_answer():
            yield "I could not find any information related to your question in the documents I have access to."
        return empty_answer(), []
    answer_stream = generate_answer(query, context, model_name, conversation_history=conversation_history)
    return answer_stream, sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG backend for health documents.")
    parser.add_argument("--rebuild", action="store_true", help="Force a complete rebuild of the database and all cahced files.")
    args = parser.parse_args()
    
    if args.rebuild:
        logging.info("---- REBUILD FLAG DETECTED ----")
        logging.info("Deleting old database and caches for a clean build")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        
    logging.info("Starting RAG backend setup...")
    docs = load_and_process_pdfs(PDF_DIRECTORY)
    
    if docs:
        client = chromadb.PersistentClient(path=DB_PATH)
        if args.rebuild and COLLECTION_NAME in [c.name for c in client.list_collections()]:
            client.delete_collection(name=COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        if collection.count() == 0:
            logging.info("Database is empty. Populating with hybrid embedding strategy...")
            # english_docs = [doc for doc in docs if doc['metadata']['language'] == 'en']
            # other_docs = [doc for doc in docs if doc['metadata']['language'] != 'en']
            english_embedding = get_embedding_model("english_medical")
            # other_embedding = get_embedding_model("multilingual")
            batch_size = 512
            # if english_docs:
            total_english_batches = (len(docs) + batch_size - 1) // batch_size
            logging.info(f"Embedding {len(docs)} English documents with PubMedBERT in {total_english_batches} batches...")
            for i in range(0, len(docs), batch_size):
                batch_num = (i // batch_size) + 1
                logging.info(f"Processing English batch {batch_num}/{total_english_batches}...")
                batch = docs[i:i + batch_size]
                contents = [doc['content'] for doc in batch]
                metadata = [doc['metadata'] for doc in batch]
                for meta in metadata:
                    meta['model'] = 'english_medical'
                ids = [f"en_doc_{i+j}" for j in range(len(batch))]
                embeddings = english_embedding.encode(contents, show_progress_bar=True).tolist()
                collection.add(embeddings=embeddings, documents=contents, metadatas=metadata, ids=ids)
            # if other_docs:
            #     total_other_batches = (len(other_docs) + batch_size - 1) // batch_size
            #     logging.info(f"Embedding {len(other_docs)} non-English documents with multilingual model in {total_other_batches} batches...")
            #     for i in range(0, len(other_docs), batch_size):
            #         current_batch_num = (i // batch_size) + 1
            #         logging.info(f"Processing multilingual batch {current_batch_num}/{total_other_batches}...")
            #         batch = other_docs[i:i + batch_size]
            #         contents = [doc['content'] for doc in batch]
            #         metadatas = [doc['metadata'] for doc in batch]
            #         for meta in metadatas:
            #             meta['model'] = 'multilingual'
            #         ids = [f"multi_doc_{i+j}" for j in range(len(batch))]
            #         embeddings = other_embedding.encode(contents, show_progress_bar=True).tolist()
            #         collection.add(embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids)
            logging.info(f"Hybrid embedding complete. Total documents in DB: {collection.count()}")
        else:
            logging.info(f"Database already populated. Skipping ingestion.")
        logging.info("RAG backend setup complete.")
    else:
        logging.warning("No documents were loaded. The application might not function as expected.")