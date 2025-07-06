import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import logging
import json
from langdetect import detect, LangDetectException
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
TOPIC_SUMMARY_CACHE = os.path.join(SCRIPT_DIRECTORY, "topic_summary.md")
PROCESSED_DOCS_CACHE = os.path.join(CACHE_DIR, "processed_docs.json")
COLLECTION_NAME = "example_health_docs"
EMBEDDING_MODEL_NAMES = {"english_medical":"NeuML/pubmedbert-base-embeddings",
                         "multilingual":"paraphrase-multilingual-mpnet-base-v2"}
_embedding_models = {}
_llm_models = {}
SENTENCES_PER_CHUNK = 6
STRIDE = 2

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
    text = re.sub(r"© 2000-2027 The StayWell Company, LLC.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"This information is not intended as a substitute for professional medical care.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text

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
                try:
                    detected_language = detect(cleaned_text[:100])
                    logging.info(f"Detected language: {detected_language} for {filename}")
                except LangDetectException:
                    logging.warning(f"Could not detect language for {filename}. Defaulting to English ('en').")
                    detected_language = 'en'
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

def get_llm(model_name, temperature=0.25, timeout=300.0):
    cache_key = (model_name, temperature)
    if cache_key not in _llm_models:
        logging.info(f"Loading LLM model: '{model_name}' with temperature={temperature} for the first time.")
        _llm_models[cache_key] = Ollama(model=model_name, temperature=temperature, request_timeout=timeout)
    return _llm_models[cache_key]

def generate_hypothetical_document(query, model_name="gemma3:1b-it-qat"):
    prompt = f"""
    Generate a single, comprehensive paragraph for a medical education document in response to the user's query: '{query}'.
    This paragraph will be used to improve database search retrieval for answer generation.
    
    The paragraph must be authoritative and detailed. Structure it to include the following, in a logical flow:
    1. Start with a clear **definition** of the main subject.
    2. Describe the primary **indications and purpose** (why and when it is used).
    3. Incorporate relevant **technical details**, such as different types, materials, or procedural aspects.
    4. Detail its **applications**, including the types of substances administered (e.g., nutritional formulas, medications, etc.).
    5. Conclude by discussing **management, key monitoring parameters, and potential complications**.
    
    Ensure the text is dense with relevant medical terminology and flows in a single, cohesive paragraph. Do not include any introductory or concluding text, just the paragraph itself.
    
    HYPOTHETICAL ANSWER:
    """
    try:
        logging.info(f"Generating hypothetical document for query: '{query}'")
        llm = get_llm(model_name)
        response = llm.complete(prompt)
        logging.info(f"Successfully generated hypothetical document. Hypothetical answer: {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error generating hypothetical document: {e}")
        return query

#---- Retrieval and Generation ----
def retrieve_context(query, collection, language, n_results=7):
    if language == "en":
        model_key = "english_medical"
        where_filter = {"$and": [{"language": {"$eq": "en"}}, {"model": {"$eq": "english_medical"}}]}
    else:
        model_key = "multilingual"
        where_filter = {"$and": [{"language": {"$eq": language}}, {"model": {"$eq": "multilingual"}}]}
    embedding_model = get_embedding_model(model_key)
    logging.info(f"Retrieving context for {language} query: '{query}' using '{model_key}' model.")
    hypothetical_document = generate_hypothetical_document(query)
    query_embedding = embedding_model.encode(hypothetical_document).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where_filter)
    if not results['documents'] or not results['documents'][0]:
        logging.warning(f"No documents found for language {language}. Retrieval failed.")
        return "", []
    context = "\n\n---\n\n".join(results['documents'][0])
    sources = sorted(list(set(meta['source'] for meta in results['metadatas'][0])))
    logging.info(f"Retrieved context from sources: {list(sources)}")
    return context, sources

def generate_answer(query, context, model_name, language_name):
    language_map = {"es": "Spanish", "en": "English", "ar": "Arabic", "zh":"Chinese", "hy":"Armenian", "fa":"Farsi/Persian", 
                    "ht":"Hatian/French-Creole", "hi":"Hindi", "ja":"Japanese", "ko":"Korean", "pa":"Punjabi", "ru":"Russian",
                    "ph":"Tagalog", "vi":"Vietnamese"}
    friendly_language_name = language_map.get(language_name.lower(), language_name)
    prompt_template = f"""
    You are a friendly and empathetic medical information assistant from Children's Hospital Los Angeles.
    Your task is to answer the user's question in a clear, simple, and reassuring way, based *only* on the provided context.
    ***IMPORTANT: You must write your entire answer in {friendly_language_name}.***
    - If the user uses informal language (like "puffer things"), acknowledge it and use the correct medical term in your answer (e.g., "The 'puffer thing' you mentioned is called on inhaler...").
    - If the context does not contain the answer, explicity state that you cannot answer the question with the provided information (in {friendly_language_name}) and do not list any sources.
    - Structure your answer with paragraphs and bullet points if it makes it easier to read.
    - Always base your answer *strictly* on the provided context. Do not use outside knowledge.
    
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

def handle_query(query, collection, model_name):
    logging.info(f"Handling specific query with RAG: '{query}'")
    try:
        query_lang_code = detect(query)
    except LangDetectException:
        logging.warning("Could not detect query language. Defaulting to English ('en').")
        query_lang_code = "en"
    context, sources = retrieve_context(query, collection, query_lang_code)
    if not context.strip():
        logging.warning("Retrieval returned empty context. The model will likely be unable to answer.")
        def empty_answer():
            yield "I could not find any information related to your question in the documents I have access to."
        return empty_answer(), []
    answer_stream = generate_answer(query, context, model_name, query_lang_code)
    return answer_stream, sources

def generate_topic_summary(collection, model_name):
    if os.path.exists(TOPIC_SUMMARY_CACHE):
        logging.info(f"Loading topic summary from cache: {TOPIC_SUMMARY_CACHE}")
        with open(TOPIC_SUMMARY_CACHE, 'r', encoding='utf-8') as f:
            return f.read()
    logging.info("No topic summary cache found. Generating a new topic summary with the LLM...")
    db_contents = collection.get()
    metadatas = db_contents.get('metadatas')
    if not metadatas:
        logging.error("Could not retrieve documents from the database to generate a summary.")
        return "Error: Could not retrieve document topics."
    unique_sources = sorted(list(set(meta['source'] for meta in metadatas)))
    cleaned_topics = []
    for source in unique_sources:
        topic = re.sub(r'(_English|_202\d)?\.pdf', '', source, flags=re.IGNORECASE)
        topic = topic.replace('_', ' ').replace('ALL', '(ALL)').replace('AML', '(AML)')
        cleaned_topics.append(topic)
    topics_text = "\n".join(f"- {topic}" for topic in cleaned_topics)
    logging.info(f"Topics to summarize: {topics_text}")
    prompt = f"""
    You are a helpful assistant. Based on the following list of medical document titles, please generate a clean, user-friendly, and concise bulleted list of the main health topics covered.
    Group related topics together under a clear, bolded heading (e.g., **Luekemia**). Do not use more than 5-6 top-level categories.
    The final output should be formatted in Markdown. Do not include any introductory or concluding text, just the Markdown list.
    
    DOCUMENT TITLES:
    {topics_text}
    
    CONCISE TOPIC LIST:
    """
    try:
        llm = get_llm(model_name, request_timeout=1200.0)
        response = llm.complete(prompt)
        with open(TOPIC_SUMMARY_CACHE, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logging.info(f"Topic summary generated and cached to {TOPIC_SUMMARY_CACHE}")
        return response.text
    except Exception as e:
        logging.error(f"Error in generating topic summary: {e}")
        return "I am currently unable to generate a list of topics as an error occurred. Please try asking a specific question."

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
        if os.path.exists(TOPIC_SUMMARY_CACHE):
            logging.info(f"Deleting old topic summary cache: {TOPIC_SUMMARY_CACHE}")
            os.remove(TOPIC_SUMMARY_CACHE)
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
            english_docs = [doc for doc in docs if doc['metadata']['language'] == 'en']
            other_docs = [doc for doc in docs if doc['metadata']['language'] != 'en']
            english_embedding = get_embedding_model("english_medical")
            other_embedding = get_embedding_model("multilingual")
            batch_size = 128
            if english_docs:
                total_english_batches = (len(english_docs) + batch_size - 1) // batch_size
                logging.info(f"Embedding {len(english_docs)} English documents with PubMedBERT in {total_english_batches} batches...")
                for i in range(0, len(english_docs), batch_size):
                    batch_num = (i // batch_size) + 1
                    logging.info(f"Processing English batch {batch_num}/{total_english_batches}...")
                    batch = english_docs[i:i + batch_size]
                    contents = [doc['content'] for doc in batch]
                    metadata = [doc['metadata'] for doc in batch]
                    for meta in metadata:
                        meta['model'] = 'english_medical'
                    ids = [f"en_doc_{i+j}" for j in range(len(batch))]
                    embeddings = english_embedding.encode(contents, show_progress_bar=True).tolist()
                    collection.add(embeddings=embeddings, documents=contents, metadatas=metadata, ids=ids)
            if other_docs:
                total_other_batches = (len(other_docs) + batch_size - 1) // batch_size
                logging.info(f"Embedding {len(other_docs)} non-English documents with multilingual model in {total_other_batches} batches...")
                for i in range(0, len(other_docs), batch_size):
                    current_batch_num = (i // batch_size) + 1
                    logging.info(f"Processing multilingual batch {current_batch_num}/{total_other_batches}...")
                    batch = other_docs[i:i + batch_size]
                    contents = [doc['content'] for doc in batch]
                    metadatas = [doc['metadata'] for doc in batch]
                    for meta in metadatas:
                        meta['model'] = 'multilingual'
                    ids = [f"multi_doc_{i+j}" for j in range(len(batch))]
                    embeddings = other_embedding.encode(contents, show_progress_bar=True).tolist()
                    collection.add(embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids)
            logging.info(f"Hybrid embedding complete. Total documents in DB: {collection.count()}")
        else:
            logging.info(f"Database already populated. Skipping ingestion.")
        logging.info("RAG backend setup complete.")
    else:
        logging.warning("No documents were loaded. The application might not function as expected.")