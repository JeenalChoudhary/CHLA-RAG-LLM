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
INITIAL_RETRIEVAL_COUNT = 25
FINAL_CONTEXT_COUNT = 7

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
    text = re.sub(r"© 2000-2027 The StayWell Company, LLC.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"This information is not intended as a substitute for professional medical care.*", "", text, flags=re.IGNORECASE)
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

def generate_hypothetical_document(query, model_name="gemma3:1b-it-qat"):
    query_lower = query.lower()
    word_count = len(query_lower.split())
    is_question = any(query_lower.startswith(q) for q in ['how', 'what', 'when', 'why', 'can', 'do', 'is']) or '?' in query_lower
    if word_count < 5 and not is_question:
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
        logging.info(f"Generating definitional document for simpler terms: '{query}'.")
    else:
        prompt = f"""
        You are an expert medical writer. Your task is to generate an ideal, detailed paragraph that directly and hypothetically answers the user's question. 
        This paragraph will be used to find the most relevant information in a medical database.
        
        The user's question is: '{query}'
        
        **Instructions:**
        1. **Directly Answer:** Begin by creating a hypothetical, direct answer to the question based on common medical knowledge.
        2. **Elaborate:** Expand on the answer by explaining the 'how' and 'why'. Include details about related procedures, different methods, or what a caregiver should know. For example, if a question is about talking with a trach tube, explain the mechanincs of using a speaking valve or finger occlusion.
        3. **Maintain Focus:** Keep the entire paragraph focused on answering the specific question. **DO NOT** just define the general topic.
        4. **Format:** Write a single, dense, and cohesive paragraph. NEVER use lists or headings. NEVER add any introductory phrases like "The answer is...".
        
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
def retrieve_context(query, collection, language, n_results=FINAL_CONTEXT_COUNT):
    model_key = "english_medical"
    where_filter = {"$and": [{"language": {"$eq": "en"}}, {"model": {"$eq": "english_medical"}}]}
    embedding_model = get_embedding_model(model_key)
    logging.info(f"Retrieving context for {language} query: '{query}' using '{model_key}' model.")
    hypothetical_document = generate_hypothetical_document(query)
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

def generate_answer(query, context, model_name, language_name, conversation_history=""):
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
    You are an expert medical educator and assistant at Children's Hospital Los Angeles. Your purpose is to act as a guide, providing clear, comprehensive, and reassuring answers to patient families who have an average 8th-grade reading level.
    You MUST answer the user's question based STRICTLY on the provided context. NEVER include an introduction sentence or explain your reasoning.
    
    {history_prompt}
    
    Follow these rules meticulously to create the best possible answer:
    
    **RULE 0: Handle Vague/Short Queries ONLY.**
    - **CONDITION:** Check if the user's query, '{query}', has **FEWER THAN 4 words**.
    - **ACTION (if the condition is met):**
        1. Synthesize a breif, helpful overview of the topic in 1-2 paragraphs. The goal is to answer "What is {query}?" in simple terms.
        2. Append the following text: "This is a general overview. To give you more specific information, could you tell me more about your question? For example, you can ask about:"
        3. Present three clarifying follow-up prompts as a bulleted or numbered list (e.g., Symptoms and warning signs; Home care or treatment steps; When to call the doctor for {query}).
        4. **STOP** and do not follow any other rules.
    ----
    **RULE 1: Answer All Specific Questions Directly.**
    - **Condition:** If the user's query has **4 OR MORE words**, you must provide a direct and comprehensive answer.
    - **Action (if the condition is met):** Synthesize information from ALL relevant context passages to form a single, complete, and seamless answer. **DO NOT** use phrases like "This is a general overview" or try to ask clarifying questions. Answer the user's question completely.
    ----
    2. **Tone and Introduction:** For serious or stressful topics (like choking or asthma attacks), ALWAYS begin with a short, reassuring sentence that acknowledges the user's question. Example: "That's a very important question, and it is smart to prepare. Here are the steps to follow based on the provided information."
    3. **Emergency Actions:** For any emergency, first synthesize all immediate actions from the context. State the most critical step (like "**First, call 911 immediately.**") and then detail any "while-you-wait" steps, such as using a rescue inhaler, as described in the documents.
    4. **Complete Procedures:** When the context describes a procedure (like first aid), you MUST extract and list **ALL** of the steps in order from beginning to end. Use a numbered list. A partial or summarized procedure is a failed answer. Be exhaustive.
    5. **Comprehensive Answers:** NEVER give a simple "Yes" or "No." After providing a direct answer, you MUST thoroughly explain the "how" and "why" by synthesizing all relevant details from the context. For example, after answering "Yes, a child can talk with a trach tube," you must then immediately explain the methods described in the text, such as speaking valves or finger occlusion.
    6. **Handling Missing Information:** If the context truly does not contain the information to answer the question, state that clearly. For example: "Based on the documents I have, I cannot find specific information on that topic." Do not list sources if you cannot answer.
    
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
    answer_stream = generate_answer(query, context, model_name, query_lang_code)
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