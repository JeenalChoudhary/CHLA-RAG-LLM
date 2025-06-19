# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import logging
from llama_index.llms.ollama import Ollama
import nltk

# ---- Configuration and Constraints ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PDF_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "docs")
DB_PATH = os.path.join(SCRIPT_DIRECTORY, "db")
COLLECTION_NAME = "example_health_docs"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

SENTENCES_PER_CHUNK = 4
BROAD_TERMS = ['Leukemia', 'Acute Lymphocytic Leukemia (ALL)', 'Acute Myeloid Leukemia (AML)',
               'Asthma', 'Abdominal Aortic Aneurysm', 'ACL Injury', 'Strabismus',
               'Peripheral Arterial Disease (PAD)', 'Abdominal Pain', 'Acoustic Neuroma',
               'Acute Bronchitis', 'Acute Liver Failure', 'Abscess'] # For query expansion
# CHUNK_SIZE = 256
# CHUNK_OVERLAP = 30

# ---- Setup NLTK ----
try:
    nltk.data.find('tokenizer/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' model...")
    nltk.download('punkt')
    logging.info("NLTK 'punkt' model downloaded successfully.")
    
# ---- Text extraction and cleaning ----
def clean_pdf_text(text):
    text = re.sub(r'MAYO\s*CLINIC.*(?:\n.*)*?(?:Request an Appointment|Log in|Symptoms &|causes|Diagnosis &|treatment|Doctors &|departments)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2,2} [AP]M', '', text)
    text = re.sub(r'\d+/\d+', '', text)
    
    text = re.sub(r'Request an appointment', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Print', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Show references', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Advertisement', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Close', '', text, flags=re.IGNORECASE)
    text = re.sub(r'By Mayo Clinic Staff', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Enlarge image', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Image \d+\]', '', text)
    
    text = re.sub(r'From Mayo Clinic to your inbox.*Subscribe!', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not re.match(r'^(Overview|Symptoms|Causes|Risk factors|Complications|Prevention|Diagnosis|Treatment|Doctors & departments|When to see a doctor)\s*↓*$', line.strip())]
    text = '\n'.join(cleaned_lines)
    return text.strip()

def load_and_process_pdfs(directory):
    documents = []
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        return documents

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(directory, filename)
            logging.info(f"Processing PDF: {filename}")
            try:
                doc = fitz.open(path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                cleaned_text = clean_pdf_text(full_text)
                sentences = nltk.sent_tokenize(cleaned_text)
                for i in range(0, len(cleaned_text), SENTENCES_PER_CHUNK):
                    chunk = cleaned_text[i:i + SENTENCES_PER_CHUNK]
                    documents.append({
                        "content":chunk,
                        "metadata":{
                            "source": filename,
                            "language": "english",
                            "category": None
                        }
                    })
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    logging.info(f"Successfully processed {len(os.listdir(directory))} PDFs into {len(documents)} chunks.")
    return documents


#---- Vector database setup ----
def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device '{device}'")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

def setup_chromadb(documents, embedding_model, rebuild=False):
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if rebuild:
        try:
            if COLLECTION_NAME in [c.name for c in client.list_collections()]:
                logging.info(f'Rebuilding DB... Deleting existing collection {COLLECTION_NAME}')
                client.delete_collection(name=COLLECTION_NAME)
        except Exception as e:
            logging.error(f"Error in deleting collection: {e}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    if collection.count() == 0 or rebuild:
        if not documents:
            logging.warning(f"There are 0 documents to populate the database with. Please check you have added documents to the {PDF_DIRECTORY} directory.")
            return collection
        logging.info(f"Database is empty or rebuild is forced. Populating with new data...")
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            embeddings = embedding_model.encode(batch, convert_to_tensor=True).tolist()
            all_embeddings.extend(embeddings)
            logging.info(f"Embedded batch {i//batch_size+1}/{(len(contents) + batch_size - 1)//batch_size}")
        
        collection.add(
            embeddings=all_embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(documents)} chunks to ChromaDB")
    else:
        logging.info("Existing documents found and loaded")
    
    return collection


#---- Retrieval and Generation ----
def retrieve_context(query, collection, embedding_model, n_results=2):
    logging.info(f"Retrieving context for query: '{query}'")
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    context = "\n\n---\n\n".join(results['documents'][0])
    sources = sorted(list(set(meta['source'] for meta in results['metadatas'][0])))
    logging.info(f"Retrieved context from sources: {list(sources)}")
    return context, sources

def generate_answer(query, context, model_name):
    prompt_template = f"""
    You are a helpful medical information assistant. 
    Your task is to answer the user's question based *only* on the provided context from Mayo Clinic documents. 
    If the information is not in the context, explicitly state that you cannot answer the question with the given information and do not list any sources. 
    
    CONTEXT: {context}
    
    QUESTION: {query}
    
    ANSWER:
    """
    logging.info(f"Sending prompt to Ollama model via LlamaIndex: {model_name}")
    llm = Ollama(model=model_name, request_timeout=120.0, temperature=0)
    response_iter = llm.stream_complete(prompt_template)
    for token in response_iter:
        yield token.delta
        
def generate_query_expansion_options(query, model_name="gemma3:1b"):
    if not any(term in query.lower() for term in BROAD_TERMS):
        return []
    
    prompt = f"""
    A user has provided the following health-related query: {query}
    This is too broad. Your task is to generate 3-4 clarifying questions to help them narrow down their search.
    These questions should be presented as distinct, actionable options.
    For example, for "diabetes", you could suggest "What are the symptoms of diabetes?" or "How is diabetes treated?".
    Return ONLY a Python-parseable list of strings. Do not include any other text or explanation.
    
    Example format:
    ["What are the symptoms of {query}", "What are the treatment options for {query}", "What are the risk factors for {query}"]
    """
    try:
        logging.info(f"Generating query expansion options for '{query}' using {model_name}")
        llm = Ollama(model=model_name, request_timeout=120.0, temperature=0)
        response = llm.complete(prompt)
        match = re.search(r'\[\s*".*?"\s*(,\s*".*?"\s*)*\]', response)
        if match:
            options = eval(match.group(0))
            logging.info(f"Generated options: {options}")
            return options
        else:
            logging.warning("Could not parse the query expansion options from LLM response.")
            return []
    except Exception as e:
        logging.error(f"Error during query expansion: {e}")
        return []