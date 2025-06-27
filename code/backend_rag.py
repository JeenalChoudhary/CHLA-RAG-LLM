import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import logging
import json
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
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
SPECIAL_QUERY_TRIGGER = "what can you teach me?"
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
    # lines = text.split('\n')
    # unwanted_patterns = [r"What is this test\?", r"Does this test have other names\?", r"Why do I need this test\?", r"What other tests might I have along with this test\?",
    #     r"What do my test results mean\?", r"How is this test done\?", r"Does this test pose any risks\?", r"What might affect my test results\?",
    #     r"How do I get ready for this test\?", r"What is an? .*\?", r"What causes .*\?", r"What are the symptoms of .*\?", r"How is .* diagnosed\?", r"How is .* treated\?",
    #     r"What are possible complications of .*\?", r"Can .* be prevented\?", r"Who is at risk for .*\?", r"Are there screening tests for .*\?",
    #     r"What tests might I have after being diagnosed\?", r"Key points about .*", r"Symptoms of .*", r"Diagnosing .*", r"Treating .*", r"Causes of .*", r"Medicines",
    #     r"Triggers", r"Asthma symptoms", r"Asthma triggers", r"Asthma symptoms and triggers", r"Physical activity", r"Emergency information", r"Self-care tips",
    #     r"Recovery and follow-up", r"Getting your test results", r"What is cancer\?", r"What is leukemia\?", r"What is ALL\?", r"How is ALL treated\?", r"What is AML\?",
    #     r"How is AML treated\?", r"Coping with ALL", r"Coping with AML", r"Subtypes of ALL", r"Subtypes of AML", r"What tests might I need\?", r"How blood or bone marrow is tested",
    #     r"Terms you may hear", r"In remission \(or complete remission\)", r"Minimal residual disease", r"Refractory (ALL|AML)", r"Relapsed \(recurrent\) (ALL|AML)",
    #     r"Types of treatment for (ALL|AML)", r"What is chemotherapy\?", r"When might chemotherapy be used for (ALL|AML)\?", r"How is chemotherapy given for (ALL|AML)\?",
    #     r"What is intrathecal chemotherapy\?", r"Treatment in the remission induction phase", r"Treatment in the consolidation \(intensification\) phase",
    #     r"Treatment in the maintenance phase", r"What are common side effects of chemotherapy\?", r"What is radiation therapy\?", r"When might radiation therapy be used for .*\?",
    #     r"Where is radiation therapy given\?", r"Getting ready for radiation therapy", r"During a radiation treatment session", r"During total body irradiation \(TBI\)",
    #     r"Possible side effects of radiation therapy", r"Short-term side effects", r"Long-term side effects", r"When is targeted therapy used for (ALL|AML)\?",
    #     r"Types of targeted therapy for (ALL|AML)", r"How targeted therapy is done", r"Possible side effects of targeted therapy", r"What is a stem cell transplant\?",
    #     r"When might a stem cell transplant be used for (ALL|AML)\?", r"Types of stem cell transplant", r"What happens during a stem cell transplant for (ALL|AML)",
    #     r"How stem cells are collected", r"From the blood", r"From the bone marrow", r"Having the transplant", r"Possible short-term side effects", r"Possible long-term side effects",
    #     r"Making a decision", r"Clinical trials for new treatments", r"Preparing for surgery", r"The day of surgery", r"During the surgery", r"After the surgery",
    #     r"Recovering at home", r"Risks and possible complications", r"Things to Note After the Procedure:", r"Dressing Care:", r"Activity:", r"Pain:", r"Increasing your agility",
    #     r"Returning to favorite activities", r"Imaging tests", r"Blood tests", r"Lumbar puncture", r"Bone marrow biopsy", r"Coping with fear", r"Working with your healthcare team",
    #     r"Learning about treatment options", r"Getting support", r"Working with your healthcare provider", r"Working with your healthcare providers", r"Partnering with your care team",
    #     r"Talk with your healthcare provider", r"Talking with your healthcare providers", r"Deciding on a treatment", r"Getting ready for treatment", r"Coping during treatment",
    #     r"Call 911", r"When to call .* healthcare provider", r"When should I call my healthcare provider\?", r"Return to the Emergency Department \(ED\) if:",
    #     r"For non-emergent questions:", r"For emergencies:", r"Next steps", r"Tips to help you get the most from a visit to your healthcare provider:", r"Instructions:",
    #     r"The Vision Center - Ophthalmology", r"Clinical Nutrition and Lactation Services", r"Pathology and Laboratory Medicine", r"Cancer and Blood Diseases Institute \(CBDI\)",
    #     r"Heart Institute", r"Division of Gastroentero(l|og)?y, Hepatology, and Nutrition", r"Nephrology", r"Division of Plastic and Maxillofacial Surgery",
    #     r"Division of Adolescent Medicine and Young Adult Medicine", r"Division of Clinical Immunology and Allergy", r"Interventional Radiology", 
    #     r"Interventional Radiology Contact Information:", r"Rehabilitation Services", r"Division of Otolaryngology - Head and Neck Surgery", r"Jackie and Gene Autry Orthopedic Center",
    #     r"Pulmonology and Sleep Disorders", r"Hepatology", r"Journey Stage", r"Follow-up", r"Hello Parents and Caregivers!", r"Let's get started!",
    #     r"DISASTER PREPAREDNESS ACTIVITIES FOR YOUR FAMILY", r"Activity #\d:.*", r"Game #\d:.*", r"Caregiver's tip:", r"Video Health Sheets TM", r"To watch the video:",
    #     r"Scan the QR code", r"Using your mobile device, scan the following code:", r"Go to the website:", r"www\.kramesvideo\.com", r"Enter the prescription code:", r"[A-Z0-9]{3}",
    #     r"Putting it all together", r"What to do", r"Going Back to School or Work:",r"Stage [1-4]"
    #     ]
    # combined_pattern = re.compile(r"^\s*(" + "|".join(unwanted_patterns) + r")\s*$", flags=re.IGNORECASE)
    # cleaned_lines = [line for line in lines if not combined_pattern.match(line.strip())]
    # return '\n'.join(cleaned_lines).strip()
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
                sentences = nltk.sent_tokenize(cleaned_text)
                for i in range(0, len(sentences) - SENTENCES_PER_CHUNK + 1, STRIDE):
                    chunk = " ".join(sentences[i:i + SENTENCES_PER_CHUNK])
                    documents.append({
                        "content":chunk,
                        "metadata":{"source": filename, "language": "english", "start_sentence_index": i}
                    })
                if (len(sentences) % STRIDE != 0) and (len(sentences) > SENTENCES_PER_CHUNK):
                    last_chunk_start = len(sentences) - SENTENCES_PER_CHUNK
                    if not documents or last_chunk_start > documents[-1]['metadata']['start_sentence_index']:
                        chunk = " ".join(sentences[last_chunk_start:])
                        documents.append({
                        "content":chunk,
                        "metadata":{"source": filename, "language": "english", "start_sentence_index": last_chunk_start}
                        })
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    
    logging.info(f"Successfully processed {len(os.listdir(directory))} PDFs into {len(documents)} chunks.")
    with open(PROCESSED_DOCS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    logging.info(f"Saved processed chunks to cache: {PROCESSED_DOCS_CACHE}")
    return documents

#---- Vector database setup ----
def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device '{device}'")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

def setup_chromadb(documents, embedding_model, rebuild=False):
    client = chromadb.PersistentClient(path=DB_PATH)
    if rebuild:
        if COLLECTION_NAME in [c.name for c in client.list_collections()]:
            logging.info(f'Rebuilding DB... Deleting existing collection {COLLECTION_NAME}')
            client.delete_collection(name=COLLECTION_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        logging.info(f"Database is empty or rebuild is forced. Populating with new data...")
        batch_size = 256
        for i in range(0, len(documents), batch_size):
            document_batch = documents[i:i + batch_size]
            contents_batch = [doc['content'] for doc in document_batch]
            metadata_batch = [doc['metadata'] for doc in document_batch]
            id_batch = [f"doc_{i+j}" for j in range(len(document_batch))]
            embeddings_batch = embedding_model.encode(contents_batch, show_progress_bar=True).tolist()
            collection.add(embeddings=embeddings_batch, documents=contents_batch, metadatas=metadata_batch, ids=id_batch)
            logging.info(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} to ChromaDB.")
        logging.info(f"Successfully added {len(documents)} documents to ChromaDB.")
    else:
        logging.info("Existing documents found and loaded")
    return collection

def generate_hypothetical_document(query, model_name="gemma3:1b"):
    prompt = f"""
    A user is asking the following question: "{query}"
    Please write a detailed, high-quality paragraph that answers this question as if it were from a reliable medical education document.
    Focus on addressing the key terms and concepts in the user's question.
    This will be used to find the most relevant documents in a database.
    
    HYPOTHETICAL ANSWER:
    """
    try:
        logging.info(f"Generating hypothetical document for query: '{query}'")
        llm = Ollama(model=model_name, request_timeout=60.0)
        response = llm.complete(prompt)
        logging.info("Successfully generated hypothetical document.")
        return response.text
    except Exception as e:
        logging.error(f"Error generating hypothetical document: {e}")
        return query

#---- Retrieval and Generation ----
def retrieve_context(query, collection, embedding_model, n_results=2):
    logging.info(f"Retrieving context for query: '{query}'")
    hypothetical_document = generate_hypothetical_document(query)
    query_embedding = embedding_model.encode(hypothetical_document).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    context = "\n\n---\n\n".join(results['documents'][0])
    sources = sorted(list(set(meta['source'] for meta in results['metadatas'][0])))
    logging.info(f"Retrieved context from sources: {list(sources)}")
    return context, sources

def generate_answer(query, context, model_name):
    prompt_template = f"""
    You are a friendly and empathetic medical information assistant from Children's Hospital Los Angeles.
    Your task is to answer the user's question in a clear, simple, and reassuring way, based *only* on the provided context.
    - If the user uses informal language (like "puffer things"), acknowledge it and use the correct medical term in your answer (e.g., "The 'puffer thing' you mentioned is called on inhaler...").
    - If the context does not contain the answer, explicity state that you cannot answer the question with the provided information and do not list any sources.
    - Structure your answer with paragraphs and bullet points if it makes it easier to read.
    - Always base your answer *strictly* on the provided context. Do not use outside knowledge.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    logging.info(f"Sending prompt to Ollama model via LlamaIndex: {model_name}")
    llm = Ollama(model=model_name, request_timeout=120.0, temperature=0)
    response_iter = llm.stream_complete(prompt_template)
    for token in response_iter:
        yield token.delta

def handle_query(query, collection, embedding_model, model_name="gemma3"):
    logging.info(f"Handling specific query with RAG: '{query}'")
    context, sources = retrieve_context(query, collection, embedding_model)
    if not context.strip():
        logging.warning("Retrieval returned empty context. The model will likely be unable to answer.")
        def empty_answer():
            yield "I could not find any information related to your question in the documents I have access to."
        return empty_answer(), []
    answer_stream = generate_answer(query, context, model_name)
    return answer_stream, sources

def generate_topic_summary(documents, model_name):
    if os.path.exists(TOPIC_SUMMARY_CACHE):
        logging.info(f"Loading topic summary from cache: {TOPIC_SUMMARY_CACHE}")
        with open(TOPIC_SUMMARY_CACHE, 'r', encoding='utf-8') as f:
            return f.read()
    logging.info("No topic summary cache found. Generating a new topic summary with the LLM...")
    unique_sources = sorted(list(set(doc['metadata']['source'] for doc in documents)))
    cleaned_topics = []
    for source in unique_sources:
        topic = re.sub(r'(_English|_202\d)?\.pdf', '', source, flags=re.IGNORECASE)
        topic = topic.replace('_', ' ').replace('ALL', '(ALL)').replace('AML', '(AML)')
        cleaned_topics.append(topic)
    topics_text = "\n".join(f"- {topic}" for topic in cleaned_topics)
    
    prompt = f"""
    You are a helpful assistant. Based on the following list of medical document titles, please generate a clean, user-friendly, and concise bulleted list of the main health topics covered.
    Group related topics together under a clear, bolded heading (e.g., **Luekemia**). Do not use more than 5-6 top-level categories.
    The final output should be formatted in Markdown. Do not include any introductory or concluding text, just the Markdown list.
    
    DOCUMENT TITLES:
    {topics_text}
    
    CONCISE TOPIC LIST:
    """
    try:
        llm = Ollama(model=model_name, request_timeout=300.0)
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
        os.makedirs(CACHE_DIR)
        
    logging.info("Starting RAG backend setup...")
    docs = load_and_process_pdfs(PDF_DIRECTORY)
    if docs:
        model = get_embedding_model()
        collection = setup_chromadb(docs, model, rebuild=True)
        logging.info("RAG backend setup complete.")
    else:
        logging.warning("No documents were loaded. The application might not function as expected.")