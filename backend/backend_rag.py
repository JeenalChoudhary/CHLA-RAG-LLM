import os
import re
import json
import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import logging
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

COLLECTION_NAME = "example_health_docs"
EMBEDDING_MODEL_NAMES = "BAAI/bge-m3"
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L4-v2'
LLM_MODEL_NAME = "granite3-dense:2b" #qwen3:4b and qwen3:4b-thinking and gemma3:4b-it-q4_K_M and gemma3:4b and phi3:3.8b
SENTENCES_PER_CHUNK = 6
STRIDE = 2
INITIAL_RETRIEVAL_COUNT = 20
FINAL_CONTEXT_COUNT = 10
RERANKER_SCORE_THRESHOLD = 0.75

_embedding_models = None
_llm_models = None
_reranker_model = None
_chroma_client = None
_collection = None

# ---- Setup NLTK ----
try:
    nltk.data.find('tokenizer/punkt')
except LookupError:
    logging.info("Downloading NLTK 'punkt' and 'punkt_tab'tokenizer...")
    nltk.download(['punkt', 'punkt_tab'])
    logging.info("NLTK 'punkt' and 'punkt_tab' downloaded.")

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
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        logging.info(f"Connecting to Ollama at: {ollama_host}")
        _llm_models = Ollama(model=LLM_MODEL_NAME, base_url=ollama_host, request_timeout=600.0)
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
    text = re.sub(r'(?im)^\s*(Division of .*|Otolaryngology|Approved by PFE .*)\s*$', '', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) #Removes words broken across lines like resuscita-tion to resuscitation
    text = re.sub(r'^(.*?)\s*\.{3,}\s*\d+\s*$', '', text, flags=re.MULTILINE) #Removes TOC style dot leaders
    text = re.sub(r'•\s*([a-zA-Z])', r'\n\n• \1', text) #Add space and break before bullets
    text = re.sub(r'([a-zA-Z0-9]\.)([a-zA-Z])', r'\1 \2', text) #Adds space before numbered lists
    text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text) #separates concatenated sentences from headings
    text = re.sub(r'[ \t]+', ' ', text) #Replaces multiple spaces with one
    text = re.sub(r'\n\s*\n', '\n\n', text) #Replaces multi-line breaks with standard double break
    text = '\n'.join(line for line in text.splitlines() if line.strip())
    return text.strip()

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

def generate_hypothetical_document(query: str) -> list:
    prompt = f"""
    You are a helpful AI assistant specializing in generating hypothetical user questions for a medical information retrieval system. Your goal is to anticipate how different users might ask about the same topic.
    Your task is to take the original 'User Question' and generate 4 alternative, complete questions that a real user might ask to find the same information.
    
    **--- CRITICAL RULES ---**
    1. **Format as Full Questions** Each alternate MUST be a complete, grammatically correct question. It should sound like a natural question a person would type (e.g., start with "What", "How", "Is", and "When").
    2. **Preserve Core Concepts:** You MUST identify and preserve all critical medical or technical terms from the original question (e.g., "CPR", "choked", "DNR", "tracheostomy"). Do not replace them with vague synonyms.
    3. **Maintain User Intent:** Each new question must seek the same core answer as the original. Do not change the topic or introduce new concepts.
    4. **Focus on Actionable Information:** Generate questions that seek concrete steps, definitions, or procedures. Avoid abstract or philosophical questions.
    
    **--- EXAMPLES ---**
    
    **User Question:** "What if I see a 'DNR' bracelet on them?"
    **Output:**
    [
        "How should I respond to a DNR order in an emergency?",
        "Am I allowed to perform CPR on someone with a DNR bracelet?",
        "What does a 'Do Not Resuscitate' bracelet mean for providing first aid?",
        "What are the legal requirements when a patient has a DNR?"
    ]
    
    **User Question:** "What if I think they choked on something?"
    **Output:**
    [
        "How do I perform first aid on a choking child?",
        "What are the steps to help someone who is choking?",
        "What should I do if a person's airway is obstructed by an object?",
        "Is the response for choking different from performing CPR?"
    ]
    
    **--- YOUR TASK ---**
    
    **User Question:** "{query}"
    **Output (provide a single, valid JSON list of 4 strings):**
    """
    logging.info(f"Generating hypothetical questions for query: '{query}'")
    response = _llm_models.complete(prompt)
    response_text = response.text.strip()
    try:
        match = re.search(r'\[.*\]', response_text, flags=re.DOTALL)
        if not match:
            logging.warning(f"Could not find a JSON list in the LLM response for query expansion. Raw response: {response_text}")
            return []
        json_content = match.group(0)
        sub_questions = json.loads(json_content)
        if isinstance(sub_questions, list) and len(sub_questions) > 0:
            logging.info(f"Successfully generated keyword-preservedsub-questions: {sub_questions}")
            return sub_questions
        else:
            logging.warning("Regex found no questions in the JSON structure. Falling back to original query.")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from LLM response: {e}. Raw text: '{response_text}'")
    except Exception as e:
        logging.error(f"Failed to parse the questions with Regex: {e}. Raw text: '{response_text}'")
        return []
    
def rewrite_query_with_history(query: str, conversation_history: str) -> str:
    if not conversation_history.strip():
        logging.info(f"No conversation history detected. Using the original query: '{query}'")
        return query
    prompt = f"""
    You are an expert AI assistant specializing in analyzing conversational context and rewriting queries. Your task is to make user questions self-contained, but ONLY if they are a direct follow-up to the existing chat history.
    
    You must perform two steps:
    1. **Assess Relevance:** First, determine if the 'New Question' is a direct follow-up to the 'Chat History', or if it introduces a new, unrelated topic.
    2. **Act Accordingly:**
        * **If the question IS a follow-up**, rewrite it to be a standalone query, incorporating necessary context from the history. Your output MUST be in the format "(Topic: [Identified Topic]) [Rewritten Question]".
        * **If the question IS A NEW TOPIC**, you MUST ignore the history and output the 'New Question' exactly as it is, without any changes or added topic.
    
    --- EXAMPLES OF FOLLOW-UP QUESTIONS (REWRITING IS NEEDED) ---

    ### Example 1:
    Chat History:
    User: How is CPR for kids different from adult CPR?
    AI: The main difference is in the chest compressions...
    New Question:
    At what age do I switch to the adult method?
    Rewritten Question:
    (Topic: CPR for Children) At what age should you switch from the child CPR method to the adult CPR method?

    ### Example 2:
    Chat History:
    User: Tell me about cleaning a trach tube.
    AI: You should clean it twice a day...
    New Question:
    What supplies do I need?
    Rewritten Question:
    (Topic: Tracheostomy Tube Care) What supplies are needed for cleaning a tracheostomy tube?

    --- EXAMPLES OF NEW TOPICS (REWRITING IS NOT NEEDED) ---

    ### Example 3:
    Chat History:
    User: Tell me about cleaning a trach tube.
    AI: You should clean it twice a day...
    New Question:
    What are the symptoms of asthma?
    Rewritten Question:
    What are the symptoms of asthma?

    ### Example 4:
    Chat History:
    User: How is CPR for kids different from adult CPR?
    AI: The main difference is in the chest compressions...
    New Question:
    Tell me about the causes of obesity.
    Rewritten Question:
    Tell me about the causes of obesity.
    
    ### CURRENT TASK
    Chat History:
    {conversation_history}
    New Question:
    {query}
    Rewritten Question:
    """
    try:
        logging.info("Rewriting query based on conversation history...")
        response = _llm_models.complete(prompt)
        rewritten_query = response.text.strip()
        logging.info(f"Rewritten query: '{rewritten_query}'")
        if rewritten_query.lower() != query.lower():
            logging.info(f"Successfully rewrote user query: '{rewritten_query}'")
        else:
            logging.warning(f"Query was already self-contained and identified as a new topic. No rewrite needed.")
        return rewritten_query
    except Exception as e:
        logging.error(f"Failed to rewrite query: {e}. Using original query.")
        return query

def sanitize_rewritten_query(query: str) -> str:
    smart_quotes_map = {
        u'\u201c': '"',
        u'\u201d': '"',
        u'\u2018': "'",
        u'\u2019': "'"
    }
    pattern = re.compile("|".join(smart_quotes_map.keys()))
    sanitize_text = pattern.sub(lambda m: smart_quotes_map[m.group(0)], query)
    return sanitize_text.strip()

def normalize_scores(logits: list[float]) -> list[float]:
    tensor_logits = torch.tensor(logits)
    probabilities = torch.sigmoid(tensor_logits)
    return probabilities.tolist()

def retrieve_context(query: str, conversation_history: str = "", n_results: int = FINAL_CONTEXT_COUNT):
    logging.info(f"Retrieving context for query: '{query}'")
    raw_standalone_query = rewrite_query_with_history(query, conversation_history)
    standalone_query = sanitize_rewritten_query(raw_standalone_query)
    all_queries = [standalone_query] + generate_hypothetical_document(standalone_query)
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
        return [], [], standalone_query
    retrieved_docs = list(retrieved_doc_set.keys())
    retrieved_metadata = list(retrieved_doc_set.values())
    logging.info(f"Retrieved {len(retrieved_docs)} documents for reranking.")
    logging.info(f"Reranking documents against the query: '{standalone_query}'")
    rerank_pairs = [[query, doc] for doc in retrieved_docs]
    logit_scores = _reranker_model.predict(rerank_pairs, show_progress_bar=False)
    normalized_scores = normalize_scores(logit_scores)
    sorted_docs = sorted(zip(normalized_scores, retrieved_docs, retrieved_metadata), key=lambda x: x[0], reverse=True)
    top_5_scores = [f"{score:.4f}" for score, _, in sorted_docs[:5]]
    logging.info(f"Top 5 reranker scores: {top_5_scores}")
    top_reranked_docs = [doc for doc in sorted_docs if doc[0] >= RERANKER_SCORE_THRESHOLD][:n_results]
    if not top_reranked_docs:
        logging.warning(f"No documents met the relevance threshold of {RERANKER_SCORE_THRESHOLD}. Top score: {sorted_docs[0][0]} from the document '{sorted_docs[0][1]}'. Aborting generation and falling back.")
        top_fallback_docs = sorted_docs[:5]
        fallback_sources = sorted(list(set(meta['source'] for socre, doc, meta in top_fallback_docs)))
        return [], fallback_sources, standalone_query
    final_docs = [doc for score, doc, meta in top_reranked_docs]
    final_metadata = [meta for score, doc, meta in top_reranked_docs]
    sources = sorted(list(set(meta['source'] for meta in final_metadata)))
    logging.info(f"Reranked and selected top {len(final_docs)} documents.")
    logging.info(f"Retrieved final context from sources: {sources}")
    return final_docs, sources, standalone_query
   
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
    **Persona:** You are an expert medical educator at Children's Hospital Los Angeles. Your mission is to provide helpful, reassuring, and empathetic guidance to worried parents and patients. Your language MUST be simple, clear, and easy to understand.
   
    **--- HEIRARCHY OF RULES (MOST IMPORTANT FIRST) ---**
    
    ## 1. THE PRIME DIRECTIVE: DO NO HARM
    Your primary responsibility is user safety. If you are ever faced with ambiguity or conflicting information, you MUST choose the safest, most conservative interpretation. Never provide information that could be misinterpreted as a dangerous instruction.
    
    ## 2. THE CONTEXT IS YOUR ONLY UNIVERSE
    Your entire response MUST be generated using ONLY the information from the `CONTEXT` section below. Do not use any outside knowledge. If the context does not contain the answer, you MUST state that you cannot find the information.
    
    ## 3. ANTI-SYNTHESIS SAFETY PROTOCOL (CRITICAL)
    This is your most important logical rule. You MUST NOT merge facts from different documents to create a *new* instruction or conclusion that is not explicitly stated within a *single* source document.
    - **Correct Example:** If one doc says "Start CPR if there is no pulse" and another doc defines "Brain Death", you must NOT combine them. You would answer the CPR question and ignore the brain death information as it is out of context.
    - **Incorrect (Forbidden) Example:** Combining the two facts above to claim that starting CPR is related to brain death.
    - **Action:** If documents seem to conflict or discuss different topics, address only the parts directly relevant to the user's question and ignore the rest.
    
    ## 4. GRACEFUL IGNORANCE PROTOCOL
    If the provided `CONTEXT` is weak, irrelevant, or does not directly and confidently answer the user's specific `Question`, you MUST refuse to answer.
    - **Trigger this rule if:** The context talks about a general topic (e.g., "transfers") but the question is highly specific and not addressed (e.g., "Should I move someone having a seizure?").
    - **Your response in this case MUST be:** "I was able to find some general information on [Topic], but I could not find a specific answer to your question about [User's Specific Question] in my knowledge base."
    
    ## 5. STANDARD OPERATING RULES
    - **Synthesize, Don't List:** Combine relevant facts from a *single, coherent source* or *multiple sources that explicitly agree* into a cohesive answer. Use lists and bolding for clarity.
    - **Guide on Board Queries:** For broad queries, provide a brief overview and suggest 2-3 specific follow-up questions to guide the user.
    - **No External Info:** Never provide URLs, suggest external websites, or include disclaimers about medical advice.

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
        logging.error("Cannot handle query because the ChromaDB collection has not been initialized.")
        yield "error", "The knowledge base is not ready. Please contact CHLA Support."
        yield "sources", []
        return
    logging.info(f"Handling streamed query: '{query}'")
    cleaned_history = []
    for msg in chat_history:
        role = "User" if msg.get('isUser') else "Assistant"
        content = msg.get('content', '')
        if role == "Assistant":
            content = content.split('**Sources:**')[0].strip()
        if content:
            formatted_role = 'User' if role == 'User' else 'Assistant'
            cleaned_history.append(f"{formatted_role}: {content}")
    history_str = "\n".join(cleaned_history)
    try:
        context_docs, sources, final_query = retrieve_context(query, history_str)
        if not context_docs:
            if sources:
                logging.warning(f"No direct context found or context deemed irrelevant for '{query}'. Providing fallback resources: {sources}")
                fallback_message = "I couldn't find a direct answer to your question in my knowledge base. However, the query returned the following documents which may contain related information. You can review them to see if they are helpful in any way:"
                yield {"text": fallback_message}
                yield {"sources": sources}
            else:
                logging.warning(f"No relevant documents found for query: '{query}'. Aborting generation.")
                yield {"text": "I couldn't find any information related to your query in my knowledge base. Please try rephrasing your question."}
                yield {"sources": []}
            return
        deduplicated_docs = deduplicate_context(context_docs)
        answer_generator = generate_answer_stream(query, deduplicated_docs, history_str)
        full_response_text = "".join([token for token in answer_generator])
        cleaned_response = parse_and_clean_output(full_response_text)
        if cleaned_response:
            yield {"text": cleaned_response}
        yield {"sources": sources}
    except Exception as e:
        logging.error(f"An error occurred while handling the query: {e}", exc_info=True)
        yield {"error": "An internal error occurred while generating the response."}
        yield {"sources": []}

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
            logging.info(f"Rebuild process using device: {device}")
            embedding_model_build = SentenceTransformer(EMBEDDING_MODEL_NAMES, device=device)
            client_build = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, is_persistent=True, anonymized_telemetry=False))
            collection_build = client_build.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
            batch_size = 256
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