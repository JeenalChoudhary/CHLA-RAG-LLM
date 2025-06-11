__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
import backend_rag as main

# ---- Configurations ----

# EMBEDDING_MODEL_NAME = main.EMBEDDING_MODEL_NAME
# DB_PATH = main.DB_PATH
# PDF_DIRECTORY = main.PDF_DIRECTORY

if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'collection' not in st.session_state:
    st.session_state.collection = None

# ---- Page Setup ----
st.set_page_config(page_title="CHLA Health Education Chatbot", layout="wide")
st.title("CHLA Health Education RAG Chatbot")
st.markdown("Ask questions about family health education and get cited, document-grounded answers.")

# ---- Sidebar ----
st.sidebar.header("Database Configuration")
st.sidebar.info(
    "Click the button below to load the model and prepare the document database. "
    "This may take a few minutes on the first run."
)
if st.sidebar.button("Initialize/Refresh Database", key="init_db"):
    with st.spinner("Initializing... This may take a moment!"):
        # Load embedding model
        st.write("Loading embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.embedding_model = SentenceTransformer(main.EMBEDDING_MODEL_NAME, device=device)
        st.write("Embedding model loaded!")
        
        # Load & process PDFs as needed
        st.write("Processing PDFs...")
        documents = main.load_and_process_pdfs(main.PDF_DIRECTORY)
        st.write("PDFs processed!")
        
        # Setup ChromaDB collection
        if documents:
            st.write("Setting up vector database...")
            # Always rebuild when initializing from the UI
            st.session_state.collection = main.setup_chromadb(documents, st.session_state.embedding_model, rebuild=True)
            st.write("Database is ready.")
            st.session_state.db_ready = True
            st.sidebar.success("Database is ready!")
        else:
            st.error("No documents found. Please check the 'docs' folder in your repository.")
            st.session_state.db_ready = False

# Ensure database is initialized
# if 'collection' not in st.session_state:
#     with st.spinner("Initializing database..."):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         st.session_state.embedding_model = SentenceTransformer(main.EMBEDDING_MODEL_NAME, device=device)
#         documents = []
#         db_exists = os.path.exists(main.DB_PATH) and len(os.listdir(main.DB_PATH)) > 0
#         if not db_exists:
#             documents = main.load_and_process_pdfs(main.PDF_DIRECTORY)
#         st.session_state.collection = main.setup_chromadb(documents, st.session_state.embedding_model, rebuild=rebuild_db)
#     st.success("Database initialized!")

# ---- Query Interface ----
if st.session_state.db_ready:
    query = st.text_input("Enter your question about family health education:", key="query_input")
    if st.button("Get Answer", key="get_answer"):
        if not query:
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Searching for relevant documents..."):
                context, sources = main.retrieve_context(query, st.session_state.collection, st.session_state.embedding_model)
            
            with st.spinner("Contacting Ollama to generate answer..."):
                # Display
                answer = main.generate_answer(query, context, sources)
            st.subheader("Answer")
            st.write(answer)
            if sources:
                st.subheader("Sources")
                for src in sources:
                    st.markdown(f"- {src}")
else:
    st.info("Please initialize the database by using the button in the sidebar to begin.")
