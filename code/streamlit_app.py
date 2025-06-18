# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import backend_rag as main

# ---- Configurations ----
if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'collection' not in st.session_state:
    st.session_state.collection = None

# ---- Page Setup ----
st.set_page_config(page_title="CHLA Health Education Chatbot", layout="wide")
st.title("CHLA's Family Health Education Chatbot")

# ---- Caching Functions ----
@st.cache_resource
def get_embedding_model():
    return main.get_embedding_model()

@st.cache_data
def load_docs(directory):
    return main.load_and_process_pdfs(directory)

@st.cache_resource
def setup_db(_docs, _embedding_model):
    return main.setup_chromadb(_docs, _embedding_model, rebuild=True)

# ---- Sidebar ----
st.sidebar.header("Database Configuration")
if st.sidebar.button("Initialize/Refresh Database", key="init_db"):
    with st.spinner("Initializing... This may take a moment!"):
        docs = load_docs(main.PDF_DIRECTORY)
        if docs:
            embedding_model = get_embedding_model()
            st.session_state.collection = setup_db(docs, embedding_model)
            st.session_state.db_ready = True
            st.success("Database is ready for generation")
        else:
            st.error(f"No documents found. Please check the {main.PDF_DIRECTORY} folder in your repository.")

st.sidebar.info("Click the button above to load your PDF documents. This is required before querying the chatbot.")

# ---- Query Interface ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])
    
if prompt := st.chat_input("Ask a question relating to your health documents!"):
    if "collection" not in st.session_state or not st.session_state.db_ready:
        st.warning("Please initialize the database using the button on the sidebar before querying.")
        st.stop()
    
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            embedding_model = get_embedding_model()
            collection = st.session_state.collection
            context, sources = main.retrieve_context(prompt, collection, embedding_model)
            response_generator = main.generate_answer(prompt, context, "gemma3:latest")
            
            for chunk in response_generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        if sources:
            source_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sorted(list(set(sources))))
            full_response += source_text
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role":"assistant", "content":full_response})
