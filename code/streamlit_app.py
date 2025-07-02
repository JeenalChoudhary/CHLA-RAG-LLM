import streamlit as st
import os
import time
import backend_rag as main
import chromadb
import re

def initialize_app():
    if 'db_ready' not in st.session_state:
        st.session_state.db_ready = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'topic_summary' not in st.session_state:
        st.session_state.topic_summary = ""
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if "language" not in st.session_state:
        st.session_state.language = "en"

def initialize_backend():
    if not st.session_state.db_ready:
        with st.spinner("Preparing the Health Education Navigator... This may take a few moments for initial startup!"):
            if not os.path.exists(main.DB_PATH) or not os.listdir(main.DB_PATH):
                st.error(f"Database not found at '{main.DB_PATH}'")
                st.info("Please run the build script 'backend_rag.py' first to create the database.")
                st.code("python backend_rag.py --rebuild")
                st.stop()
            client = chromadb.PersistentClient(path=main.DB_PATH)
            try:
                st.session_state.collection = client.get_collection(main.COLLECTION_NAME)
                st.session_state.topic_summary = main.generate_topic_summary(st.session_state.collection, model_name="gemma3:1b")
                with open(main.TOPIC_SUMMARY_CACHE, 'w', encoding='utf-8') as f:
                    f.write(st.session_state.topic_summary)
                st.session_state.db_ready = True
                if not st.session_state.messages:
                    st.session_state.messages.append({"role":"assistant", "content":"Hello! I am an assistant with Children's Hospital Los Angeles. I can help you understand various health topics. How can I assist you today? \n\nTo see what I know about, just ask me: **'What can you teach me?'** \n\nTo exit this application, please type 'exit'."})
            except ValueError as e:
                st.error(f"Could not load the database collection: {e}")
                st.info("It seems the database is present but the collection is missing. Please run the build script to create it.")
                st.code("python backend_rag.py --rebuild")
                st.stop()

def draw_sidebar():
    with st.sidebar:
        st.header("Patient Health Education Navigator")
        st.markdown("-----")
        st.header("Status")
        st.success("Database is ready for querying!")
        st.info(f"{st.session_state.collection.count()} document chunks loaded.")
        st.markdown("-----")
        st.markdown("Built with ❤️ for CHLA's Patient Family Education Initiative")
        
def draw_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def draw_topic_buttons():
    if st.session_state.topic_summary and len(st.session_state.messages) <= 1:
        main_topics = re.findall(r'\*\*(.*?)\*\*', st.session_state.topic_summary, re.DOTALL)
        if main_topics:
            st.markdown("#### Suggested Health Education Topics")
            buttons_per_row = 6
            rows_of_topics = [main_topics[i:i+buttons_per_row] for i in range(0, len(main_topics), buttons_per_row)]
            for row in rows_of_topics:
                cols = st.columns(len(row))
                for i, topic in enumerate(row):
                    if cols[i].button(topic, key=f"topic_{topic}"):
                        st.session_state.user_input = topic
                        st.rerun()

def handle_user_query(prompt):
    if not prompt:
        return
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            response_generator, sources = main.handle_query(
                prompt, 
                st.session_state.collection, 
                model_name="gemma3:latest"
            )
            if response_generator:
                for chunk in response_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.0075)
            if sources:
                cleaned_sources = [s.replace("_", " ").replace(".pdf", "") for s in sorted(list(set(sources)))]
                source_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in cleaned_sources)
                full_response += source_text
            message_placeholder.markdown(full_response)
    if full_response:
        st.session_state.messages.append({"role":"assistant", "content":full_response})
        
if __name__ == "__main__":
    initialize_app()
    draw_sidebar()
    initialize_backend()
    draw_chat_history()
    draw_topic_buttons()
    
    if prompt := st.chat_input("Ask a question about a health topic!") or st.session_state.user_input:
        if prompt.lower() == "exit":
            st.stop()
        st.session_state.user_input = ""
        handle_user_query(prompt)