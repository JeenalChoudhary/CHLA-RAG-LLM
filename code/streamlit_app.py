import streamlit as st
import os
import time
import backend_rag as main
import chromadb

def initialize_app():
    if 'db_ready' not in st.session_state:
        st.session_state.db_ready = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

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
                st.session_state.db_ready = True
                if not st.session_state.messages:
                    st.session_state.messages.append({"role":"assistant", "content":"Hello! I am an assistant with Children's Hospital Los Angeles. I can help you understand various health topics. How can I assist you today? \n\nTo exit this application, please type 'exit'."})
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
        st.header("Conversation Limit")
        st.info("The chat will reset after 10 messages to ensure relevance and accuracy.")
        st.metric(label="Message Progress", value=f"{st.session_state.message_count} / 10")
        st.markdown("-----")
        st.markdown("Built with ❤️ for CHLA's Patient Family Education Initiative")
        
def draw_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_query(prompt):
    if not prompt:
        return
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.session_state.message_count += 1
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            history_for_context = st.session_state.messages[-7:-1]
            conversation_history = "\n".join([f"{msg["role"]}: {msg['content']}" for msg in history_for_context])
            response_generator, sources = main.handle_query(
                prompt, 
                st.session_state.collection, 
                model_name="gemma3:1b-it-qat",
                conversation_history=conversation_history
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
        st.session_state.message_count += 1
        
if __name__ == "__main__":
    initialize_app()
    initialize_backend()
    draw_sidebar()
    draw_chat_history()
    
    if prompt := st.chat_input("Ask a question about a health topic!") or st.session_state.user_input:
        if prompt.lower() == "exit":
            st.stop()
        st.session_state.user_input = ""
        handle_user_query(prompt)
        if st.session_state.message_count >= 10:
            st.success("Conversation limit has been reached. Starting a fresh chat for you! ✨")
            st.balloons()
            time.sleep(3)
            st.session_state.messages = [{"role":"assistant", "content":"Hello! I am an assistant with Children's Hospital Los Angeles. I can help you understand various health topics. How can I assist you today? \n\nTo exit this application, please type 'exit'."}]
            st.session_state.message_count = 0
            st.rerun()
        else:
            st.rerun()