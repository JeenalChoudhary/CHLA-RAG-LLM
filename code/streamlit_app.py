import streamlit as st
import time
import backend_rag as main
import re

def initialize_app():
    if 'db_ready' not in st.session_state:
        st.session_state.db_ready = False
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'topic_summary' not in st.session_state:
        st.session_state.topic_summary = ""
    if 'query_options' not in st.session_state:
        st.session_state.query_options = []
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

def initialize_backend():
    if not st.session_state.db_ready:
        with st.spinner("Preparing the Health Education Navigator... This may take a few moments for initial startup!"):
            st.session_state.documents = main.load_and_process_pdfs(main.PDF_DIRECTORY)
            if st.session_state.documents:
                st.session_state.embedding_model = main.get_embedding_model()
                st.session_state.collection = main.setup_chromadb(st.session_state.documents, st.session_state.embedding_model, rebuild=False)
                st.session_state.topic_summary = main.generate_topic_summary(st.session_state.documents, model_name="gemma3:1b")
                st.session_state.db_ready = True
                if not st.session_state.messages:
                    st.session_state.messages.append({"role":"assistant", "content":"Hello! I am an assistant with Children's Hospital Los Angeles. I can help you understand various health topics. How can I assist you today? \n\nTo see what I know about, just ask me: **'What can you teach me?'** \n\nTo exit this application, please type 'exit'."})
            else:
                st.error(f"No PDF Documents found in the '{main.PDF_DIRECTORY}' folder. Please check your repository and add your PDF documents.")
                st.stop()

def draw_sidebar():
    with st.sidebar:
        st.header("Patient Health Education Navigator")
        st.markdown("-----")
        st.header("Status")
        if st.session_state.db_ready:
            st.success("Database is ready for querying!")
            st.info(f"{len(st.session_state.documents)} document chunks loaded.")
        else:
            st.warning("Database is initializing...")
        st.markdown("-----")
        st.header("Filters")
        st.info("These filters are for demonstration and are not yet functional.")
        st.selectbox("Language", ["English", "Spanish"], key="language_filter")
        st.selectbox("Audience", ["Pediatric", "Adult"], key="audience_filter")
        st.selectbox("Category", ["Asthma", "Diabetes"], key="category_filter")
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
                cols = st.columns(row)
                for i, topic in enumerate(row):
                    if cols[i].button(topic, key=f"topic_{topic}"):
                        st.session_state.user_input = topic
                        st.rerun()

def draw_clarification_buttons():
    if st.session_state.query_options:
        clarification_text = "That is a great question! To give you the most relevant information, could you please specify what you're looking for?"
        if not st.session_state.messages or st.session_state.messages[-1]['content'] != clarification_text:
            st.session_state.messages.append({"role":"assistant", "content":clarification_text})
        st.experimental_rerun()
    if st.session_state.messages and "please specify" in st.session_state.messages[-1].get("content", ""):
        with st.chat_message("assistant"):
            st.markdown(st.session_state.messages[-1]["content"])
            for option in st.session_state.query_options:
                if st.button(option, key=f"option_{option}"):
                    st.session_state.user_input = option
                    st.session_state.query_options = []
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
            options = main.generate_query_expansion_options(prompt, model_name="gemma3:1b")
            if options:
                st.session_state.query_options = options
                st.rerun()
            response_generator, sources = main.handle_query(
                prompt, 
                st.session_state.collection, 
                st.session_state.embedding_model, 
                st.session_state.documents, 
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
    draw_clarification_buttons()
    
    if prompt := st.chat_input("Ask a question about a health topic!") or st.session_state.user_input:
        if prompt.lower() == "exit":
            st.stop()
        st.session_state.user_input = ""
        handle_user_query(prompt)
        