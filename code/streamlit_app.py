# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import time
import backend_rag as main


def initialize_app():
    if 'db_ready' not in st.session_state:
        st.session_state.db_ready = False
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    
    if not st.session_state.db_ready:
        with st.spinner("Preparing the Health Education Navigator... This may take a few moments for initial startup!"):
            st.session_state.documents = main.load_and_process_pdfs(main.PDF_DIRECTORY)
            if st.session_state.documents:
                embedding_model = main.get_embedding_model()
                st.session_state.collection = main.setup_chromadb(st.session_state.documents, embedding_model, rebuild=True)
                st.session_state.db_ready = True
                st.session_state.messages.append({"role":"assistant", "content":"Hello! I am an assistant with Children's Hospital Los Angeles. I can help you understand various health topics. How can I assist you today? \n\nTo see what I know about, just ask me: **'What can you teach me?'**"})
            else:
                st.error(f"No PDF Documents found in the '{main.PDF_DIRECTORY}' folder. Please check your repository and add your PDF documents.")

initialize_app()
# ---- Page Setup ----
st.sidebar.header("Status")
if st.session_state.db_ready:
    st.sidebar.success("Database is ready.")
    st.sidebar.info(f"{len(st.session_state.documents)} document chunks loaded into database.")
else:
    st.sidebar.warning("Database is not initialized. Please check logs for more information.")

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def handle_user_query(prompt):
    st.session_state.messages.append({'role':'user', 'content':prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            response_generator, sources = main.handle_query(
                prompt,
                st.session_state.collection,
                main.get_embedding_model(),
                st.session_state.documents,
                model_name="gemma3:latest"
            )
            
            for chunk in response_generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.005)
                
        if sources:
            cleaned_sources = [s.replace('_', ' ').replace('.pdf', '') for s in sorted(list(set(sources)))]
            source_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in cleaned_sources)
            full_response += source_text
            
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({'role':'assistant', 'content':full_response})

if prompt := st.chat_input("Ask a question about a health topic..."):
    if main.SPECIAL_QUERY_TRIGGER in prompt.lower():
        handle_user_query(prompt)
    else:
        options = main.generate_query_expansion_options(prompt, model_name="gemma3:1b")
        if options:
            st.session_state.messages.append({'role':'user', 'content':prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            clarification_text = "That is a great question! To give you the most relevant information, could you please specify what you are looking for?"
            st.session_state.messages.append({"role": "assistant", "content": clarification_text})
            
            with st.chat_message("assistant"):
                st.markdown(clarification_text)
                for option in options:
                    if st.button(option, key=f"option_{option}"):
                        handle_user_query(option)
                        st.rerun()
        else:
            handle_user_query(prompt)