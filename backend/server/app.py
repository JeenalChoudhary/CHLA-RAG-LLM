# app.py
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import backend_rag as main # Import your existing RAG backend
import logging
import os
import chromadb

# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, adjust as needed for production

# Initialize backend components once when the Flask app starts
# This will load the models and connect to ChromaDB
main.initialize_backend_components()

# Global variables to store the ChromaDB collection
# We'll use these to avoid re-initializing on every request
global_collection = None

@app.before_request
def check_db_ready():
    global global_collection
    if not global_collection:
        # This check is similar to what was in streamlit_app.py
        # Ensure the DB exists and is accessible
        if not os.path.exists(main.DB_PATH) or not os.listdir(main.DB_PATH):
            logging.error(f"Database not found at '{main.DB_PATH}'. Please run 'python backend_rag.py --rebuild'.")
            # In a real API, you might return an error status here or stop the app.
            # For now, we'll just log and let the request proceed, though it will fail if DB operations are attempted.
        try:
            client = chromadb.PersistentClient(path=main.DB_PATH)
            global_collection = client.get_collection(main.COLLECTION_NAME)
            logging.info("ChromaDB collection loaded successfully for API.")
        except Exception as e:
            logging.error(f"Error loading ChromaDB collection: {e}")
            global_collection = None # Ensure it's None if loading fails

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks."""
    return jsonify({"status": "healthy", "message": "Backend is running."}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    conversation_history = data.get('history', [])

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    if not global_collection:
        return jsonify({"error": "RAG backend not initialized or database not found. Please ensure the database is built."}), 500

    logging.info(f"Received query: {user_query}")

    # Pass the actual collection object to handle_query
    try:
        def generate():
            response_generator = main.handle_query_stream(
                user_query,
                conversation_history,
                collection=global_collection
            )
            for chunk_type, content in response_generator:
                if chunk_type == 'text':
                    yield f"data: {content}\n\n"
                elif chunk_type == 'sources':
                    # Send sources as a final JSON object after all text
                    yield f"data: {{\"sources\": {json.dumps(content)}}}\n\n"
                elif chunk_type == 'error':
                    yield f"data: {{\"error\": \"{content}\"}}\n\n"


        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        logging.error(f"Error processing chat query: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    # You might want to remove --rebuild from the startup in `backend_rag.py`
    # if you want to explicitly control when the DB is built.
    # The `check_db_ready` function above will log if it's missing.
    app.run(debug=True, port=5000) # Run on port 5000