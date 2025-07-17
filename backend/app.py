# app.py
import sys
import os

# Explicitly add the current directory (project root) to Python's path
# This ensures that 'code' package can be found when app.py is run from the root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from backend_code import backend_rag as main
import logging
import chromadb
import json # Ensure json is imported for dumps

# Configure logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# Enable CORS for all routes. In a production environment, you might want to restrict
# this to specific origins (e.g., CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}))
CORS(app)

# Initialize backend components once when the Flask app starts
# This will load the models and connect to ChromaDB.
# We call this here so it's ready when the Flask app serves requests.
main.initialize_backend_components()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks."""
    # You can add more detailed health checks here, e.g., if ChromaDB is reachable
    # and if models are loaded. For now, a simple status is fine.
    return jsonify({"status": "healthy", "message": "Backend is running."}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    conversation_history = data.get('history', [])

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Ensure the backend components (including ChromaDB collection) are initialized
    # backend_rag.py's initialize_backend_components will log if it fails to load the collection.
    # We can check if main._collection is None to indicate a problem.
    if main._collection is None:
        logging.error("RAG backend's ChromaDB collection is not initialized. Cannot process chat.")
        return jsonify({"error": "RAG backend not fully initialized or database not found. Please ensure the database is built."}), 500

    logging.info(f"Received query: {user_query}")

    try:
        def generate():
            # Call the streaming function from backend_rag.py
            # It now uses the internally managed _collection
            response_generator = main.handle_query_stream(
                user_query,
                conversation_history
            )
            for chunk_type, content in response_generator:
                if chunk_type == 'text':
                    # Send text chunks as plain data
                    yield f"data: {content}\n\n"
                elif chunk_type == 'sources':
                    # Send sources as a JSON object
                    yield f"data: {json.dumps({'sources': content})}\n\n"
                elif chunk_type == 'error':
                    # Send errors as a JSON object
                    yield f"data: {json.dumps({'error': content})}\n\n"

        # Return a streaming response with Server-Sent Events (SSE) mimetype
        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        logging.error(f"Error processing chat query: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Run on port 5000
