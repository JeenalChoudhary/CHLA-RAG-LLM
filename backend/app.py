import sys
import os
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import backend_rag as main
import logging
import json

# Add the project root to the sys.path to ensure modules can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define static folder and initialize Flask app
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app) # Enable CORS for all routes

# Get public API URL from environment variable, default to localhost
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:5000").rstrip('/')

# Define the directory where PDF documents are stored
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'docs')

# Ensure the documents directory exists
if not os.path.exists(DOCUMENTS_DIR):
    logging.error(f"Documents directory not found: {DOCUMENTS_DIR}. Please ensure your PDF files are in this location.")
    # In a production environment, you might want to exit or handle this more robustly.

# --- Flask Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks."""
    return jsonify({"status": "healthy", "message": "Backend is running."}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests by querying the RAG backend and streaming responses.
    Expects a JSON payload with 'query' (user's message) and 'history' (conversation history).
    """
    data = request.json
    user_query = data.get('query')
    conversation_history = data.get('history', [])

    if not user_query:
        return jsonify({"error": "Query not provided"}), 400

    logging.info(f"Received query: {user_query}")

    def generate():
        """Generator function to stream RAG responses."""
        try:
            full_response_content = ""
            sources_info = [] # To accumulate sources

            # Iterate through chunks received from the RAG backend
            for chunk in main.rag_query(user_query, conversation_history):
                if "text" in chunk:
                    text_content = chunk["text"]
                    full_response_content += text_content
                    # Send text chunk as JSON data
                    yield f"data: {json.dumps({'text': text_content})}\n\n"
                
                if "sources" in chunk:
                    # Update sources as they come in (assuming cumulative updates or final set)
                    sources_info = chunk["sources"]
                    formatted_sources = []
                    for source in sources_info:
                        filename = os.path.basename(source)
                        # Construct the URL for serving the PDF, using PUBLIC_API_URL
                        # This URL will be used by the frontend to link to the document
                        source_url = f"{PUBLIC_API_URL}/download_source/{filename}"
                        formatted_sources.append({"filename": filename, "url": source_url})
                    # Send updated sources as JSON data
                    yield f"data: {json.dumps({'sources': formatted_sources})}\n\n"
                
                if "error" in chunk:
                    # If an error occurs in the backend, stream it and break
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break # Stop streaming on error

        except Exception as e:
            # Catch any unexpected errors during streaming and send an error message
            logging.error(f"Error during chat streaming: {e}")
            yield f"data: {json.dumps({'error': f'An internal server error occurred: {str(e)}'})}\n\n"

    # Return a streaming response with server-sent events (SSE) mimetype
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/document_count', methods=['GET'])
def document_count():
    """Returns the current document count in the ChromaDB collection."""
    if main._collection:
        count = main._collection.count()
        return jsonify({"count": count})
    else:
        # Return an error if the collection hasn't been initialized yet
        return jsonify({"count": 0, "error": "Collection not initialized"}), 503

@app.route('/download_source/<filename>', methods=['GET'])
def download_source(filename):
    """
    Endpoint to serve PDF documents.
    Ensures that only files within the DOCUMENTS_DIR can be accessed.
    The 'as_attachment=False' (default behavior) makes the browser try to display the file
    instead of forcing a download.
    """
    try:
        # Serve the file from the DOCUMENTS_DIR.
        # By default, as_attachment is False, so it will try to display in browser.
        return send_from_directory(DOCUMENTS_DIR, filename)
    except FileNotFoundError:
        logging.error(f"File not found for download: {filename} in {DOCUMENTS_DIR}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}")
        return jsonify({"error": "Could not serve file"}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    """
    Serves the Single Page Application (SPA) frontend.
    If a specific path exists in the static folder, it serves that file.
    Otherwise, it serves index.html for client-side routing.
    """
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index_path = os.path.join(app.static_folder, 'index.html')
    if not os.path.exists(index_path):
        return jsonify({"error": "index.html not found in static folder"}), 404
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # This block is for local development only.
    # In a production deployment (e.g., with Gunicorn), this block will not be executed.
    logging.info("Starting Flask app in development mode.")
    # Uncomment the line below if you need to initialize backend components when running app.py directly
    # main.initialize_backend_components()
    app.run(host='0.0.0.0', port=5000, debug=True)

