import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import backend_rag as main
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)

PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:5000").rstrip('/')
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'docs')

# Ensure the documents directory exists
if not os.path.exists(DOCUMENTS_DIR):
    logging.error(f"Documents directory not found: {DOCUMENTS_DIR}. Please ensure your PDF files are in this location.")
    # You might want to handle this more gracefully or exit if critical

# main.initialize_backend_components()

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
    if main._collection is None:
        logging.error("RAG backend's ChromaDB collection is not initialized. Cannot process chat.")
        return jsonify({"error": "RAG backend not fully initialized or database not found. Please ensure the database is built."}), 500
    
    logging.info(f"Received query: {user_query}")
    logging.info(f"Received history: {conversation_history}")
    
    def generate():
        response_generator = main.handle_query_stream(user_query, conversation_history)
        full_text_for_history = ""
        sources_for_history = []
        for chunk_type, content in response_generator:
            if chunk_type == 'text' and content:
                full_text_for_history += content
                yield f"data: {json.dumps({'text': content})}\n\n"
            elif chunk_type == 'sources':
                sources_with_links = []
                for filename in content:
                    download_url = f"{PUBLIC_API_URL}/download_source/{filename}"
                    sources_with_links.append({'filename': filename, 'url': download_url})
                sources_for_history = sources_with_links
                yield f"data: {json.dumps({'sources': sources_with_links})}\n\n"
            elif chunk_type == 'error':
                yield f"data: {json.dumps({'error': content})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/document_count', methods=['GET'])
def document_count():
    """Returns the current document count in the ChromaDB collection."""
    if main._collection:
        count = main._collection.count()
        return jsonify({"count": count})
    else:
        return jsonify({"count": 0, "error": "Collection not initialized"}), 503

@app.route('/download_source/<filename>', methods=['GET'])
def download_source(filename):
    """
    Endpoint to serve PDF documents for download.
    Ensures that only files within the DOCUMENTS_DIR can be accessed.
    """
    try:
        return send_from_directory(DOCUMENTS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"File not found for download: {filename} in {DOCUMENTS_DIR}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}")
        return jsonify({"error": "Could not serve file"}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index_path = os.path.join(app.static_folder, 'index.html')
    if not os.path.exists(index_path):
        return jsonify({"error":"Index.html not found in static assets."}), 404
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    main.initialize_backend_components()
    app.run(debug=True, port=5000)