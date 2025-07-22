import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import backend_rag as main
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

main.initialize_backend_components()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks."""
    return jsonify({"status": "healthy", "message": "Backend is running."}), 200

@app.route('/chat', methods=['GET'])
def chat():
    user_query = request.args.get('query')
    history_json = request.args.get('history', [])
    try:
        conversation_history = json.loads(history_json)
    except json.JSONDecodeError:
        logging.error("Failed to decode conversation history JSON.")
        return jsonify({"error": "Invalid history format"}), 400
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    if main._collection is None:
        logging.error("RAG backend's ChromaDB collection is not initialized. Cannot process chat.")
        return jsonify({"error": "RAG backend not fully initialized or database not found. Please ensure the database is built."}), 500
    logging.info(f"Received query: {user_query}")
    logging.info(f"Received history: {conversation_history}")
    def generate():
        response_generator = main.handle_query_stream(user_query, conversation_history)
        for chunk_type, content in response_generator:
            if chunk_type == 'text' and content:
                yield f"data: {json.dumps({'text': content})}\n\n"
            elif chunk_type == 'sources':
                yield f"data: {json.dumps({'sources': content})}\n\n"
            elif chunk_type == 'error':
                yield f"data: {json.dumps({'error': content})}\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/document_count', methods=['GET'])
def document_count():
    if main._collection:
        count = main._collection.count()
        return jsonify({"count": count})
    else:
        return jsonify({"count": 0, "error": "Collection not initialized"}), 503

if __name__ == '__main__':
    app.run(debug=True, port=5000)