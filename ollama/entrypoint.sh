#!/bin/sh

set -e

MODEL_NAME="gemma3:1b-it-qat"
SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/:/_/g')
SETUP_FLAG_FILE="/root/.ollama/models/setup_complete_${SANITIZED_MODEL_NAME}"

echo "Starting Ollama server in the background..."
ollama serve &
pid=$!

if [ ! -f "$SETUP_FLAG_FILE" ]; then
    echo "Setup flag not found. Model ${MODEL_NAME} may not be present."
    echo "Waiting for the Ollama server to be ready"

    sleep 5

    echo "Pulling model ${MODEL_NAME}. This may take up to 15 minutes."
    ollama pull "$MODEL_NAME"

    mkdir -p /root/.ollama/models

    touch "$SETUP_FLAG_FILE"
    echo "Model pull complete. Setup flag created."
else
    echo "Setup flag found. Model '${MODEL_NAME}' should be present. Skipping pull."
fi

echo "Initialization complete. Bringing Ollama server to the foreground..."
wait $pid