#!/bin/sh

set -e

echo "Starting ollama server in the background..."
ollama serve & pid=$!

echo "Waiting for ollama server to be ready..."
while ! ollama list > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama server is ready."

SANITIZIED_MODEL_NAME=$(echo "$OLLAMA_MODEL" | sed 's/:/_/g' | sed 's/\//_/g')
FLAG_FILE="/root/.ollama/models/setup_complete_${SANITIZIED_MODEL_NAME}"

if [ ! -f "$FLAG_FILE" ]; then
    echo "Setup flag not found for model '$OLLAMA_MODEL'. Pulling model..."
    ollama pull "$OLLAMA_MODEL"
    echo "Model pull complete. Creating setup flag..."
    touch "$FLAG_FILE"
else
    echo "Setup flag found for model '$OLLAMA_MODEL'. Skipping model pull."
fi

echo "Bringing ollama server to the foreground..."
wait $pid