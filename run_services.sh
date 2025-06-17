#!/bin/bash

# Script to run all services in parallel for the RAG application

# Function to handle cleanup on script exit
cleanup() {
    echo "Stopping all services..."
    # Kill all background processes started by this script
    jobs -p | xargs -r kill
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Activate virtual environment
source venv/bin/activate

echo "Starting all services..."

# From the .env file, load llm_model and embedding_model
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
# Check if required environment variables are set
if [ -z "$llm_model" ] || [ -z "$embedding_model" ]; then
    echo "Error: llm_model and embedding_model must be set in .env file."
    exit 1
fi

# Start LLM server (gemma-3-4b-it)
echo "Starting LLM server on port 8080..."
llama-server -hf $llm_model -c 4096 &
LLM_PID=$!

# Start embedding server (Qwen3-Embedding)
echo "Starting embedding server on port 8081..."
llama-server -hf $embedding_model --embeddings --port 8081 --pooling cls &
EMBED_PID=$!

# Wait a bit for servers to start up
echo "Waiting for servers to initialize..."
sleep 10

# Start Chainlit application
echo "Starting Chainlit application..."
chainlit run main.py &
CHAINLIT_PID=$!

echo "All services started!"
echo "LLM Server PID: $LLM_PID"
echo "Embedding Server PID: $EMBED_PID"
echo "Chainlit PID: $CHAINLIT_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
