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

# Start LLM server (gemma-3-4b-it)
echo "Starting LLM server on port 8080..."
llama-server -hf ggml-org/gemma-3-4b-it-GGUF -c 4096 &
LLM_PID=$!

# Note: Embedding server is no longer needed as we use SentenceTransformers locally

# Wait a bit for server to start up
echo "Waiting for LLM server to initialize..."
sleep 10

# Start Chainlit application
echo "Starting Chainlit application..."
#chainlit run main.py &
#CHAINLIT_PID=$!

echo "All services started!"
echo "LLM Server PID: $LLM_PID"
echo "Note: Embeddings are now handled locally with SentenceTransformers"
echo "Chainlit PID: $CHAINLIT_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
