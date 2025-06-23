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

# Take the variables from the .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Verify id llm_model is set
if [ -z "$llm_model" ]; then
    echo "Error: llm_model is not set in .env file."
    exit 1
fi

# Start LLM server
echo "Starting LLM server on port 8080..."
llama-server -hf $llm_model -c 4096 &
LLM_PID=$!

# Wait a bit for server to start up
echo "Waiting for LLM server to initialize..."
sleep 10

# Start Chainlit application
# echo "Starting Chainlit application..."
# chainlit run main.py &
# CHAINLIT_PID=$!

echo "All services started!"
echo "LLM Server PID: $LLM_PID"
echo "Chainlit PID: $CHAINLIT_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
