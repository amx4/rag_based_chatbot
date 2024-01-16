#!/bin/bash
ollama serve &
sleep 10 &
# Run ollama pull
ollama pull $OLLAMA_MODEL_NAME &
# Wait for ollama serve to start
sleep 300 &
# Run your Python script
python app.py