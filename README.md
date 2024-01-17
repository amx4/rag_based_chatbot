# Customer Support Chatbot 

## Brief Description
This repository contains a chat interface utilizing the Ollama language model for document retrieval and question answering. The interface allows users to interact with the language model either by uploading documents (in `.docx` or `.pdf` format) or by asking questions directly. The code is built using Gradio for the user interface.

## Installation Guide

### Prerequisites
- Linux or Unix-based environment (WSL is also supported)

### Steps
0. Clone this repository on your machine:
    ```bash
    git clone https://github.com/amx4/rag_based_chatbot/
    cd rag-based-chatbot
    ```

1. Install Ollama on your host machine:
   
    1.1 Install Ollama by following the instructions at `https://ollama.ai/download/linux`.
    
    1.2 Run the command to confirm the installation of Ollama server:
    ```bash
    ollama
    ```
    
    1.3 Pull the desired language model (e.g., mistral:instruct) using the following command:
    ```bash
    ollama pull mistral:instruct
    ```
    
3. Create a virtual environment and install the required Python packages:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4. Install additional packages for PDF support:
    ```bash
    pip install "unstructured[pdf]"
    ```

5. After downloading the necessary dependencies, you can run the UI using:
    ```bash
    python3 app.py
    ```

Now, the Ollama Language Model Chat Interface should be accessible at `http://localhost:7860`. You can upload documents or ask questions directly through the provided interface.
