import shutil
import os

import gradio as gr
import docx
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

retrieval_chain = None
DATA_DIR = './data'
ollama_model_name = os.environ.get("OLLAMA_MODEL_NAME", "mistral:instruct") #"phi" #mistral:instruct"
temperature=0.8
OLLAMA_SERVER_URL = 'http://localhost:11434'

if ollama_model_name == "phi" or ollama_model_name == "phi:chat":
    prompt_template = """Instruct: Act as a customer support executive. With this context\n\n{context}\n\nQuestion: {input}\nOutput:"""
elif  ollama_model_name == "mistral:instruct":
    prompt_template = """[INST] {{ .System }} 
     Act as a customer support executive. Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input} [/INST]"""
else:
    prompt_template = """Act as a customer support executive. Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}"""
    

def check_and_add_docx(directory:str = DATA_DIR):
    """
    Checks if the specified directory exists and creates it if not. If the directory is empty,
    generates a sample Word document (".docx") named "example.docx" with a simple paragraph.
    This function is designed to initialize the vector database and create the retrieval chain.
    """
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the list of files in the directory
    files = os.listdir(directory)

    # Check if the directory is empty
    if not files:
        # If it's empty, add a .docx file
        docx_file_path = os.path.join(directory, "example.docx")

        # Create a simple Word document
        doc = docx.Document()
        doc.add_paragraph("Hello, this is a sample document to make you understand.")
        doc.save(docx_file_path)

        print(f"A new .docx file '{docx_file_path}' has been added to the directory.")
    


def load_language_model(model_name="phi", temperature=0.8):
    """
    Loads the Language Model (LLM) from the Ollama server with the specified model name and temperature.
    This function initializes a global variable `llm` and returns it.
    """
    global llm
    llm = Ollama(model=model_name, temperature=temperature)
    return llm


def create_document_retrieval_chain(vector_db, prompt_template):
    """
    Creates a document retrieval chain using the provided vector database and prompt template.
    This function initializes a global variable `retrieval_chain` and returns it.
    """
    global llm
    global retrieval_chain
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    #print(f"Generated the {retrieval_chain = }, {type(retrieval_chain)}")
    return retrieval_chain

def generate_test_retrieval_chain(questions: list = None):
    """
    Generates a test retrieval chain by invoking the global `retrieval_chain` with a list of questions.
    If no questions are provided, it uses a default set of test questions.
    Returns a list of question-answer pairs.
    """
    global retrieval_chain
    if questions is None:
        questions = [
        "What is the current price of ASUS Vivobook Go 15 (OLED) 2023  M1505YA?",
        "Can I avail of EMI options for this laptop?",
        "What are the specifications of the ASUS Vivobook Go 15 (OLED) 2023?",
        "Are there any discounts available on HSBC Credit Card EMI transactions?",
        "How can I avail the HSBC Credit Card EMI offers?",
        "What are the color options available for the Vivobook Go 15 (OLED)?"
    ]
    qa_pairs = []
    n = len(questions) 
    print("Testing started....\n", 40*"%%")
    #print(retrieval_chain)
    if not retrieval_chain:
        vector_db = vectorize_data()
        if vector_db:
            retrieval_chain = create_document_retrieval_chain(vector_db, prompt_template)
    for index, question in enumerate(questions):
        question = question.strip()
        
        print(f"\n{index + 1}/{n}: {question}")

        response = retrieval_chain.invoke({"input": question})
        qa_pairs.append((question.strip(), response["answer"]))  # Add to our output array

    return qa_pairs

def test_backend_for_rag(qa_pairs):
    """
    Tests the backend for the RAG (Retrieval Augmented Generation) model by printing question-answer pairs.
    """
    for index, (question, answer) in enumerate(qa_pairs, start=1):
        print(f"{index}/{len(qa_pairs)} {question}\n\n{answer}\n\n--------\n")


def predict(*args):
    """
    Predicts the answer using the global `retrieval_chain` based on the provided question.
    If `retrieval_chain` is empty, initializes it by invoking `check_and_add_docx` and `vectorize_data`.
    """
    global retrieval_chain
    if not retrieval_chain:
        print("Empty retrieval_chain")
        check_and_add_docx(DATA_DIR)
        vector_db = vectorize_data()
        retrieval_chain = create_document_retrieval_chain(vector_db, prompt_template)
        return "Hello, I am your AI Assistant. You can upload a PDF or DOCX file and questions about it."  # Provide a default response
    #print(retrieval_chain)
    question = args[0].strip()  # Assuming the input data is the first argument
    response = retrieval_chain.invoke({"input": question})

    return response["answer"]


def save_files(file_paths):
    """
    Moves files from the temporary location to the specified DATA_DIR and updates the global `retrieval_chain`.
    """
    global retrieval_chain
    if file_paths :
        for file_path in file_paths:
            # Copy the file from the temporary location to DATA_DIR
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(DATA_DIR, file_name)
            shutil.copy(file_path, destination_path)
            #print(f"Moved: from {file_path} to {destination_path}")
            vector_db = vectorize_data()
            if vector_db:
                retrieval_chain = create_document_retrieval_chain(vector_db, prompt_template)



def move_files(source_path = DATA_DIR, destination_path = "cached"):
    """
    Moves files from the source directory to the destination directory, excluding the 'example.docx' file.
    """
    # Ensure the source directory exists
    if not os.path.exists(source_path):
        print(f"Source directory '{source_path}' does not exist.")

    # Ensure the destination directory exists, create it if not
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Get a list of all files in the source directory
    files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]

    # Move each file to the destination directory
    for file_name in files:
        if "example.docx" in file_name:
            continue
        else:
            source_file_path = os.path.join(source_path, file_name)
            destination_file_path = os.path.join(destination_path, file_name)
            # Use shutil.move to perform the file move operation
            shutil.move(source_file_path, destination_file_path)
            
def vectorize_data():
    """
    Initializes vectors using the Ollama model and creates a vector database from '.docx' and '.pdf' documents
    in the specified DATA_DIR. Updates the global `retrieval_chain` and returns the vector database.
    
    """
    vector_db = None
    pdf_docs = None
    print("1. Initializing vectors...")
    embeddings = OllamaEmbeddings(model=ollama_model_name,  temperature=temperature, base_url=OLLAMA_SERVER_URL, show_progress=True)
    loader = DirectoryLoader(DATA_DIR, glob="**/*.docx")
    docs = loader.load()
    try:
        pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf")
        pdf_docs = pdf_loader.load()
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
    print("2. Docs Loaded. :", len(docs))
    if docs:
        text_splitter = RecursiveCharacterTextSplitter()
        doc_chunks = text_splitter.split_documents(docs)        
        if doc_chunks:
            if pdf_docs:
                pdf_chunks = text_splitter.split_documents(pdf_docs)
                doc_chunks.extend(pdf_chunks)
                print("3.2 Chunks of PDF docs generated...!", len(doc_chunks),"\n",40*"-", embeddings)
            #vector_db = Chroma.from_documents(documents, embeddings)
            vector_db = FAISS.from_documents(doc_chunks, embeddings)
            print("4. Veectors generated...!")
            print("VD", vector_db)
    #move files so that it wont create redundunt data
    move_files()
    return vector_db

    
def main():
    global retrieval_chain  
    global llm
    #check_and_add_docx(DATA_DIR)

    llm = load_language_model(ollama_model_name)
    # test_qa_pairs = generate_test_retrieval_chain()
    # test_backend_for_rag(test_qa_pairs)

    # Create the Gradio ChatInterface
    file_interface = gr.Interface(
        save_files,
        gr.File(file_count="multiple", file_types=[".docx", ".pdf"]), outputs = "label"
    )
    # Create the Gradio ChatInterface
    chatbot_interface = gr.ChatInterface(predict)

    # Launch the Gradio UI
    gr.TabbedInterface(
        [file_interface, chatbot_interface], ["Upload", "Chatbot"]
    ).launch(share=True, debug=True)



if __name__ == "__main__":
    
    main()