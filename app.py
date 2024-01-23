import os
import shutil
from urllib.parse import urlparse
import docx
import gradio as gr
from bs4 import BeautifulSoup as Soup
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
class AIAssistant:
    def __init__(self):
        self.data_dir: str = './data'
        self.ollama_model_name: str = os.environ.get("OLLAMA_MODEL_NAME", "mistral:instruct")
        self.vertex_model_name:str = os.environ.get("VERTEX_MODEL_NAME", "gemini-pro")
        self.temperature: float = 0.8
        self.ollama_server_url: str = 'http://localhost:11434'
        self.llm: Ollama = None
        self.embeddings: OllamaEmbeddings = None
        self.retrieval_chain = None
        self.prompt_template: str = self._get_prompt_template()

    def _get_prompt_template(self) -> str:
        """
        Get the prompt template based on the Ollama model name.
        """
        if self.ollama_model_name in ["phi", "phi:chat"]:
            return """Instruct: Act as a customer support executive. With this context\n\n{context}\n\nQuestion: {input}\nOutput:"""
        elif self.ollama_model_name == "mistral:instruct":
            return """[INST] {{ .System }} 
            Act as a customer support executive. Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input} [/INST]"""
        else:
            return """Act as a customer support executive. Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}"""

    def _load_language_model(self, server: str = "ollama") -> None:
        """
        Load the language model based on the specified server.
        """
        try:
            if server == "ollama" or server is None:
                self.llm = Ollama(model=self.ollama_model_name, temperature=self.temperature)
            elif server == "vertex":
                self.llm = VertexAI(model_name=self.vertex_model_name)
        except Exception as e:
            print(f"Error loading language model: {e}")

    def _create_document_retrieval_chain(self, vector_db, prompt_template: str) -> None:
        """
        Create the document retrieval chain using the vector database and prompt template.
        """
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            if self.llm is None:
                self._load_language_model()
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            retriever = vector_db.as_retriever()
            self.retrieval_chain = create_retrieval_chain(retriever, document_chain)
        except Exception as e:
            print(f"Error creating document retrieval chain: {e}")

    def _vectorize_data(self, url: str = None) :
        """
        Vectorize the data, generate the vector index, and return the FAISS vector database.
        """
        try:
            print("1. Initializing vectors...")
            self.embeddings = self._get_embeddings()
            if url:
                doc_chunks = self._get_doc_chunks_from_url(url)
            else:
                doc_chunks = self._get_doc_chunks_from_file()

            vector_db = FAISS.from_documents(doc_chunks, self.embeddings)
            print("4. Vector Index generated...!")
            self._move_files()
            return vector_db
        except Exception as e:
            print(f"Error vectorizing data: {e}")
            return None

    def _get_embeddings(self, llm_server: str = "ollama") :
        """
        Get the embeddings based on the specified language model server.
        """
        try:
            if llm_server == "ollama" or llm_server is None:
                return OllamaEmbeddings(model=self.ollama_model_name, temperature=self.temperature,
                                        base_url=self.ollama_server_url, show_progress=True)
            elif llm_server == "vertex":
                embeddings = VertexAIEmbeddings()
                return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def _get_doc_chunks_from_url(self, url: str = "https://docs.python.org/3.9/") -> list:
        """
        Get document chunks from the specified URL using web scraping.
        """
        try:
            loader = RecursiveUrlLoader(
                url=url, max_depth=1,
                extractor=lambda x: Soup(x, "html.parser").text
            )
            docs = loader.load()
            print("HTML DOCS:", docs)

            if docs:
                text_splitter = RecursiveCharacterTextSplitter()
                doc_chunks = text_splitter.split_documents(docs)
                print(f"Got {len(doc_chunks)} doc chunks from the URL.")
                return doc_chunks
        except Exception as e:
            print(f"Error getting doc chunks from URL: {e}")
            return None

    def _get_doc_chunks_from_file(self) -> list:
        """
        Get document chunks from the local files in the data directory.
        """
        try:
            loader = DirectoryLoader(self.data_dir, glob="**/*.docx")
            docs = loader.load()
            pdf_loader = DirectoryLoader(self.data_dir, glob="**/*.pdf")
            pdf_docs = pdf_loader.load()

            print("2. Docs Loaded. :", len(docs))
            if docs:
                text_splitter = RecursiveCharacterTextSplitter()
                doc_chunks = text_splitter.split_documents(docs)
                if doc_chunks:
                    if pdf_docs:
                        pdf_chunks = text_splitter.split_documents(pdf_docs)
                        doc_chunks.extend(pdf_chunks)
                        print("3.2 Chunks of PDF docs generated...!", len(doc_chunks), "\n", 40 * "-")
                return doc_chunks
        except Exception as e:
            print(f"Error getting doc chunks from file: {e}")
            return None

    def _move_files(self, source_path: str = None, destination_path: str = "cached") -> None:
        """
        Move files from the source path to the destination path.
        """
        try:
            if not source_path:
                source_path = self.data_dir

            if not os.path.exists(source_path):
                print(f"Source directory '{source_path}' does not exist.")

            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]

            for file_name in files:
                if "example.docx" in file_name:
                    continue
                else:
                    source_file_path = os.path.join(source_path, file_name)
                    destination_file_path = os.path.join(destination_path, file_name)
                    shutil.move(source_file_path, destination_file_path)
        except Exception as e:
            print(f"Error moving files: {e}")

    def is_valid_url(self, text: str) -> bool:
        """
        Check if the given text is a valid URL.
        """
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def process_link(self, url: str) -> str:
        """
        Process the link, vectorize the data, and create the document retrieval chain.
        """
        try:
            if self.is_valid_url(url):
                vector_db = self._vectorize_data(url=url)
                if vector_db:
                    self._create_document_retrieval_chain(vector_db, self.prompt_template)
                    return "Processed the URL. You can ask questions now...."
                else:
                    return "Error in processing the URL."
            else:
                return "Please enter a valid working URL!"
        except Exception as e:
            print(f"Error processing link: {e}")
            return "Error processing the link."

    def predict(self, *args: str) -> str:
        """
        Predict the response based on the user's question.
        """
        try:
            if not self.retrieval_chain:
                print("Empty retrieval_chain")
                self.check_and_add_docx(self.data_dir)
                vector_db = self._vectorize_data()
                self._create_document_retrieval_chain(vector_db, self.prompt_template)
                return "Hello, I am your AI Assistant. You can upload a PDF or DOCX file and ask questions about it."

            question = args[0].strip()
            response = self.retrieval_chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            print(f"Error predicting answer: {e}")
            return "Error predicting the answer."

    def save_files(self, file_paths: list) -> str:
        """
        Save the uploaded files to the data directory and update the knowledge base.
        """
        try:
            if file_paths:
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    destination_path = os.path.join(self.data_dir, file_name)
                    shutil.copy(file_path, destination_path)
                    vector_db = self._vectorize_data()
                    if vector_db:
                        self._create_document_retrieval_chain(vector_db, self.prompt_template)
                        return "Successfully added the document into the knowledge base. You can ask questions now..."
            else:
                return "Please upload any source document."
        except Exception as e:
            print(f"Error saving files: {e}")
            return "Error saving the files."


def main() -> None:
    """
    Main function to create and launch Gradio interfaces for the AI Assistant.
    """
    ai_assistant = AIAssistant()

    file_interface = gr.Interface(
        ai_assistant.save_files,
        gr.File(file_count="multiple", file_types=[".docx", ".pdf"]), outputs="textbox"
    )

    link_interface = gr.Interface(ai_assistant.process_link, inputs="textbox", outputs="textbox")

    chatbot_interface = gr.ChatInterface(ai_assistant.predict)

    gr.TabbedInterface(
        [file_interface, link_interface, chatbot_interface], ["Upload Source Document", "Enter Source URL", "Chatbot"]
    ).launch(share=True, debug=True)


if __name__ == "__main__":
    main()