from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

BASE = os.getenv("azure_embedding_endpoint")
KEY = os.getenv("azure_embedding_key")
EMB_MODEL = os.getenv("azure_embedding_model")

pdf_folder_path = os.path.join(os.getcwd(), "data")
pdf_files = [os.path.join(pdf_folder_path,f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

if __name__ == "__main__":
    DB_FAISS_PATH = "vectorstore/db_faiss"

    # Load documents from multiple PDFs
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in FAISS vectorstore
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=AzureOpenAIEmbeddings(
            api_key=KEY,
            openai_api_type="azure",
            azure_endpoint=BASE,
            azure_deployment=EMB_MODEL,
            dimensions=1024
        )
    )

    # Save the vectorstore locally
    vectorstore.save_local(DB_FAISS_PATH)
