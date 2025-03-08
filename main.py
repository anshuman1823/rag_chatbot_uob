from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableLambda
import os

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

BASE = os.getenv("azure_embedding_endpoint")
KEY = os.getenv("azure_embedding_key")
EMB_MODEL = os.getenv("azure_embedding_model")

LLM_KEY = os.getenv("azure_llm_key")
LLM_MODEL = os.getenv("azure_llm_model")
LLM_BASE = os.getenv("azure_llm_endpoint")

def load_knowladge_base():
    embeddings = AzureOpenAIEmbeddings(
        api_key=KEY,
        openai_api_type="azure",
        azure_endpoint=BASE,
        azure_deployment=EMB_MODEL,
        dimensions=1024
    )
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    llm = AzureAIChatCompletionsModel(
        endpoint=LLM_BASE,
        credential=LLM_KEY,
        model_name=LLM_MODEL
    )
    return llm

def load_prompt():
    prompt = """You are an AI assistant that helps people find information related to postgraduate courses at the University of Birmingham.
    Answer the questions based on the context given below.

    Given below is the context and question of the user.
    
    content = {context}
    question = {question}

    If the answer is not in the pdf answer reply that you don't know the correct answer
    """

    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

if __name__ == "__main__":
    knowledgebase = load_knowladge_base()
    llm = load_llm()
    prompt = load_prompt()
    similarity_threshold = 0

    print("********************LLM with RAG loaded successfully!********************")
    print("Enter your question: ")
    query = input()
    if query:
        results = knowledgebase.similarity_search_with_relevance_scores(query, k=3)
        print(f"Results: {results}")
        filtered_results = [doc for doc, score in results if score >= similarity_threshold]
        similar_embs = FAISS.from_documents(
            documents=filtered_results,
            embedding=AzureOpenAIEmbeddings(
                api_key=KEY,
                openai_api_type="azure",
                azure_endpoint=BASE,
                azure_deployment=EMB_MODEL,
                dimensions=1024
            )
        )

        def log_context(data):
            print("\n*********CONTEXT SENT TO LLM*********")
            print(data["context"])  # 'context' holds the retrieved documents
            return data  # Ensure data continues down the chain

        retriever = similar_embs.as_retriever()
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(log_context)
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        print(response)
