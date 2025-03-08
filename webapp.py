import os
from flask import Flask, request, Response, render_template_string
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Azure API credentials
BASE = os.getenv("azure_embedding_endpoint")
KEY = os.getenv("azure_embedding_key")
EMB_MODEL = os.getenv("azure_embedding_model")

LLM_KEY = os.getenv("azure_llm_key")
LLM_MODEL = os.getenv("azure_llm_model")
LLM_BASE = os.getenv("azure_llm_endpoint")

# Flask app
app = Flask(__name__)

def load_knowledge_base():
    embeddings = AzureOpenAIEmbeddings(
        api_key=KEY,
        openai_api_type="azure",
        azure_endpoint=BASE,
        azure_deployment=EMB_MODEL,
        dimensions=1024
    )
    db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    return AzureAIChatCompletionsModel(
        endpoint=LLM_BASE,
        credential=LLM_KEY,
        model_name=LLM_MODEL,
        temperature=0.4,
        max_tokens=400
    )

def load_prompt():
    prompt = """You are an AI assistant that helps people find information related to postgraduate courses at the University of Birmingham.\n\n    Given below is the context and question of the user.\n    content = {context}\n    question = {question}\n\n    If the answer is not in the provided context, reply that you don't know the correct answer.
    Remember that you are an AI assistant which is there to help people find information related to postgraduate courses at the University of Birmingham. Be courteous and professional."""
    return ChatPromptTemplate.from_template(prompt)

knowledge_base = load_knowledge_base()
llm = load_llm()
prompt = load_prompt()

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>UoB Postgraduate Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                #chat { max-width: 800px; margin: auto; }
                #messages { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: auto; margin-bottom: 10px; }
                .message { margin: 5px 0; padding: 5px; border-radius: 5px; }
                .user { background-color: #e3f2fd; }
                .assistant { background-color: #f5f5f5; }
                #input { width: calc(100% - 80px); padding: 8px; }
                button { width: 70px; padding: 8px; }
                h1 { text-align: center; color: #333; }
            </style>
        </head>
        <body>
            <h1>UoB Postgraduate Assistant</h1>
            <div id="chat">
                <div id="messages"></div>
                <form onsubmit="sendMessage(event)">
                    <input type="text" id="input" placeholder="Type your message...">
                    <button type="submit">Send</button>
                </form>
            </div>
            <script>
                async function sendMessage(event) {
                    event.preventDefault();
                    const input = document.getElementById('input');
                    const userMsg = input.value;
                    input.value = '';
                    appendMessage("You: " + userMsg, "user");
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: userMsg })
                        });
                        const text = await response.text();
                        appendMessage("Assistant: " + text, "assistant");
                    } catch (error) {
                        console.error('Error:', error);
                        appendMessage("Assistant: Failed to get response", "assistant");
                    }
                }
                function appendMessage(content, className) {
                    const messagesDiv = document.getElementById('messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + className;
                    messageDiv.textContent = content;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/chat', methods=['POST'])
def chat():
    
    def log_context(data):
        print("\n*********CONTEXT SENT TO LLM*********")
        print(data["context"])
        print("\n*********QUESTION SENT TO LLM*********")
        print(data["question"])
        return data

    data = request.get_json()
    query = data['message']
    results = knowledge_base.similarity_search_with_score(query, k=2)
    filtered_results = [doc for doc, score in results if score != 0]
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

    retriever = similar_embs.as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(log_context)
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)    
    # print(f"Entire prompt: {prompt}")
    return Response(response, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)