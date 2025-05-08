import os
import time
from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from flask_session import Session
from werkzeug.middleware.proxy_fix import ProxyFix

# App setup
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'fallback-secret-key')
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # 1 hour session lifetime
Session(app)

# For deployment behind a proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

# Load vector store once
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load LLM
def load_llm():
    model = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return model

# Custom Prompt Template
def set_custom_prompt():
    template_text = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=template_text, input_variables=["context", "question"])

# Initialize components once at startup
vectorstore = get_vectorstore()
llm = load_llm()
prompt = set_custom_prompt()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

@app.route("/")
def home():
    if "messages" not in session:
        session["messages"] = []
    return render_template("chat.html", messages=session["messages"])

@app.route("/chat", methods=["POST"])
def chat():
    if "messages" not in session:
        session["messages"] = []

    user_input = request.form.get("prompt")
    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    session["messages"].append({"role": "user", "content": user_input})

    try:
        start_time = time.time()
        response = qa_chain.invoke({'query': user_input})
        result = response["result"]
        source_docs = [doc.metadata for doc in response["source_documents"]]
        
        session["messages"].append({
            "role": "assistant", 
            "content": result,
            "sources": source_docs
        })
        
        return jsonify({
            "response": result,
            "sources": source_docs,
            "processing_time": time.time() - start_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    session["messages"] = []
    return jsonify({"status": "chat cleared"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))