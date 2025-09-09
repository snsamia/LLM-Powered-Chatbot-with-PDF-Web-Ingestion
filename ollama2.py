from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangChain API Key Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "your_default_key")
os.environ["LANGCHAIN_API_KEY"] = "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"

# LangChain Components
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# Streamlit UI
#st.set_page_config(page_title="LLM Chatbot with PDF/Web Ingestion")
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🤖 LLM Chatbot</h1>
    <p style='text-align: center; color: gray; font-size: 18px;'>
        Ask questions from PDFs, Websites, or general knowledge using LLaMA 2 + LangChain 🔍
    </p>
    <hr style='border-top: 1px solid #bbb;'>
""", unsafe_allow_html=True)

st.title('🤖 LLM Chatbot with LangChain, PDF & Web Ingestion')

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF and Web Ingestion
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
web_url = st.text_input("Or Enter a Website URL")

# Text input
input_text = st.text_input("Ask your question:")

# Load and process documents
documents = []
if uploaded_file is not None:
    # Save PDF to a temp file
    with open("temp.pdf", "wb") as f:   # <-- "wb" = write binary
        f.write(uploaded_file.read())

    # Load PDF into LangChain
    pdf_loader = PyPDFLoader("temp.pdf")
    documents.extend(pdf_loader.load())


if web_url:
    try:
        web_loader = WebBaseLoader(web_url)
        documents.extend(web_loader.load())
    except Exception as e:
        st.warning(f"Failed to load from website: {e}")

# If docs are loaded, create vectorstore
retriever = None
if documents:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Build chain (with retrieval if data provided)
if retriever:
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )
else:
    chain = prompt | llm | output_parser

# Handle input
if input_text:
    if retriever:
        response = chain.run(input_text)
    else:
        response = chain.invoke({"question": input_text})

    # Add to chat history
    st.session_state.chat_history.append((input_text, response))

    # Log chat
    with open("chat_logs.txt", "a") as f:
        f.write(f"You: {input_text}\n")
        f.write(f"Bot: {response}\n\n")

# Display chat history
if st.session_state.chat_history:
    st.subheader("🗣️ Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
