# 🤖 LLM-Powered Chatbot with PDF & Web Ingestion

A chatbot application built with **LangChain**, **LLaMA 2 (via Ollama)**, **HuggingFace embeddings**, and **FAISS**.  
The app allows users to upload PDFs or enter website URLs, then ask natural language questions that are answered using a **retrieval-augmented generation (RAG)** pipeline.  

Built with **Streamlit** for an interactive UI.

---

## ✨ Features
- 📄 **PDF ingestion** → Upload a PDF and query its contents.  
- 🌐 **Web ingestion** → Provide a website URL and ask questions from its text.  
- 🧠 **LLM backend** → Uses **LLaMA 2 (Ollama)** with LangChain integration.  
- 🔍 **RAG pipeline** → Text is chunked, embedded using HuggingFace MiniLM, and stored in FAISS for efficient retrieval.  
- 💬 **Chat history** → Keeps track of previous questions and responses.  
- 📝 **Logging** → Saves all conversations in a `chat_logs.txt` file.  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – UI framework  
- [LangChain](https://www.langchain.com/) – LLM orchestration  
- [Ollama](https://ollama.ai/) – Running LLaMA 2 locally  
- [HuggingFace Sentence Transformers](https://www.sbert.net/) – Embeddings (`all-MiniLM-L6-v2`)  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector store  

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llm-pdf-web-chatbot.git
   cd llm-pdf-web-chatbot

