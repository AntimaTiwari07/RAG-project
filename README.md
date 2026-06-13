# 🤖 AI PDF Assistant - RAG Application

A Retrieval-Augmented Generation (RAG) application built using **LangChain**, **Mistral AI**, **ChromaDB**, and **Streamlit** that allows users to upload PDF documents and interact with them using natural language queries.

---

## 🚀 Features

* 📄 Upload your own PDF documents
* 💬 Chat with PDFs using AI
* 🔍 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* ⚡ Mistral AI LLM Integration
* 🗂️ Chroma Vector Database
* 📚 Automatic PDF chunking
* 🎯 MMR (Maximum Marginal Relevance) Retrieval
* 🖥️ Modern Streamlit Interface
* 📑 Source Context Viewer
* 🔄 Dynamic PDF Processing

---

## 🏗️ Project Architecture

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Splitting
(Chunk Size = 1000)
    ↓
Mistral Embeddings
(codestral-embed)
    ↓
Chroma Vector Store
    ↓
MMR Retriever
    ↓
Prompt Template
    ↓
Mistral LLM
(mistral-small-2506)
    ↓
AI Response
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python
* LangChain

### Embeddings

* Codestral Embed (`codestral-embed`)

### LLM

* Mistral Small (`mistral-small-2506`)

### Vector Database

* ChromaDB

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

---

## 📂 Project Structure

```text
RAG-Project/
│
├── app.py
├── create_database.py
├── chroma-db/
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/RAG-project.git
cd RAG-project
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

Get your API key from:

https://console.mistral.ai/

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

## 📖 How It Works

### Step 1: Upload PDF

Users upload a PDF document through the Streamlit interface.

### Step 2: Document Processing

The PDF is:

* Loaded using PyPDFLoader
* Split into chunks
* Embedded using Codestral Embeddings

### Step 3: Vector Storage

Chunks are stored inside ChromaDB.

### Step 4: Retrieval

When a user asks a question:

* Similar chunks are retrieved
* MMR retrieval ensures diversity
* Relevant context is collected

### Step 5: Response Generation

The retrieved context and question are passed to Mistral AI to generate the final answer.

---

## 📸 Features Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* Prompt Engineering
* LangChain Pipelines
* Embeddings
* Large Language Models
* Streamlit Deployment

---

## 🎯 Learning Outcomes

This project demonstrates understanding of:

* Generative AI
* LangChain Framework
* Vector Databases
* Embedding Models
* Prompt Engineering
* Document Question Answering
* Retrieval Systems
* Streamlit Development

---

## 🔮 Future Improvements

* Multi-PDF Support
* Persistent User Workspaces
* Conversation Memory
* PDF Preview Panel
* Citation Generation
* Hybrid Search
* Reranking Models
* Authentication System
* Cloud Deployment

---

## 👩‍💻 Author

Antima Tiwari

Final Year B.Tech Student

Interested in:

* Artificial Intelligence
* Machine Learning
* Retrieval-Augmented Generation
* Generative AI
* Full Stack Development

---

## ⭐ If you found this project useful

Give the repository a star and share it with others.
