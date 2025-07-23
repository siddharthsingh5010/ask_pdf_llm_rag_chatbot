# 📄 Smart ChatBot 🤖

This is a simple **Streamlit ChatBot** app powered by **OpenAI**, **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **FAISS** for vector storage.  
The app allows users to **upload a text-based PDF document** and ask natural language questions related to the content of the uploaded file.

---

## 🚀 Features

- 🔥 Ask questions from any uploaded PDF document  
- 💡 Uses OpenAI (`gpt-3.5-turbo`) to generate smart answers  
- 🔎 Powered by LangChain's RAG framework and FAISS vector store  
- 📚 Dynamically parses PDFs and generates embeddings  
- 🧠 Retrieval-based contextual answering with document chunking

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) - UI frontend  
- [LangChain](https://www.langchain.com/) - Chain and retrieval logic  
- [OpenAI](https://platform.openai.com/docs/models) - LLM for answer generation  
- [FAISS](https://github.com/facebookresearch/faiss) - Local in-memory vector storage  
- PDF document parsing via `PyPDFLoader`

---

## 📦 Folder Structure

```
your-repo/
├── app.py             # Main Streamlit app file
└── README.md           # Project documentation
```

---

## 💡 How It Works

1. **Upload** a PDF document (text-based only).  
2. It gets split into chunks using LangChain’s `RecursiveCharacterTextSplitter`.  
3. Each chunk is converted into a vector using OpenAI Embeddings (`text-embedding-3-small` or similar).  
4. Vectors are stored in a temporary FAISS index.  
5. At query time, most relevant chunks are retrieved and passed as context to GPT.  
6. GPT returns an answer grounded in the uploaded document.

---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

Create a virtual environment and install required packages:

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API Key

You can export it in your terminal session:

```bash
export openai_key=sk-...your_key_here...
```

Or add this in your shell config (`~/.bashrc`, `~/.zshrc`, etc.).

### 4. Run the app

```bash
streamlit run app.py
```

---

---

## 🔐 Notes

- This app only supports **text-based PDFs** (not scanned images).  
- For best performance, make sure your OpenAI API key has access to `gpt-3.5-turbo`.

---

## 📄 License

MIT License

Author
Siddharth Singh
