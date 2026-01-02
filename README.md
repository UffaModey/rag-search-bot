# AI-Powered Internal Search Bot (RAG)

This project is a prototype **AI-powered internal search bot** built using **Retrieval-Augmented Generation (RAG)**. It allows users to ask natural language questions about internal company documents (such as HR policies, handbooks, and guides stored as PDFs) and receive **clear, human-readable answers grounded strictly in those documents**.

The application is designed to simulate conversational interactions while ensuring responses are **accurate, explainable, and privacy-preserving**.

---

## Features

* Semantic search over internal PDF documents
* Retrieval-Augmented Generation (RAG) using LangChain
* FAISS-powered vector similarity search
* Grounded answers with source citations (file name + page number)
* Streamlit-based interactive UI
* Single LLM call per query for efficient inference
* Designed for sensitive, non-public documents

---

## Tech Stack

* **Python 3.10+**
* **LangChain** – orchestration framework
* **OpenAI API** – embeddings and LLM inference
* **FAISS** – vector indexing and similarity search
* **Streamlit** – UI layer
* **PyPDF** – PDF parsing

---

## Project Structure

```
.
├── app.py                 # Streamlit application entry point
├── documents/             # Folder containing source PDF files
│   ├── policy_1.pdf
│   ├── handbook.pdf
│   └── ...
├── requirements.txt       # Project dependencies (optional)
├── .env                   # Environment variables (not committed)
└── README.md
```

---

## How It Works (High-Level)

1. **Document Ingestion**
   PDF documents are loaded from a directory using LangChain’s `DirectoryLoader`.

2. **Text Splitting**
   Large documents are split into overlapping chunks to preserve context.

3. **Embedding & Indexing**
   Each chunk is converted into a vector embedding and stored in a FAISS index.

4. **Retrieval**
   User queries are embedded and matched against the vector store using semantic similarity.

5. **Generation**
   Retrieved chunks are passed as context to an LLM, which generates a response constrained strictly to the provided content.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/UffaModey/rag-search-bot.git
cd rag-search-bot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install streamlit python-dotenv faiss-cpu openai \
langchain langchain-text-splitters langchain-community pypdf
```

---

## Environment Variables

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, the app will prompt you to enter the key at runtime if it’s not found.

> **Note:** API-based usage of OpenAI models does not add your data to the model’s training set.

---

## Adding Documents

1. Place all PDF files you want the bot to reference inside the `documents/` directory.
2. Ensure the files are readable (not scanned images without OCR).
3. Restart the application if documents are added or removed.

---

## Running the Application

Start the Streamlit app locally:

```bash
python -m streamlit run app.py
```

Once running:

* Enter a natural language question related to the documents
* View the generated answer
* Expand the **Sources** section to see cited documents and page numbers

---

## Example Queries

* "How many days of annual leave do I get?"
* "What is the company’s sick leave policy?"
* "Is remote work allowed, and under what conditions?"

If the answer is not present in the documents, the assistant will respond accordingly.

---

## Limitations

* This is a prototype and not production-hardened
* No authentication or access control
* No document-level permissioning
* Not optimised for very large document collections
* Assumes clean, text-based PDFs

---

## Possible Improvements

* Add user authentication and role-based access control
* Support additional document formats (DOCX, HTML, Markdown)
* Add re-ranking for improved retrieval quality
* Introduce conversation memory
* Swap FAISS for a managed vector database
* Add observability and evaluation metrics

---

## Why RAG?

Retrieval-Augmented Generation enables AI systems to:

* Work with **private, sensitive data**
* Produce **grounded, explainable answers**
* Avoid hallucinations
* Keep knowledge sources up to date without retraining models

This makes it particularly well-suited for internal enterprise use cases.

---

## License

This project is provided for educational and experimental purposes. Add a license if you intend to distribute or open-source it.

---

## Author

Built by **Fafa Modey**

AI Engineer 

---

If you have questions, ideas, or want to extend this project, feel free to fork it and experiment.
