# fastformula-rag

# 🧠 Fast Formula Assistant

This is a local Retrieval-Augmented Generation (RAG) assistant designed to help with Oracle Fusion Fast Formula queries. It uses a hybrid search approach (vector + keyword) to answer questions based on a personal library of Fast Formula examples, documentation, and notes.

The assistant is built using:
- **Flask** for the web interface
- **LangChain** for retrieval + QA chain logic
- **OpenAI GPT-4o** as the LLM
- **FAISS** and **BM25** for hybrid document retrieval
- **Markdown rendering** and syntax detection for clean responses

---

## 🚀 Features

- Ask free-text questions related to Oracle Fast Formula
- See answers with source references
- Returns full Fast Formula code when requested by name
- Hybrid retriever (semantic + keyword) ensures more accurate recall
- Lightweight Flask web interface with multiline support and keyboard shortcuts

---

## 📁 Project Structure

fastformula-rag/
│
├── app.py # Flask app with LangChain pipeline
├── templates/
│ └── index.html # Web UI
│
├── data/ # All documents (Word, Markdown, SQL, etc.)
│ └── Absence Management Fast Formula Examples/
│ └── *.sql # Fast Formula examples
│
├── scripts/
│ ├── load_documents.py # Loads and splits documents
│ └── build_vector_store.py # Rebuilds FAISS vector index
│
├── vectorstore/ # Auto-generated vector DB (after indexing)
├── .env # API keys and model settings
└── requirements.txt # Dependencies


1. Add Environment Variables
Create a .env file in the project root:

env
Copy
Edit
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o
2. Add Your Data
Place your source documents (e.g. .sql, .docx, .md, etc.) into the /data folder. You can organise them into subfolders if needed.

3. Build the Vector Store
bash
Copy
Edit
python scripts/build_vector_store.py
This parses and embeds your documents, readying them for search.

4. Run the App
bash
Copy
Edit
python app.py
Then visit http://127.0.0.1:5000 in your browser.

🛠️ Editing and Updating
To add or amend source material, update the data/ folder then re-run:

python scripts/build_vector_store.py
To change the behaviour or prompt of the assistant, edit the SYSTEM_PROMPT in app.py.

🧩 Future Plans
Host the assistant on a private cloud or public web server

Expand to additional assistants (e.g. Compensation, Payroll, Journeys)

Improve handling of ambiguous or partial questions

Add download/share/export options for retrieved code

📝 Notes
This tool is intended for personal/internal use to support Oracle Fusion HCM project work, particularly involving Fast Formula implementation and troubleshooting. It’s not affiliated with Oracle.