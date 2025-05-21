import os
from dotenv import load_dotenv
from load_documents import load_and_split_all_documents
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load API key
load_dotenv()

print("📚 Loading & chunking documents …")
docs = load_and_split_all_documents("data")

print("🔢 Building embeddings & FAISS index …")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",             # ⬅ NEW, more accurate model
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
vectordb = FAISS.from_documents(docs, embeddings)
vectordb.save_local("vectorstore")

print("✅ Vector store saved to ./vectorstore")
