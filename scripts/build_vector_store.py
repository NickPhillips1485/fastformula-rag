import os
from dotenv import load_dotenv
from load_documents import load_and_split_all_documents
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load .env variables
load_dotenv()

# Load and split all documents from /data
docs = load_and_split_all_documents("data")

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectordb = FAISS.from_documents(docs, embeddings)
vectordb.save_local("vectorstore")

