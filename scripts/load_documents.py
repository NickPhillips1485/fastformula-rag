from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import docx2txt

def load_docx_with_docx2txt(path):
    text = docx2txt.process(path)
    return [Document(page_content=text, metadata={"source": str(path)})]

def load_and_split_all_documents(data_path="data"):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    for file_path in Path(data_path).rglob("*"):
        print(f"Found file: {file_path.name}")

        # Skip very large PDFs for now
        if file_path.name == "administering-fast-formulas.pdf":
            print("Skipping large PDF for now (known crash).")
            continue

        # Load supported file types
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
            elif file_path.suffix.lower() in [".txt", ".md", ".sql"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents = loader.load()
            elif file_path.suffix.lower() == ".docx":
                documents = load_docx_with_docx2txt(str(file_path))
            else:
                print(f"Skipping unsupported file: {file_path}")
                continue

            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"{file_path.name} → {len(chunks)} chunks")

        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    return all_chunks
