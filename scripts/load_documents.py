from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,               # ← correct import
)


def _make_loader(path: Path):
    suf = path.suffix.lower()
    if suf == ".pdf":
        return PyPDFLoader(str(path))
    if suf in {".txt", ".md", ".sql"}:
        return TextLoader(str(path), encoding="utf-8")
    if suf == ".docx":
        return Docx2txtLoader(str(path))
    return None


def load_and_split_all_documents(data_path: str = "data") -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks: List[Document] = []

    for fp in Path(data_path).rglob("*"):
        if fp.is_dir():
            continue
        loader = _make_loader(fp)
        if not loader:
            continue

        docs = loader.load()

        # Ensure every doc has source metadata + filename in text
        for d in docs:
            d.metadata["source"] = str(fp.relative_to(data_path))
            d.page_content = f"[FILE: {fp.stem}]\n{d.page_content}"

        chunks.extend(splitter.split_documents(docs))

    print(f"✅ Loaded & chunked {len(chunks)} chunks")
    return chunks

