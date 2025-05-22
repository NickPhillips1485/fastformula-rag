import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from markdown import markdown

# ── ENV ──────────────────────────────────────────────────────────────
load_dotenv()

# ── VECTOR STORE ─────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
db = FAISS.load_local(
    "vectorstore", embeddings,
    allow_dangerous_deserialization=True,
)

# ── HYBRID RETRIEVER (Vector + BM25) ────────────────────────────────
texts = []
metadatas = []
for doc in db.docstore._dict.values():
    texts.append(doc.page_content)
    metadatas.append(doc.metadata)

bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas, k=5)

retriever: EnsembleRetriever = EnsembleRetriever(
    retrievers=[db.as_retriever(), bm25],
    weights=[0.6, 0.4],
)

# ── LLM ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ── PROMPT TEMPLATE ─────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an Oracle Fusion Fast-Formula assistant.

• If the user’s question names an exact file you have indexed,
  reproduce that file’s content inside triple back-ticks (```).
• Otherwise answer normally and list your sources.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT.strip(),
)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context",
    },
)

# ── FLASK APP ───────────────────────────────────────────────────────
app = Flask(__name__)

def looks_like_code(text: str) -> bool:
    t = text.strip()
    return (
        "```" in t
        or t.startswith("/*")
        or t.upper().startswith("CREATE ")
        or "DEFAULT_DATA_VALUE" in t
    )

@app.route("/", methods=["GET", "POST"])
def index():
    answer_html, sources = "", []
    query = ""

    if request.method == "POST":
        query = request.form.get("question", "").strip()
        if query:
            result = qa_chain({"question": query})
            raw = result.get("answer", "")
            sources = [s for s in result.get("sources", "").split("\n") if s]

            if looks_like_code(raw):
                cleaned = raw.replace("```sql", "").replace("```", "").strip()
                answer_html = f"<pre><code>{cleaned}</code></pre>"
            else:
                answer_html = markdown(raw)

    return render_template("index.html", result=answer_html, query=query, sources=sources)

if __name__ == "__main__":
    app.run(debug=True)


