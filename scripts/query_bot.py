import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Load vector store with embeddings
db = FAISS.load_local(
    "vectorstore",
    OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    allow_dangerous_deserialization=True
)

# Use a deeper retrieval pool
retriever = db.as_retriever(search_kwargs={"k": 15})

# Define improved prompt
prompt_template = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
You are an expert in Oracle Fusion HCM, specialising in Payroll and Fast Formula.

Use the following document context to answer the question accurately and concisely. 
If there is a relevant example (such as a formula or SQL snippet), quote it directly. 

Question: {question}
Context: {summaries}

Answer:
"""
)

# Set up the LLM
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0
)

# Create QA chain with source citation support
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Interactive loop
print("\n📘 Fast Formula Assistant. Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()
    if query.lower() in {"exit", "quit"}:
        break

    result = qa_chain.invoke({"question": query})

    print("\n💡 Answer:\n", result["answer"].strip())

    print("\n📚 Source Preview:")
    for doc in result["source_documents"]:
        filename = os.path.basename(doc.metadata.get("source", ""))
        preview = doc.page_content.strip().split("\n")[0][:100]
        print(f" - {filename}: {preview}...")

    print()

