from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# -------------------------
# Gemini model
# -------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

question = "What threshold and retrieval depth does the Aurora RAG benchmark use?"

# -------------------------
# WITHOUT RAG
# -------------------------

without_rag_prompt = f"""
Answer the following question using only your own knowledge.

Question:
{question}

If you do not know the answer, say that you do not know.
"""

without_rag_response = llm.invoke(without_rag_prompt)

# -------------------------
# WITH RAG
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

results = vectorstore.similarity_search(
    question,
    k=6
)

context = "\n\n".join(
    doc.page_content for doc in results
)

with_rag_prompt = f"""
Answer the question using ONLY the supplied document context.

Document Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I could not find the answer in the provided document."
"""

with_rag_response = llm.invoke(with_rag_prompt)

# -------------------------
# RESULTS
# -------------------------

print("\n==============================")
print("WITHOUT RAG")
print("==============================")

print(without_rag_response.content)

print("\n==============================")
print("WITH RAG")
print("==============================")

print(with_rag_response.content)

print("\n==============================")
print("RETRIEVED CHUNKS")
print("==============================")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content)