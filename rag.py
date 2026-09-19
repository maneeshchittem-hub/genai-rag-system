from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# -------------------------
# 1. Create embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 2. Connect to ChromaDB
# -------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# -------------------------
# 3. Create Gemini LLM
# -------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -------------------------
# 4. Ask a question
# -------------------------

question = input("\nEnter your question: ")

# -------------------------
# 5. Retrieve relevant chunks
# -------------------------

results = vectorstore.similarity_search(
    question,
    k=6
)

# -------------------------
# 6. Create context
# -------------------------

context = "\n\n".join(
    doc.page_content for doc in results
)

# -------------------------
# 7. Create grounded prompt
# -------------------------

prompt = f"""
You are a document-based question answering assistant.

Answer the question using ONLY the information provided
in the document context.

Document Context:
{context}

Question:
{question}

Rules:
- Use only the provided document context.
- Do not make up information.
- If the answer is not present in the context, say:
  "I could not find the answer in the provided document."
"""

# -------------------------
# 8. Generate answer
# -------------------------

response = llm.invoke(prompt)

# -------------------------
# 9. Display answer
# -------------------------

print("\n==============================")
print("RAG ANSWER")
print("==============================")

print(response.content)

# -------------------------
# 10. Display sources
# -------------------------

print("\n==============================")
print("RETRIEVED SOURCES")
print("==============================")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Chunk {i} ---")
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:500])