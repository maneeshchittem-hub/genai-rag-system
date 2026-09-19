from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

question = "What threshold and retrieval depth does the Aurora RAG benchmark use?"

results = vectorstore.similarity_search(
    question,
    k=6
)

print("Question:", question)
print("\nRetrieved Chunks:\n")

for i, doc in enumerate(results, start=1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)
    print()