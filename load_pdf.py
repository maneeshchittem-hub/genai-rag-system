from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
pdf_path = "data/RAG_Generative_AI_Study_Guide.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

# Split documents
chunks = text_splitter.split_documents(documents)

print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0].page_content)

print("\nFirst chunk metadata:\n")
print(chunks[0].metadata)