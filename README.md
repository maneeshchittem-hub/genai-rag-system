# GenAI RAG System

A simple Retrieval-Augmented Generation (RAG) system built using Python, LangChain, ChromaDB, Hugging Face Embeddings, Google Gemini, and Streamlit.

## What is RAG?

RAG allows an AI model to retrieve information from a document before generating an answer.

Basic flow:

PDF → Chunks → Embeddings → ChromaDB → Retrieval → Gemini → Answer

Streamlit is used as the final web interface for interacting with the RAG system.

## Technologies Used

- Python
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Google Gemini
- PyPDF
- Python-dotenv
- Streamlit

## Project Structure

genai-rag-system/
│
├── data/
│   └── RAG_Generative_AI_Study_Guide.pdf
│
├── ingest.py
├── load_pdf.py
├── retrieve.py
├── rag.py
├── without_rag.py
├── app.py
├── test.py
├── requirements.txt
├── .gitignore
└── .env
## How It Works
Load the PDF.
Split the document into chunks.
Convert chunks into embeddings.
Store embeddings in ChromaDB.
Retrieve relevant chunks for the user's question.
Send the retrieved context to Gemini.
Gemini generates the answer.
Streamlit provides the final user interface.
Installation

1.Clone the repository:

git clone https://github.com/maneeshchittem-hub/genai-rag-system.git

2.Go to the project folder:

cd genai-rag-system

3.Create a virtual environment:

python -m venv venv

4.Activate it on Windows:

venv\Scripts\activate

5.Install dependencies:

pip install -r requirements.txt
API Key

Create a .env file in the project folder:

GOOGLE_API_KEY=YOUR_API_KEY

Do not upload the .env file to GitHub.

Run the Project
1. Load PDF
python load_pdf.py
2. Create Embeddings and ChromaDB
python ingest.py
3. Test Retrieval
python retrieve.py
4. Run RAG in Terminal
python rag.py
5. Run Without RAG
python without_rag.py
6. Run Streamlit Interface
streamlit run app.py

After running the command, Streamlit will open the RAG application in your browser.

Streamlit Interface

The Streamlit application provides the final interface where users can:

Enter questions
Retrieve information from the document
Generate answers using Gemini
Interact with the RAG system through a web interface
With RAG vs Without RAG

The same question was tested:

What threshold and retrieval depth does the Aurora RAG benchmark use?
Without RAG

The question is sent directly to Gemini without retrieving information from the document.

Result:

I do not know.
With RAG

The system retrieves the relevant information from the document and provides it to Gemini.

Result:

Aurora benchmark threshold: 0.83
Retrieval depth: 4
Difference
Without RAG:

Question → Gemini → Answer


With RAG:

Question → ChromaDB → Relevant Context → Gemini → Answer

This demonstrates how RAG allows an LLM to use information from an external document.

Author

Maneesh Chittem

GitHub:
https://github.com/maneeshchittem-hub
