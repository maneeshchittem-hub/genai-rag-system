import streamlit as st
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="GenAI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# LOAD AI RESOURCES
# ==========================================

@st.cache_resource(show_spinner=False)
def load_resources():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return vectorstore, llm


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title("🤖 GenAI RAG Assistant")

st.write(
    "Compare a normal LLM response with a "
    "Retrieval-Augmented Generation response."
)

st.divider()


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_input(
    "Ask a question about the document",
    placeholder="Example: What is RAG?"
)


# ==========================================
# COMPARE ANSWERS BUTTON
# ==========================================

if st.button(
    "🔍 Compare Answers",
    use_container_width=True
):

    # Check whether user entered a question
    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ==========================================
        # THINKING / PROCESSING
        # ==========================================

        with st.spinner("🤔 Thinking..."):

            # Load ChromaDB and Gemini
            vectorstore, llm = load_resources()


            # ==========================================
            # WITHOUT RAG
            # ==========================================

            without_rag_prompt = f"""
Answer the following question using only your
own knowledge.

Question:
{question}

If you do not know the answer, say:
"I do not know."
"""

            without_rag_response = llm.invoke(
                without_rag_prompt
            )


            # ==========================================
            # RETRIEVAL FROM CHROMADB
            # ==========================================

            results = vectorstore.similarity_search(
                question,
                k=6
            )


            # Combine retrieved chunks
            context = "\n\n".join(
                doc.page_content
                for doc in results
            )


            # ==========================================
            # WITH RAG
            # ==========================================

            rag_prompt = f"""
You are a document-based question answering assistant.

Answer the question using ONLY the information
provided in the document context.

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

            rag_response = llm.invoke(
                rag_prompt
            )


        # ==========================================
        # ANSWER COMPARISON
        # ==========================================

        st.subheader("📊 Answer Comparison")

        col1, col2 = st.columns(2)


        # ==========================================
        # WITHOUT RAG ANSWER
        # ==========================================

        with col1:

            st.markdown(
                "### ❌ Without RAG"
            )

            st.info(
                without_rag_response.content[0]['text']
            )


        # ==========================================
        # WITH RAG ANSWER
        # ==========================================

        with col2:

            st.markdown(
                "### ✅ With RAG"
            )

            st.success(
                rag_response.content[0]['text']
            )


        # ==========================================
        # RETRIEVED SOURCES
        # ==========================================

        st.divider()

        st.subheader(
            "📚 Retrieved Sources"
        )

        st.caption(
            "These document chunks were retrieved "
            "from ChromaDB and provided to the LLM."
        )


        # Display retrieved chunks
        for i, doc in enumerate(
            results,
            start=1
        ):

            page = doc.metadata.get(
                "page",
                "Unknown"
            )


            with st.expander(
                f"Source {i} — Page {page}"
            ):

                st.write(
                    doc.page_content
                )