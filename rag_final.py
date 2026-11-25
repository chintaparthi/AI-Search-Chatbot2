
"""
RAG Chat with PDFs & Web Search
Streamlit Cloud Deployment Version
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    ArxivQueryRun,
    WikipediaQueryRun
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)
from langchain_core.messages import HumanMessage, SystemMessage


# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="wide"
)

# ==================== API KEY CONFIGURATION ====================

# Read API key from Streamlit Secrets.
# Use environment variable as a fallback for local development.

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY", "")


# ==================== MAIN UI ====================

st.title("🔍 RAG Chat with PDFs & Tools")

st.sidebar.title("⚙️ Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Stop if API key is missing
if not api_key:
    st.error(
        "Groq API key is not configured. "
        "Please configure GROQ_API_KEY in Streamlit Cloud Secrets."
    )
    st.info("Get your API key at: https://console.groq.com/keys")
    st.stop()


# ==================== VECTOR STORE ====================

@st.cache_resource
def get_embeddings():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def load_and_vectorize_pdfs(pdf_files):
    """Convert uploaded PDFs into a FAISS vector store."""

    try:
        documents = []

        for pdf_file in pdf_files:

            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(pdf_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if not documents:
            st.warning("No pages found in the uploaded PDFs.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        embeddings = get_embeddings()

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vector_store

    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return None


# ==================== SEARCH FUNCTIONS ====================

def search_pdfs(query, vector_store):
    """Search uploaded PDF documents."""

    if vector_store is None:
        return ""

    try:
        docs = vector_store.similarity_search(
            query,
            k=3
        )

        if not docs:
            return ""

        return "\n---\n".join(
            [
                f"📄 {doc.page_content[:300]}"
                for doc in docs
            ]
        )

    except Exception as e:
        st.warning(f"PDF search error: {str(e)}")
        return ""


def search_web(query):
    """Search the internet using DuckDuckGo."""

    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)[:500]

    except Exception:
        return ""


def search_arxiv(query):
    """Search academic research papers."""

    try:
        arxiv = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(
                top_k_results=2
            )
        )

        return arxiv.run(query)[:500]

    except Exception:
        return ""


def search_wikipedia(query):
    """Search Wikipedia."""

    try:
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=2
            )
        )

        return wiki.run(query)[:500]

    except Exception:
        return ""


# ==================== SESSION INITIALIZATION ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []


# ==================== PDF UPLOAD ====================

if uploaded_files:

    current_file_names = [
        pdf.name for pdf in uploaded_files
    ]

    if current_file_names != st.session_state.uploaded_file_names:

        with st.spinner("📚 Loading and processing PDFs..."):

            st.session_state.vector_store = load_and_vectorize_pdfs(
                uploaded_files
            )

            st.session_state.uploaded_file_names = current_file_names

        if st.session_state.vector_store is not None:
            st.success(
                f"✅ Successfully loaded {len(uploaded_files)} PDF(s)"
            )

else:

    st.session_state.vector_store = None
    st.session_state.uploaded_file_names = []


# ==================== CHAT DISPLAY ====================

st.markdown("### 💬 Chat")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ==================== USER INPUT ====================

if user_input := st.chat_input("Ask me anything..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        try:

            # Search all available sources
            st.write("🔍 Searching...")

            pdf_results = search_pdfs(
                user_input,
                st.session_state.vector_store
            )

            web_results = search_web(user_input)
            arxiv_results = search_arxiv(user_input)
            wiki_results = search_wikipedia(user_input)

            # Build context
            context = ""

            if pdf_results:
                context += (
                    f"From your PDFs:\n{pdf_results}\n\n"
                )

            if web_results:
                context += (
                    f"From Web:\n{web_results}\n\n"
                )

            if arxiv_results:
                context += (
                    f"From Academic Papers:\n{arxiv_results}\n\n"
                )

            if wiki_results:
                context += (
                    f"From Wikipedia:\n{wiki_results}\n\n"
                )

            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=api_key,
                model="openai/gpt-oss-20b",
                temperature=0.7
            )

            # System message
            system_msg = SystemMessage(
                content="""
                You are a helpful AI assistant.

                Answer questions clearly using the provided context.

                If search results are available, use them to support
                your answers.

                Be concise, accurate and helpful.
                """
            )

            # User message
            if context:

                user_msg = HumanMessage(
                    content=(
                        f"Context:\n{context}\n\n"
                        f"Question: {user_input}"
                    )
                )

            else:

                user_msg = HumanMessage(
                    content=user_input
                )

            # Generate response
            with st.spinner("💭 Generating response..."):

                response = llm.invoke(
                    [
                        system_msg,
                        user_msg
                    ]
                )

            answer = response.content

            st.write(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            error_msg = f"❌ Error: {str(e)}"

            st.error(error_msg)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg
                }
            )
