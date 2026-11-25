"""
RAG Chat with PDFs & Web Search
FINAL WORKING VERSION - LangChain 0.2+
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# ✅ CORRECT IMPORTS FOR LANGCHAIN 0.2+
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ==================== UI ====================
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("🔍 RAG Chat with PDFs & Tools")

st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key:", type="password")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ==================== VECTOR STORE ====================
def load_and_vectorize_pdfs(pdf_files):
    """Convert PDFs to vector store"""
    try:
        documents = []
        
        for pdf_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
            os.unlink(tmp_path)
        
        if not documents:
            st.warning("No pages found in PDFs")
            return None
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ==================== SEARCH FUNCTIONS ====================
def search_pdfs(query, vector_store):
    """Search PDFs"""
    if not vector_store:
        return ""
    try:
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return ""
        return "\n---\n".join([f"📄 {doc.page_content[:300]}" for doc in docs])
    except:
        return ""

def search_web(query):
    """Search web"""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)[:500]
    except:
        return ""

def search_arxiv(query):
    """Search academic papers"""
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
        return arxiv.run(query)[:500]
    except:
        return ""

def search_wikipedia(query):
    """Search Wikipedia"""
    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        return wiki.run(query)[:500]
    except:
        return ""

# ==================== MAIN ====================
if not api_key:
    st.warning("⚠️ Enter Groq API key in sidebar")
    st.info("Get free key at: https://console.groq.com/keys")
else:
    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    # Load PDFs
    if uploaded_files:
        if not st.session_state.vector_store:
            with st.spinner("📚 Loading PDFs..."):
                st.session_state.vector_store = load_and_vectorize_pdfs(uploaded_files)
            st.success(f"✅ Loaded {len(uploaded_files)} PDF(s)")
    
    # Display chat
    st.markdown("### 💬 Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # User input
    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            try:
                # Search all sources
                st.write("🔍 Searching...")
                
                pdf_results = search_pdfs(user_input, st.session_state.vector_store)
                web_results = search_web(user_input)
                arxiv_results = search_arxiv(user_input)
                wiki_results = search_wikipedia(user_input)
                
                # Build context
                context = ""
                if pdf_results:
                    context += f"**From your PDFs:**\n{pdf_results}\n\n"
                if web_results:
                    context += f"**From Web:**\n{web_results}\n\n"
                if arxiv_results:
                    context += f"**From Academic Papers:**\n{arxiv_results}\n\n"
                if wiki_results:
                    context += f"**From Wikipedia:**\n{wiki_results}\n\n"
                
                # Create LLM
                llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name="openai/gpt-oss-20b",
                    temperature=0.7
                )
                
                # Build messages
                system_msg = SystemMessage(content="""You are a helpful AI assistant. 
Answer questions clearly using the provided context.
If you get search results, cite them. Be concise and helpful.""")
                
                if context:
                    user_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {user_input}")
                else:
                    user_msg = HumanMessage(content=user_input)
                
                # Get response
                with st.spinner("💭 Generating response..."):
                    response = llm.invoke([system_msg, user_msg])
                
                answer = response.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
