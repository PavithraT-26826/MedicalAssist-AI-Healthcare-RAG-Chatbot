"""
🏥 MediAssist AI - Healthcare RAG Assistant
===========================================

An AI-powered Healthcare Assistant built using
Retrieval-Augmented Generation (RAG).

Upload healthcare PDF documents and ask questions.
The chatbot retrieves relevant medical information
from the uploaded documents and generates accurate
responses using Groq LLM.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment
load_dotenv()
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide"
)
st.title("🏥 MediAssist AI - Healthcare RAG Assistant")
st.markdown("""
This application demonstrates a complete Healthcare Retrieval-Augmented Generation (RAG) pipeline.

Upload healthcare PDF documents such as medical reports, medicine information, or disease guides, and ask questions based on their contents.

### 🏥 Healthcare RAG Pipeline

1. 📄 Load Healthcare Documents
2. ✂️ Split Medical Text into Chunks
3. 🧠 Generate Embeddings
4. 🗄️ Store in ChromaDB
5. 🔍 Retrieve Relevant Healthcare Information
6. 📝 Build Healthcare Context
7. 🤖 Generate AI Healthcare Response using Groq
""")
# ============================================================================
# SIDEBAR - Configuration
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Embedding model selection
    st.subheader("📐 Embeddings Provider")
    embedding_provider = st.radio("Choose embedding model:", ["OpenAI", "HuggingFace (Free)"])

    # LLM model selection
    st.subheader("🤖 LLM Provider")
    llm_provider = st.radio("Choose LLM model:", ["OpenAI", "Groq (Free & Fast)"])

    # Chunk parameters
    st.subheader("✂️ Chunking Parameters")
    chunk_size = st.slider("Chunk Size:", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap:", 0, 500, 200, 50)

    # Retrieval parameters
    st.subheader("🔍 Retrieval Parameters")
    k_chunks = st.slider("Number of chunks to retrieve:", 1, 10, 4)

    st.divider()

    # API Key validation
    st.subheader("🔐 API Keys")
    st.success("✅ HuggingFace (offline, no key needed)")

    api_key_llm = os.getenv("GROQ_API_KEY")
    if not api_key_llm:
        st.error("❌ GROQ_API_KEY not found in .env")
    else:
        st.success("✅ Groq LLM configured")

# ============================================================================
# MAIN APP
# ============================================================================

# File upload
st.header("📤 Upload Healthcare PDF")
uploaded_file = st.file_uploader("Upload a Healthcare PDF file", type=["pdf"])

if uploaded_file:
    # ========================================================================
    # STEP 1: LOAD - Extract text from Healthcare PDF
    # ========================================================================
    st.header("1️⃣ LOAD - Extract Text from Healthcare PDF")

    with st.spinner("Loading PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Extract text
        reader = PdfReader(tmp_path)
        raw_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted + "\n"

        os.remove(tmp_path)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Pages Loaded", len(reader.pages))
    with col2:
        st.metric("📝 Characters Extracted", len(raw_text))

    with st.expander("👀 Preview Text (First 500 characters)"):
        st.text(raw_text[:500])

    st.success("✅ Healthcare PDF loaded successfully!")

    # ========================================================================
    # STEP 2: CHUNK - Split Healthcare text into chunks
    # ========================================================================
    st.header("2️⃣ CHUNK - Split Healthcare Text into Chunks")

    with st.spinner("Chunking text..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_text(raw_text)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✂️ Number of Chunks", len(chunks))
    with col2:
        avg_chunk_size = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
        st.metric("📊 Avg Chunk Size", f"{avg_chunk_size} chars")
    with col3:
        st.metric("⚙️ Overlap", f"{chunk_overlap} chars")

    with st.expander("👀 Preview Chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.write(f"**Chunk {i+1}:**")
            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    st.success("✅ Healthcare Text chunked successfully!")

    # ========================================================================
    # STEP 3: EMBED - Convert chunks to vectors
    # ========================================================================
    st.header("3️⃣ EMBED - Convert to Vectors")

    with st.spinner("Creating embeddings..."):
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embed_info = "HuggingFace Embeddings (384 dimensions, Free)"

    st.info(f"📐 Using: {embed_info}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Total Embeddings", len(chunks))
    with col2:
        dimensions = "1536" if embedding_provider == "OpenAI" else "384"
        st.metric("📏 Vector Dimensions", dimensions)

    st.success("✅ Healthcare Embeddings created!")

    # ========================================================================
    # STEP 4: STORE - Save in ChromaDB
    # ========================================================================
    st.header("4️⃣ STORE - Save in ChromaDB")

    with st.spinner("Creating vector database..."):
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings_model,
            collection_name="rag_demo"
        )

    st.info("🗄️ All chunks + embeddings stored in ChromaDB (in-memory)")
    st.success("✅ Vector database created!")

    # ========================================================================
    # STEP 5: RETRIEVE - Find Relevant Healthcare Information
    # ========================================================================
    st.header("5️⃣ RETRIEVE - Find Relevant Healthcare Information")

    query = st.text_input(
        "💬 Ask a healthcare question:",
        placeholder="Example: What are the symptoms of diabetes?"
    )

    if query:
        with st.spinner("Searching healthcare documents..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks})
            retrieved_docs = retriever.invoke(query)

        st.metric("🔍 Retrieved Chunks", len(retrieved_docs))

        # Show retrieved chunks with similarity scores
        st.write("**Relevant Healthcare Information:**")
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"Chunk {i} - Preview"):
                st.text(doc.page_content)

        st.success("✅ Relevant healthcare information retrieved!")

        # ====================================================================
        # STEP 6 & 7: AUGMENT- Build Healthcare Context Prompt
        # ====================================================================
        st.header("6️⃣ AUGMENT -  Build Healthcare Context Prompt")

        # Show the prompt being built
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        with st.expander("👀 View Healthcare Context"):
            st.write("**Healthcare Context Sent to the AI:**")
            st.code(
                f"""You are MediAssist AI, a Healthcare Assistant.

Use ONLY the healthcare documents provided below to answer the user's question.

If the information is not available, reply:
"I couldn't find this information in the uploaded healthcare documents."


HEALTHCARE DOCUMENTS:
{context[:500]}...[truncated]""",
                language="text"
            )

        st.success("✅ Context augmented!")

        # ====================================================================
        # STEP 7:  GENERATE - Healthcare AI Response
        # ====================================================================
        st.header("7️⃣  GENERATE - Healthcare AI Response")

        with st.spinner("Generating healthcare response..."):
            api_key = os.getenv("GROQ_API_KEY")
            llm = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=api_key,
                temperature=0.2
            )
            st.info("🏥 Using: Groq Healthcare Assistant")
            # Create the full prompt
            full_prompt = f"""You are MediAssist AI, an intelligent Healthcare Assistant.

Your responsibility is to answer ONLY using the healthcare document provided below.

Instructions:
- Use only the information available in the document context.
- If the answer is not found, reply:
  "I couldn't find this information in the uploaded healthcare documents."
- Explain medical terms in simple language.
- Do NOT diagnose diseases.
- Do NOT prescribe medicines.
- Keep the answer short, clear, and easy to understand.

Healthcare Documents:
{context}

QUESTION: {query}

ANSWER:"""

            response = llm.invoke(full_prompt)
            answer = response.content

        st.success("✅ Healthcare response generated!")

        # Display the answer
        st.markdown(f"## 🤖 MediAssist AI Response")
        st.success(answer)
        st.warning(
            "⚠️ This chatbot is for educational purposes only and is not a substitute for professional medical advice."
        )

        # ====================================================================
        # SUMMARY
        # ====================================================================
        st.header("📊 Pipeline Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages", len(reader.pages))
        with col2:
            st.metric("Chunks", len(chunks))
        with col3:
            st.metric("Retrieved", len(retrieved_docs))
        with col4:
            st.metric("Answer Length", len(answer))
        st.markdown("""
        ---
        ### 🏥 Healthcare RAG Pipeline Summary

        1. ✅ Uploaded and extracted text from healthcare PDF documents
        2. ✅ Split medical content into meaningful text chunks
        3. ✅ Converted text chunks into vector embeddings using HuggingFace
        4. ✅ Stored embeddings in ChromaDB vector database
        5. ✅ Retrieved the most relevant healthcare information
        6. ✅ Combined retrieved context with the user's healthcare question
        7. ✅ Generated an AI-powered response using Groq LLM

        ### 🎯 Outcome
        The chatbot answers healthcare questions based on the uploaded medical documents using Retrieval-Augmented Generation (RAG), helping users understand medical information while reducing AI hallucinations.

        ⚠️ **Disclaimer:** This application is for educational purposes only and is not a substitute for professional medical advice.
        """)
       
else:
    st.info("👈 Upload a PDF file to start the demo")

st.divider()

st.markdown("""
### 🏥 About MediAssist AI

MediAssist AI is a Retrieval-Augmented Generation (RAG) based Healthcare Assistant.
It answers healthcare-related questions using uploaded medical documents instead of relying only on the language model.

### 🎯 Features
- 📄 Upload healthcare PDF documents
- ✂️ Extract and split document text into chunks
- 🧠 Generate embeddings using HuggingFace
- 🗄️ Store embeddings in ChromaDB
- 🔍 Retrieve the most relevant information
- 🤖 Generate AI-powered answers using Groq LLM
- 💬 Explain healthcare information in simple language

### ⚙️ Technologies Used
- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Groq API
- PyPDF

### ⚠️ Disclaimer
This application is developed for educational purposes only.
It does not diagnose diseases or replace professional medical advice.
Always consult a qualified healthcare professional for medical concerns.
""")