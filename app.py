import os
import getpass
import streamlit as st
from dotenv import load_dotenv
import glob

import faiss
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------
# Environment setup
# -----------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# -----------------------
# Streamlit config
# -----------------------
st.set_page_config(
    page_title="AI Policy Search Bot",
    page_icon="📄",
    layout="centered",
)

st.title("📄 AI-Powered Policy Search Bot")
st.write(
    "Ask questions about company policies and get clear answers sourced directly from internal documents."
)


# -----------------------
# Load LLM & embeddings
# -----------------------
@st.cache_resource
def load_models():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return llm, embeddings


llm, embeddings = load_models()


# -----------------------
# Build vector store (cached)
# -----------------------
@st.cache_resource
def build_vector_store():
    documents_dir = "documents"

    if not os.path.exists(documents_dir):
        st.error(
            f"❌ Documents directory '{documents_dir}' not found. Please create it and add PDF files."
        )
        return None

    pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))

    if not pdf_files:
        st.warning(
            f"⚠️ No PDF files found in '{documents_dir}'. Please add policy PDFs."
        )
        return None

    progress_bar = st.progress(0)
    status_text = st.empty()

    embedding_dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

    # Use DirectoryLoader for multi-PDF support
    loader = DirectoryLoader(documents_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    status_text.text("Loading PDF documents...")
    progress_bar.progress(20)

    try:
        docs = loader.load()
        status_text.text(f"Loaded {len(docs)} pages from {len(pdf_files)} files.")
        progress_bar.progress(40)
    except Exception as e:
        st.error(f"❌ Error loading PDFs: {str(e)}")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    status_text.text("Splitting documents into chunks...")
    progress_bar.progress(60)

    chunks = splitter.split_documents(docs)
    status_text.text(f"Created {len(chunks)} text chunks.")
    progress_bar.progress(80)

    status_text.text("Generating embeddings and indexing...")
    vector_store.add_documents(chunks)
    progress_bar.progress(100)
    status_text.text("✅ Vector store ready!")

    progress_bar.empty()
    status_text.empty()

    return vector_store


# -----------------------
# Sidebar for reload
# -----------------------
with st.sidebar:
    if st.button("🔄 Reload Documents", type="primary"):
        st.cache_resource.clear()
        st.success("Cache cleared! Vector store will rebuild on next query.")
        st.rerun()

# Show document status
documents_dir = "documents"
pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))
st.info(
    f"📁 Loaded {len(pdf_files)} PDF{'s' if len(pdf_files) != 1 else ''} from '{documents_dir}' folder."
)

vector_store = build_vector_store()

# -----------------------
# Query input
# -----------------------
query = st.text_input(
    "Ask a question about company policies:",
    placeholder="e.g. How many days of annual leave do I get?",
)

# -----------------------
# RAG pipeline
# -----------------------
if query and vector_store:
    with st.spinner("🔍 Searching policy documents..."):
        retrieved_docs = vector_store.similarity_search(query, k=3)

        if not retrieved_docs:
            st.warning("I couldn’t find this information in the available documents.")
        else:
            context = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})\n{doc.page_content}"
                for doc in retrieved_docs
            )

            prompt = f"""You are an internal HR policy assistant.

Answer the user's question using ONLY the information provided in the context.
If the answer is not present, say:
"I couldn’t find this information in the available documents."
Always cite the source file and page after relevant facts.

Context:
{context}

Question:
{query}

Answer:"""

            response = llm.invoke(prompt)

            st.subheader("✅ Answer")
            st.write(response.content)

            with st.expander(
                f"📚 Sources ({len(retrieved_docs)} documents)", expanded=False
            ):
                for i, doc in enumerate(retrieved_docs, 1):
                    with st.container():
                        st.markdown(
                            f"**Source {i}:** `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`"
                        )
                        st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                        st.markdown("**Content:**")
                        st.write(
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content
                        )

elif query and not vector_store:
    st.error("❌ No documents loaded. Please check the 'documents' folder and reload.")
