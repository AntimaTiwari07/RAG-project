import streamlit as st
import tempfile, os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}
.hero{text-align:center;padding:20px}
.hero h1{
font-size:60px;color:white;
}
.hero p{color:#cbd5e1}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Control Center")

    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"],
        help="Drag & Drop your PDF here"
    )

    st.divider()

    st.subheader("🤖 Models")
    st.info("Embedding: codestral-embed")
    st.info("LLM: mistral-small-2506")
    st.info("Retriever: MMR")

    if uploaded_file:
        st.subheader("📑 PDF Details")
        st.write("**Name:**", uploaded_file.name)
        st.write("**Size:**", round(uploaded_file.size/1024,2), "KB")

st.markdown("""
<div class="hero">
<h1>🤖 AI PDF Assistant</h1>
<p>Upload a PDF and chat with it using Mistral AI + ChromaDB</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_models():
    embed = MistralAIEmbeddings(model="codestral-embed")
    llm = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        ("system","Use only provided context. If answer not found say: I could not find the answer in the document."),
        ("human","Context:{context}\nQuestion:{question}")
    ])
    return embed, llm, prompt

embedding_model, llm, prompt_template = get_models()

if uploaded_file and st.button("🚀 Process PDF", use_container_width=True):

    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        docs = PyPDFLoader(pdf_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":4,
                "fetch_k":10,
                "lambda_mult":0.5
            }
        )

        st.session_state.retriever = retriever
        st.session_state.chunk_count = len(chunks)
        st.success("PDF processed successfully!")

if "chunk_count" in st.session_state:
    c1,c2 = st.columns(2)
    c1.metric("Chunks", st.session_state.chunk_count)
    c2.metric("Status", "Ready")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything about your PDF...")

if query:

    if st.session_state.retriever is None:
        st.error("Please upload and process a PDF first.")
        st.stop()

    st.session_state.messages.append(
        {"role":"user","content":query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):

            docs = st.session_state.retriever.invoke(query)

            context = "\\n\\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt_template.invoke({
                "context":context,
                "question":query
            })

            response = llm.invoke(final_prompt)

            answer = response.content

            st.markdown(answer)

            with st.expander("📄 Retrieved Context"):
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content[:1000])

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )
