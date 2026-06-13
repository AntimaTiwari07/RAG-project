import streamlit as st
import tempfile, os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="DocMind — PDF Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "source_mode" not in st.session_state:
    st.session_state.source_mode = None   # "deep_learning" | "custom_pdf"
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None

# ─── Design System ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, .stApp {
    font-family: 'Space Grotesk', sans-serif;
    background: #0a0a0f;
    color: #e8e6f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e30 !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.sidebar-brand {
    background: linear-gradient(135deg, #6c3fff 0%, #a855f7 100%);
    padding: 20px 20px 16px;
    margin: -1px -1px 0;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #fff;
}
.sidebar-brand span {
    display: block;
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.65);
    margin-top: 3px;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown { color: #9895b0 !important; font-size: 13px !important; }

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #c4c0e0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Source Selector Cards ── */
.source-card {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    border: 1.5px solid #1e1e38;
    background: #0f0f20;
    transition: border-color 0.18s, background 0.18s;
}
.source-card:hover { border-color: #5b3fd4; background: #131328; }
.source-card.active {
    border-color: #7c3aed;
    background: #140e2e;
    box-shadow: 0 0 0 1px #7c3aed22;
}
.source-card-icon { font-size: 20px; margin-bottom: 6px; }
.source-card-title {
    font-size: 13px;
    font-weight: 600;
    color: #d4d0ec;
    margin-bottom: 3px;
}
.source-card-desc { font-size: 11px; color: #4a4870; line-height: 1.5; }
.source-card.active .source-card-title { color: #c084fc; }

/* ── Model badge pills ── */
.model-badge {
    display: inline-block;
    background: #151525;
    border: 1px solid #2a2a45;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #a78bfa;
    margin-bottom: 6px;
    width: 100%;
}
.model-badge .badge-label {
    color: #554f7a;
    font-size: 10px;
    display: block;
    margin-bottom: 2px;
}

/* ── File info card ── */
.file-card {
    background: #13132b;
    border: 1px solid #23234a;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 8px;
}
.file-card .file-name {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #a78bfa;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.file-card .file-size { font-size: 11px; color: #554f7a; margin-top: 4px; }

/* ── DB info card ── */
.db-info-card {
    background: #0e1a2e;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 4px;
}
.db-info-card .db-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #60a5fa;
    font-weight: 700;
    margin-bottom: 6px;
}
.db-tag {
    display: inline-block;
    background: #0a1f3a;
    border: 1px solid #1e3d5c;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 10px;
    color: #3b82f6;
    font-family: 'Space Mono', monospace;
    margin: 2px 2px 2px 0;
}

/* ── Main Content ── */
.block-container { max-width: 900px !important; padding: 0 2rem 2rem !important; }

/* ── Hero ── */
.hero-wrapper {
    padding: 48px 0 32px;
    border-bottom: 1px solid #1a1a2e;
    margin-bottom: 32px;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6c3fff;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 48px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #f0eeff;
    margin-bottom: 12px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(90deg, #7c3aed, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: 15px; color: #6b6890; line-height: 1.6; max-width: 520px; }

/* ── Active source pill shown in hero area ── */
.active-source-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #120e28;
    border: 1px solid #3b2d8a;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    color: #a78bfa;
    margin-bottom: 20px;
}
.active-source-pill .dot {
    width: 6px; height: 6px;
    background: #7c3aed;
    border-radius: 50%;
    flex-shrink: 0;
}
.active-source-pill.blue {
    background: #0a1528;
    border-color: #1e3d8a;
    color: #60a5fa;
}
.active-source-pill.blue .dot { background: #3b82f6; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c3fff 0%, #9333ea 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(108, 63, 255, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 32px rgba(108, 63, 255, 0.45) !important;
}
.stButton > button:disabled {
    background: #1a1a2e !important;
    color: #3a3860 !important;
    box-shadow: none !important;
}

/* ── Upload zone ── */
.stFileUploader { background: transparent !important; }
[data-testid="stFileUploader"] > section {
    background: #0d0d20 !important;
    border: 1.5px dashed #2d2d50 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] > section:hover { border-color: #6c3fff !important; }

/* ── Radio buttons ── */
.stRadio > label { color: #9895b0 !important; font-size: 13px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #9895b0 !important; }

/* ── Chat ── */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 12px 0 !important; }
.stChatMessage p { font-size: 15px !important; line-height: 1.7 !important; color: #d4d0ec !important; }

[data-testid="stChatInput"] {
    background: #0f0f1e !important;
    border: 1px solid #252540 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8e6f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3a3860 !important; }

/* ── Expander ── */
.stExpander { background: #0d0d1e !important; border: 1px solid #1e1e38 !important; border-radius: 8px !important; }
.stExpander summary { color: #5a5880 !important; font-size: 12px !important; font-family: 'Space Mono', monospace !important; letter-spacing: 0.05em !important; }
.stExpander > div > div { background: #0a0a18 !important; }

/* ── Alerts ── */
.stAlert { background: #0f0f22 !important; border-radius: 8px !important; border-left: 3px solid #6c3fff !important; color: #9895b0 !important; }

hr { border-color: #1a1a2e !important; margin: 16px 0 !important; }
.stSpinner > div { border-top-color: #7c3aed !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2d2d50; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #6c3fff; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2e2d4a;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #1a1a2e; }

.empty-state { text-align: center; padding: 60px 20px; color: #2e2d4a; }
.empty-state .icon { font-size: 40px; margin-bottom: 16px; }
.empty-state p { font-size: 14px; line-height: 1.6; max-width: 320px; margin: 0 auto; }

.ready-badge {
    display: inline-block;
    background: #0d1a0d;
    border: 1px solid #1a4d1a;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
}
.ready-badge::before { content: '● '; color: #22c55e; font-size: 8px; }

.ready-badge-blue {
    display: inline-block;
    background: #0a1528;
    border: 1px solid #1a3d6a;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #60a5fa;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
}
.ready-badge-blue::before { content: '● '; color: #3b82f6; font-size: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Models ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    embed = MistralAIEmbeddings(model="codestral-embed")
    llm = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a precise document analyst. "
            "Answer only from the provided context. "
            "If the answer isn't in the document, say: "
            "\"I couldn't find that in the document.\""
        )),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    return embed, llm, prompt

embedding_model, llm, prompt_template = get_models()

# ─── Deep Learning DB loader ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_deep_learning_db(_embedding_model):
    """
    Load the pre-built Deep Learning ChromaDB.
    """
    db_path = "./chroma-db"
    if not os.path.exists(db_path):
        return None, 0
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=_embedding_model
    )
    count = vectorstore._collection.count()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    return retriever, count

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        DocMind
        <span>PDF Intelligence Engine</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Knowledge Source Selector ─────────────────────────────────────────────
    st.markdown("**Knowledge Source**")

    dl_active = "active" if st.session_state.source_mode == "deep_learning" else ""
    pdf_active = "active" if st.session_state.source_mode == "custom_pdf" else ""

    col_dl, col_pdf = st.columns(2)

    with col_dl:
        if st.button("🧠 Deep\nLearning DB", use_container_width=True, key="btn_dl"):
            if st.session_state.source_mode != "deep_learning":
                with st.spinner("Loading Deep Learning DB…"):
                    retriever, count = load_deep_learning_db(embedding_model)
                if retriever:
                    st.session_state.retriever = retriever
                    st.session_state.source_mode = "deep_learning"
                    st.session_state.chunk_count = count
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("DB not found. Set DEEP_LEARNING_DB_PATH in .env")

    with col_pdf:
        if st.button("📄 Upload\nmy PDF", use_container_width=True, key="btn_pdf"):
            if st.session_state.source_mode != "custom_pdf":
                st.session_state.source_mode = "custom_pdf"
                st.session_state.retriever = None
                st.session_state.chunk_count = None
                st.session_state.messages = []
                st.rerun()

    # Show active mode card
    if st.session_state.source_mode == "deep_learning":
        st.markdown("""
        <div class="db-info-card">
            <div class="db-title">🧠 Deep Learning Knowledge Base</div>
            <div style="font-size:11px;color:#4a6a8a;line-height:1.5;margin-bottom:8px">
                Pre-indexed corpus covering neural networks, architectures, and training methods.
            </div>
            <span class="db-tag">Neural Networks</span>
            <span class="db-tag">CNNs</span>
            <span class="db-tag">Transformers</span>
            <span class="db-tag">RNNs / LSTMs</span>
            <span class="db-tag">Backprop</span>
            <span class="db-tag">Optimization</span>
            <span class="db-tag">Regularization</span>
            <span class="db-tag">Attention</span>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.source_mode == "custom_pdf":
        st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF upload (only when in custom mode) ─────────────────────────────────
    uploaded_file = None
    if st.session_state.source_mode == "custom_pdf":
        uploaded_file = st.file_uploader(
            "Upload your PDF",
            type=["pdf"],
            help="Drag & drop or click to browse"
        )
        if uploaded_file:
            st.markdown(f"""
            <div class="file-card">
                <div class="file-name">📄 {uploaded_file.name}</div>
                <div class="file-size">{round(uploaded_file.size / 1024, 1)} KB</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stack info ────────────────────────────────────────────────────────────
    st.markdown("**Stack**")
    st.markdown("""
    <div class="model-badge"><span class="badge-label">EMBEDDINGS</span>codestral-embed</div>
    <div class="model-badge"><span class="badge-label">LLM</span>mistral-small-2506</div>
    <div class="model-badge"><span class="badge-label">RETRIEVER</span>MMR · k=4 · λ=0.5</div>
    <div class="model-badge"><span class="badge-label">STORE</span>ChromaDB</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">Retrieval-Augmented Generation</div>
    <div class="hero-title">Ask your<br><em>documents anything</em></div>
    <div class="hero-sub">Choose the Deep Learning knowledge base or upload your own PDF, then ask questions in plain English.</div>
</div>
""", unsafe_allow_html=True)

# ─── Process PDF button (only in custom mode) ─────────────────────────────────
if st.session_state.source_mode == "custom_pdf":
    col_btn, col_status = st.columns([1, 2])

    with col_btn:
        process_clicked = st.button(
            "⚡ Process PDF",
            use_container_width=True,
            disabled=(uploaded_file is None)
        )

    with col_status:
        if st.session_state.retriever and st.session_state.source_mode == "custom_pdf":
            st.markdown(
                f'<div style="padding:10px 0"><span class="ready-badge">READY</span>'
                f'&nbsp;&nbsp;<span style="color:#4a4870;font-size:13px;font-family:\'Space Mono\',monospace">'
                f'{st.session_state.chunk_count} chunks indexed</span></div>',
                unsafe_allow_html=True
            )
        elif uploaded_file is None:
            st.markdown(
                '<div style="padding:10px 0;color:#2e2d4a;font-size:13px;">'
                'Upload a PDF to get started</div>',
                unsafe_allow_html=True
            )

    if process_clicked and uploaded_file:
        with st.spinner("Splitting pages · building embeddings · indexing chunks…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            docs = PyPDFLoader(pdf_path).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
            )

            st.session_state.retriever = retriever
            st.session_state.chunk_count = len(chunks)
            st.session_state.messages = []
            st.success(f"Indexed {len(chunks)} chunks from {len(docs)} pages.")
            st.rerun()

elif st.session_state.source_mode == "deep_learning":
    # Show DB status
    if st.session_state.retriever:
        st.markdown(
            f'<div style="padding:4px 0 20px"><span class="ready-badge-blue">LOADED</span>'
            f'&nbsp;&nbsp;<span style="color:#4a6a8a;font-size:13px;font-family:\'Space Mono\',monospace">'
            f'{st.session_state.chunk_count} chunks · Deep Learning DB</span></div>',
            unsafe_allow_html=True
        )

else:
    # No mode selected yet — prompt
    st.markdown("""
    <div style="background:#0d0d1e;border:1px solid #1e1e38;border-radius:12px;padding:24px 28px;margin-bottom:24px">
        <div style="font-family:'Space Mono',monospace;font-size:11px;letter-spacing:0.14em;
                    text-transform:uppercase;color:#3b3860;margin-bottom:10px">Get started</div>
        <div style="font-size:14px;color:#5a5880;line-height:1.7">
            Choose a knowledge source in the sidebar:<br>
            <span style="color:#7c5fc0">🧠 Deep Learning DB</span> — pre-built index of deep learning concepts.<br>
            <span style="color:#7c5fc0">📄 Upload my PDF</span> — index any document you have.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Chat Area ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.messages:
    if st.session_state.retriever:
        mode_label = "Deep Learning DB" if st.session_state.source_mode == "deep_learning" else "your document"
        st.markdown(f"""
        <div class="empty-state">
            <div class="icon">💬</div>
            <p>Ready to answer from {mode_label}. Ask your first question below.</p>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.source_mode:
        pass  # handled above
else:
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ─── Chat Input ───────────────────────────────────────────────────────────────
dl_placeholder = "Ask anything about deep learning…"
pdf_placeholder = "Ask a question about your PDF…"
default_placeholder = "Choose a knowledge source first…"

if st.session_state.source_mode == "deep_learning":
    placeholder = dl_placeholder
elif st.session_state.source_mode == "custom_pdf":
    placeholder = pdf_placeholder
else:
    placeholder = default_placeholder

query = st.chat_input(placeholder)

if query:
    if st.session_state.retriever is None:
        if st.session_state.source_mode == "custom_pdf":
            st.error("Process a PDF first — upload one in the sidebar and click ⚡ Process PDF.")
        elif st.session_state.source_mode == "deep_learning":
            st.error("Deep Learning DB not found. Check DEEP_LEARNING_DB_PATH in your .env")
        else:
            st.error("Choose a knowledge source in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Reading relevant passages…"):
            retrieved_docs = st.session_state.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            final_prompt = prompt_template.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)
            answer = response.content

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
