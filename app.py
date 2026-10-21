import base64
import tempfile

import streamlit as st

import pipeline

st.set_page_config(page_title="Research Paper Intelligence System", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return pipeline.get_embeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return pipeline.get_llm()


@st.cache_resource(show_spinner=False)
def get_qwen():
    return pipeline.get_qwen()


# ---------------- Session state ----------------

for key, default in {
    "file_name": None,
    "num_documents": 0,
    "vectorstore": None,
    "bm25_retriever": None,
    "history": [],
    "status": "idle",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def process_pdf(uploaded_file):
    st.session_state.history = []

    temp_dir = tempfile.mkdtemp()
    pdf_path = f"{temp_dir}/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.status("Processing paper...", expanded=True) as status_box:
        status_box.write("Extracting PDF elements...")
        elements = pipeline.extract_pdf_elements(pdf_path)

        status_box.write(f"Organizing {len(elements)} elements into sections...")
        sections = pipeline.group_into_sections(elements)

        status_box.write("Loading Qwen2-VL for figure analysis...")
        qwen_model, qwen_processor = get_qwen()

        status_box.write("Creating text, table and figure documents...")
        text_splitter = pipeline.make_text_splitter()

        def progress_cb(done, total):
            status_box.write(f"Analyzing figure {done}/{total}...")

        all_documents = pipeline.create_documents(
            sections=sections,
            text_splitter=text_splitter,
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
            progress_cb=progress_cb,
        )

        status_box.write("Building Chroma vector database...")
        embeddings = get_embeddings()
        chroma_path = f"{temp_dir}/chroma_db"
        vectorstore = pipeline.create_vectorstore(all_documents, embeddings, chroma_path)

        status_box.write("Building BM25 retriever...")
        bm25_retriever = pipeline.create_bm25(all_documents)

        st.session_state.vectorstore = vectorstore
        st.session_state.bm25_retriever = bm25_retriever
        st.session_state.file_name = uploaded_file.name
        st.session_state.num_documents = len(all_documents)
        st.session_state.status = "ready"

        status_box.update(label="Paper processed successfully.", state="complete", expanded=False)


# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("📚 Research Paper")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Upload & Index"):
        try:
            process_pdf(uploaded_file)
        except Exception as e:
            st.session_state.status = "error"
            st.error(f"Error: {e}")

    if st.session_state.status == "ready":
        st.success(f"Indexed **{st.session_state.file_name}**\n\n{st.session_state.num_documents} documents indexed.")

    if st.button("Reset"):
        for key in ["file_name", "num_documents", "vectorstore", "bm25_retriever", "history", "status"]:
            st.session_state[key] = [] if key == "history" else (0 if key == "num_documents" else None)
        st.session_state.status = "idle"
        st.rerun()


# ---------------- Main chat UI ----------------

st.title("📄 Research Paper Intelligence System")
st.caption("Upload a research paper and ask questions about its text, tables and figures.")

if st.session_state.status != "ready":
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

question = st.chat_input("Ask a question about this paper...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            llm = get_llm()
            answer, docs = pipeline.answer_question(
                question=question,
                bm25_retriever=st.session_state.bm25_retriever,
                vectorstore=st.session_state.vectorstore,
                llm=llm,
            )
        st.write(answer)

        with st.expander(f"Sources ({len(docs)})"):
            for i, doc in enumerate(docs, start=1):
                md = doc.metadata
                st.markdown(
                    f"**Source {i}** — `{md.get('type', 'unknown')}` · "
                    f"section: {md.get('section', 'unknown')} · page: {md.get('page', 'unknown')}"
                )
                if md.get("image_base64"):
                    st.image(base64.b64decode(md["image_base64"]), width=300)
                st.text(doc.page_content[:500])
                st.divider()

    st.session_state.history.append({"question": question, "answer": answer})