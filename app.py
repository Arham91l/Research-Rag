import os
import tempfile
import threading
import uuid

from flask import Flask, jsonify, render_template, request, session

import pipeline
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# ============================================================
# SESSION-SCOPED STATE
# ============================================================
# Flask has no per-browser memory like st.session_state, so each visitor
# gets a random session_id (stored in their cookie) that keys into this
# in-memory dict. Fine for a single-instance deployment / demo; for
# multi-worker production you'd move this to Redis instead.

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def get_session_state():
    sid = get_session_id()
    with SESSIONS_LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {
                "file_name": None,
                "num_documents": 0,
                "vectorstore": None,
                "bm25_retriever": None,
                "history": [],
                "status": "idle",  # idle | processing | ready | error
                "status_message": "",
            }
        return SESSIONS[sid]


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    state = get_session_state()
    return jsonify(
        status=state["status"],
        message=state["status_message"],
        file_name=state["file_name"],
        num_documents=state["num_documents"],
    )


@app.route("/upload", methods=["POST"])
def upload():
    state = get_session_state()

    if "file" not in request.files:
        return jsonify(error="No file provided"), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="No file selected"), 400

    state["status"] = "processing"
    state["status_message"] = "Saving upload..."
    state["history"] = []

    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file.filename)
    file.save(pdf_path)

    try:
        state["status_message"] = "Extracting PDF elements..."
        elements = pipeline.extract_pdf_elements(pdf_path)

        state["status_message"] = f"Organizing {len(elements)} elements into sections..."
        sections = pipeline.group_into_sections(elements)

        state["status_message"] = "Loading Qwen2-VL for figure analysis..."
        qwen_model, qwen_processor = pipeline.get_qwen()

        state["status_message"] = "Creating text, table and figure documents..."
        text_splitter = pipeline.make_text_splitter()

        def progress_cb(done, total):
            state["status_message"] = f"Analyzing figure {done}/{total}..."

        all_documents = pipeline.create_documents(
            sections=sections,
            text_splitter=text_splitter,
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
            progress_cb=progress_cb,
        )

        state["status_message"] = "Building Chroma vector database..."
        embeddings = pipeline.get_embeddings()
        chroma_path = os.path.join(temp_dir, "chroma_db")
        vectorstore = pipeline.create_vectorstore(all_documents, embeddings, chroma_path)

        state["status_message"] = "Building BM25 retriever..."
        bm25_retriever = pipeline.create_bm25(all_documents)

        state["vectorstore"] = vectorstore
        state["bm25_retriever"] = bm25_retriever
        state["file_name"] = file.filename
        state["num_documents"] = len(all_documents)
        state["status"] = "ready"
        state["status_message"] = "Paper processed successfully."

        return jsonify(status="ready", file_name=file.filename, num_documents=len(all_documents))

    except Exception as e:
        state["status"] = "error"
        state["status_message"] = str(e)
        return jsonify(error=str(e)), 500


@app.route("/ask", methods=["POST"])
def ask():
    state = get_session_state()

    if state["status"] != "ready":
        return jsonify(error="No paper indexed yet. Upload a PDF first."), 400

    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Empty question"), 400

    llm = pipeline.get_llm()
    answer, docs = pipeline.answer_question(
        question=question,
        bm25_retriever=state["bm25_retriever"],
        vectorstore=state["vectorstore"],
        llm=llm,
    )

    sources = []
    for doc in docs:
        md = doc.metadata
        sources.append(
            {
                "type": md.get("type", "unknown"),
                "section": md.get("section", "unknown"),
                "page": md.get("page", "unknown"),
                "content": doc.page_content[:1000],
                "image_base64": md.get("image_base64"),
            }
        )

    state["history"].append({"question": question, "answer": answer})

    return jsonify(answer=answer, sources=sources)


@app.route("/history")
def history():
    state = get_session_state()
    return jsonify(history=state["history"])


@app.route("/reset", methods=["POST"])
def reset():
    sid = get_session_id()
    with SESSIONS_LOCK:
        SESSIONS.pop(sid, None)
    return jsonify(status="reset")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
