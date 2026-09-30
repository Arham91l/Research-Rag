import torch  # noqa: F401  — must import before chromadb/onnxruntime on Windows to avoid a DLL load-order conflict

import base64
import io
import os
import threading

from dotenv import load_dotenv
import requests
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads .env in the working directory into os.environ

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
UNSTRUCTURED_API_KEY = os.environ["UNSTRUCTURED_API_KEY"]

# ============================================================
# LAZY SINGLETONS (replaces st.cache_resource)
# ============================================================

_embeddings = None
_llm = None
_qwen_model = None
_qwen_processor = None
_qwen_lock = threading.Lock()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=GROQ_API_KEY,
            temperature=0,
        )
    return _llm


def get_qwen():
    """Loads once, protected by a lock so two concurrent uploads don't double-load the model."""
    global _qwen_model, _qwen_processor
    with _qwen_lock:
        if _qwen_model is None:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            _qwen_processor = AutoProcessor.from_pretrained(model_name)
            _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
    return _qwen_model, _qwen_processor


# ============================================================
# FIGURE PROMPT
# ============================================================

FIGURE_PROMPT = """
You are analyzing a figure from a research paper.

Analyze the actual visual content of the image carefully.

Describe the figure in a way that allows a text-based RAG system
to answer questions about this figure later.

Identify:
1. Figure type.
2. Panel labels if present.
3. Curves, distributions, plots, diagrams or objects.
4. Visible labels and mathematical notation.
5. Axes and their labels if present.
6. Arrows and their directions.
7. Important relationships between components.
8. What changes between different panels.
9. Any visible equations or symbols.
10. The overall meaning of the figure.

IMPORTANT:
- Describe what is actually visible.
- Do not invent information.
- If something is unreadable, say "unclear".
- Be specific rather than giving a generic description.
- Preserve mathematical notation where possible.

Return a detailed but factual description.
"""


def analyze_image_with_qwen(image_base64, model, processor):
    if not image_base64:
        return "No image data available."

    try:
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": FIGURE_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=700)

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    except Exception as e:
        return f"Figure analysis failed: {str(e)}"


# ============================================================
# UNSTRUCTURED API
# ============================================================

def extract_pdf_elements(pdf_path):
    url = "https://api.unstructuredapp.io/general/v0/general"
    headers = {"unstructured-api-key": UNSTRUCTURED_API_KEY}

    with open(pdf_path, "rb") as f:
        files = {"files": f}
        data = {
            "strategy": "hi_res",
            "extract_image_block_types": '["Image", "Table"]',
            "extract_image_block_to_payload": "true",
            "infer_table_structure": "true",
        }
        response = requests.post(url, headers=headers, files=files, data=data, timeout=300)

    response.raise_for_status()
    return response.json()


# ============================================================
# GROUP ELEMENTS INTO SECTIONS
# ============================================================

def group_into_sections(elements):
    sections = {}
    current_section = "Front Matter"

    for element in elements:
        category = element.get("type", "")
        text = (element.get("text") or "").strip()
        if not text:
            continue

        metadata = element.get("metadata", {})
        page = metadata.get("page_number")

        if category == "Title":
            current_section = text
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, [])
            sections[current_section].append(
                {"category": category, "text": text, "metadata": metadata, "page": page}
            )

    return sections


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents(sections, text_splitter, qwen_model, qwen_processor, progress_cb=None):
    """progress_cb(done, total) is called after each figure is captioned, if provided."""
    final_documents = []
    image_elements = []
    caption_documents = []

    for section_name, section_elements in sections.items():
        text_buffer, text_pages, text_sources = [], [], []

        def flush_text_buffer():
            if not text_buffer:
                return
            combined_text = "\n\n".join(text_buffer)
            pages = [p for p in text_pages if p is not None]
            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None
            source = next((s for s in text_sources if s is not None), None)

            chunks = text_splitter.create_documents(
                [combined_text],
                metadatas=[
                    {
                        "type": "text",
                        "section": section_name,
                        "page": page_start,
                        "page_end": page_end,
                        "source": source,
                    }
                ],
            )
            final_documents.extend(chunks)
            text_buffer.clear()
            text_pages.clear()
            text_sources.clear()

        for element in section_elements:
            category = element["category"]
            text = element["text"].strip()
            metadata = element.get("metadata", {})
            page = metadata.get("page_number")
            source = metadata.get("filename")

            if category in ["NarrativeText", "Text", "ListItem"]:
                if text:
                    text_buffer.append(text)
                    text_pages.append(page)
                    text_sources.append(source)

            elif category == "FigureCaption":
                flush_text_buffer()
                if text:
                    caption_documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "type": "figure_caption",
                                "section": section_name,
                                "page": page,
                                "source": source,
                            },
                        )
                    )

            elif category == "Table":
                flush_text_buffer()
                table_html = metadata.get("text_as_html")
                table_content = table_html if table_html else text
                if table_content:
                    final_documents.append(
                        Document(
                            page_content=table_content,
                            metadata={
                                "type": "table",
                                "section": section_name,
                                "page": page,
                                "source": source,
                            },
                        )
                    )

            elif category == "Image":
                flush_text_buffer()
                image_elements.append(
                    {
                        "section": section_name,
                        "page": page,
                        "source": source,
                        "image_base64": metadata.get("image_base64"),
                    }
                )

        flush_text_buffer()

    # Qwen figure captioning
    qwen_documents = []
    total_images = len(image_elements)
    for index, image_data in enumerate(image_elements):
        description = analyze_image_with_qwen(image_data["image_base64"], qwen_model, qwen_processor)
        qwen_documents.append(
            Document(
                page_content=description,
                metadata={
                    "type": "figure_description",
                    "section": image_data["section"],
                    "page": image_data["page"],
                    "source": image_data["source"],
                    "image_base64": image_data["image_base64"],
                },
            )
        )
        if progress_cb:
            progress_cb(index + 1, total_images)

    return final_documents + caption_documents + qwen_documents


# ============================================================
# CHROMA + BM25
# ============================================================

def create_vectorstore(documents, embeddings, chroma_path):
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="research_paper",
        persist_directory=chroma_path,
        collection_metadata={"hnsw:space": "cosine"},
    )


def create_bm25(documents):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 15
    return retriever


# ============================================================
# HYBRID RETRIEVAL + RRF
# ============================================================

def hybrid_retrieve(query, bm25_retriever, vectorstore, k=8, retrieval_k=15):
    bm25_retriever.k = retrieval_k
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vectorstore.similarity_search(query, k=retrieval_k)

    rrf_k = 60
    rrf_scores, documents = {}, {}

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = hash(doc.page_content)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        documents[doc_id] = doc

    for rank, doc in enumerate(vector_docs, start=1):
        doc_id = hash(doc.page_content)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        documents[doc_id] = doc

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [documents[doc_id] for doc_id in ranked_ids]


# ============================================================
# RAG PROMPT + ANSWERING
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a research assistant answering questions about a research paper.

Answer the user's question using ONLY the retrieved context.

Rules:
1. Use the retrieved context as the primary source of truth.
2. Do not invent facts that are not supported by the retrieved context.
3. If the context is insufficient, say:
   "The retrieved context does not provide enough information to answer this."
4. Explain technical concepts clearly.
5. Preserve equations, mathematical notation, variables and terminology when relevant.
6. If information comes from a figure or table, explicitly mention it.
7. Combine multiple retrieved sources when necessary.
8. Do not mention BM25, vector search, embeddings, RRF or the internal retrieval
   process unless explicitly asked.
9. Do not use outside knowledge to fill gaps.

Retrieved Context:
------------------
{context}
------------------

Question:
{question}

Answer:
""")


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        parts.append(
            f"SOURCE {i}\n"
            f"Type: {md.get('type', 'unknown')}\n"
            f"Section: {md.get('section', 'unknown')}\n"
            f"Page: {md.get('page', 'unknown')}\n"
            f"Page End: {md.get('page_end', 'unknown')}\n"
            f"Source: {md.get('source', 'unknown')}\n\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def answer_question(question, bm25_retriever, vectorstore, llm):
    docs = hybrid_retrieve(question, bm25_retriever, vectorstore, k=8, retrieval_k=15)
    context = format_context(docs)
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    if isinstance(response.content, list):
        answer = "\n".join(
            item["text"] for item in response.content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        answer = response.content

    return answer, docs


def make_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
