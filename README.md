# 🔬 Research Paper Intelligence — Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) system that allows users to upload research papers in PDF format and ask questions about their content.

Unlike a traditional text-only RAG system, this project is designed to handle **research-paper content including text, tables, figures, and diagrams**, making it suitable for scientific and technical documents.



---

## 📌 Overview

Research papers can contain hundreds of pages of information distributed across:

- Text
- Sections
- Tables
- Figures
- Diagrams
- Captions
- Mathematical explanations

Traditional keyword search often fails to retrieve the correct context from such documents.

This project combines **document parsing, embeddings, vector search, keyword search, reciprocal rank fusion, and LLM-based generation** to build a research-paper question-answering system.

### Core Pipeline

```text
User uploads PDF
       ↓
Unstructured API
       ↓
Text + Tables + Figures
       ↓
Figure Analysis
       ↓
Document Creation
       ↓
Text Chunking
       ↓
BGE Embeddings
       ↓
┌───────────────────────┐
│                       │
│   Vector Retrieval    │
│        +              │
│   BM25 Retrieval      │
│                       │
└───────────┬───────────┘
            ↓
      RRF Fusion
            ↓
       Groq LLM
            ↓
    Answer + Sources
