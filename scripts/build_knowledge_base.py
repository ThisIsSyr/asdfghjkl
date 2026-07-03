# -*- coding: utf-8 -*-
"""知识库灌入脚本。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _extract_pdf_text(pdf_path):
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] PDF {pdf_path.name}: {e}")
        return ""


def load_knowledge_docs():
    from langchain_core.documents import Document
    docs = []
    # Markdown files
    for md_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": md_path.name, "file_hash": _file_hash(md_path)},
        ))
    # PDF files
    for pdf_path in sorted(KNOWLEDGE_DIR.glob("*.pdf")):
        text = _extract_pdf_text(pdf_path)
        if not text.strip():
            print(f"[WARN] PDF {pdf_path.name}: no text extracted, skipped")
            continue
        # Paste PDF as one Document; the splitter will chunk it later
        docs.append(Document(
            page_content=text,
            metadata={"source": pdf_path.name, "file_hash": _file_hash(pdf_path)},
        ))
        print(f"[INFO] PDF {pdf_path.name}: {len(text)} chars extracted")
    return docs


def build_knowledge_base(force=False, api_key=None, base_url=None):
    from src.rag.embeddings import create_embeddings, fit_vectorizer
    from src.rag.vector_store import add_documents, clear_store
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = load_knowledge_docs()
    if not docs:
        print("[WARN] No .md files in data/knowledge/")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunk_texts = [c.page_content for c in chunks]
    print(f"[INFO] {len(docs)} docs -> {len(chunks)} chunks")

    if force:
        clear_store(None)
        print("[INFO] Old store cleared.")

    # Fit TF-IDF on all chunks (must be after clear, before store creation)
    print("[INFO] Fitting TF-IDF vectorizer...")
    fit_vectorizer(chunk_texts)
    print("[INFO] TF-IDF fitted.")

    embeddings = create_embeddings()

    count = add_documents(chunks, embeddings)
    print(f"[INFO] {count} chunks written to vector store.")
    return count


def check_knowledge_base():
    import sqlite3
    chroma_db = PROJECT_ROOT / "data" / "chroma_db" / "chroma.sqlite3"
    count = 0
    if chroma_db.is_file():
        try:
            conn = sqlite3.connect(str(chroma_db))
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            count = cur.fetchone()[0]
            conn.close()
        except Exception:
            count = 0
    return {"collection_count": count, "knowledge_files": len(list(KNOWLEDGE_DIR.glob("*.md")))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG knowledge base builder")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    if args.check:
        s = check_knowledge_base()
        print(f"Chunks: {s['collection_count']}, Docs: {s['knowledge_files']}")
    else:
        build_knowledge_base(force=args.force, api_key=args.api_key, base_url=args.base_url)