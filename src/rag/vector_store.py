# -*- coding: utf-8 -*-
"""RAG 模块 — ChromaDB 向量存储管理。"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils.config import PROJECT_ROOT

# 向量库持久化路径
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"


def get_vector_store(embeddings) -> Chroma:
    """获取 Chroma 向量存储实例（自动加载已有数据）。"""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )


def add_documents(docs: list[Document], embeddings) -> int:
    """将文档列表写入向量存储，返回写入条数。"""
    store = get_vector_store(embeddings)
    ids = store.add_documents(docs)
    return len(ids)


def clear_store(embeddings) -> None:
    """清空向量库（重建时使用）。"""
    import shutil
    # Chroma 1.5+ 默认禁用 reset()，直接删除持久化目录重建
    if CHROMA_PERSIST_DIR.exists():
        shutil.rmtree(str(CHROMA_PERSIST_DIR), ignore_errors=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
