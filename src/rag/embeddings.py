# -*- coding: utf-8 -*-
"""RAG 模块 — 文本向量化。
使用 sklearn TF-IDF，纯本地、零下载、无需 API。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import PROJECT_ROOT

VECTORIZER_PATH = PROJECT_ROOT / "data" / "chroma_db" / "tfidf_vectorizer.pkl"


class TfidfEmbeddings:
    """兼容 LangChain embedding 接口的 TF-IDF 向量化器。"""

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self._vectorizer = vectorizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not fitted")
        vectors = self._vectorizer.transform(texts).toarray()
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """在所有文档上拟合 TF-IDF 并持久化。"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        analyzer="char_wb",
        ngram_range=(2, 4),
    )
    vectorizer.fit(texts)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    return vectorizer


def load_vectorizer() -> TfidfVectorizer | None:
    """加载已持久化的 TF-IDF 向量化器。"""
    if VECTORIZER_PATH.is_file():
        with open(VECTORIZER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def create_embeddings(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> TfidfEmbeddings:
    """创建 TF-IDF Embeddings 实例（不依赖 API）。"""
    vec = load_vectorizer()
    return TfidfEmbeddings(vectorizer=vec)
