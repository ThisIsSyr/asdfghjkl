# -*- coding: utf-8 -*-
"""RAG 模块 — 检索接口。"""

from __future__ import annotations

from typing import Any

from src.rag.embeddings import create_embeddings
from src.rag.vector_store import get_vector_store

# 默认检索返回条数
DEFAULT_TOP_K = 5


class RAGRetriever:
    """封装检索引擎，供 Agent 工具调用。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._top_k = top_k
        self._embeddings = create_embeddings(api_key=api_key, base_url=base_url)
        self._store = get_vector_store(self._embeddings)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """检索与 query 最相关的知识文档片段。

        Returns
        -------
        list[dict] : 每个元素含 content, source, score 字段
        """
        k = top_k or self._top_k
        try:
            results = self._store.similarity_search_with_score(query, k=k)
        except Exception:
            return []

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知来源"),
                "score": round(score, 4),
            }
            for doc, score in results
        ]

    def search_formatted(self, query: str) -> str:
        """将检索结果格式化为 LLM 易读的文本块。"""
        hits = self.search(query)
        if not hits:
            return "【RAG 检索】未找到相关知识条目。"
        lines = [f"【RAG 检索结果 — top {len(hits)} 条相关知识】"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"\n--- 条目 {i} (来源: {hit['source']}, 相似度: {hit['score']}) ---")
            lines.append(hit["content"])
        return "\n".join(lines)


# 模块级单例缓存，避免重复初始化
_retriever_cache: dict[str, RAGRetriever] = {}


def get_retriever(
    api_key: str | None = None,
    base_url: str | None = None,
) -> RAGRetriever:
    """获取 RAGRetriever 实例（按 api_key 缓存）。"""
    cache_key = api_key or "__default__"
    if cache_key not in _retriever_cache:
        _retriever_cache[cache_key] = RAGRetriever(
            api_key=api_key,
            base_url=base_url,
        )
    return _retriever_cache[cache_key]
