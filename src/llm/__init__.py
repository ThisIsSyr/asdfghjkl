# -*- coding: utf-8 -*-
from .llm_config import DEFAULT_BASE_URL, DEFAULT_MODEL, get_llm_config, is_llm_configured, mask_api_key

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "get_llm_config",
    "is_llm_configured",
    "mask_api_key",
    "ReportAgentError",
    "run_report_agent",
    "run_report_agent_safe",
]


def __getattr__(name: str):
    """延迟加载 Agent，避免启动时拉取 langchain/transformers。"""
    if name in ("ReportAgentError", "run_report_agent", "run_report_agent_safe"):
        from . import report_agent as _ra

        return getattr(_ra, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
