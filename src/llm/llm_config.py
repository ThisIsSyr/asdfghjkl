# -*- coding: utf-8 -*-
"""
大模型 API 配置。

使用方式（任选其一）：
1. 复制 .env.example 为 .env，填入 LLM_API_KEY 等；
2. 在 Streamlit 侧边栏临时填写（仅当前会话）；
3. 直接设置系统环境变量。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils.config import PROJECT_ROOT

# 加载项目根目录 .env
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120


# ---------------------------------------------------------------------------
# 在此修改默认 API 地址与模型名（也可通过 .env 覆盖）
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"  # DeepSeek OpenAI 兼容接口
DEFAULT_MODEL = "deepseek-chat"

# 其他常用预设（取消注释并写入 .env 即可切换）
# OpenAI:     LLM_BASE_URL=https://api.openai.com/v1          LLM_MODEL=gpt-4o-mini
# 通义千问:   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-plus
# 智谱:       LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/            LLM_MODEL=glm-4-flash


def get_llm_config(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> LLMConfig:
    """合并环境变量、.env 与 Streamlit 会话传入参数。"""
    key = (api_key or os.getenv("LLM_API_KEY", "")).strip()
    url = (base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)).strip().rstrip("/")
    model = (model_name or os.getenv("LLM_MODEL", DEFAULT_MODEL)).strip()
    temp = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tok = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    timeout = int(os.getenv("LLM_TIMEOUT", "120"))
    return LLMConfig(
        api_key=key,
        base_url=url,
        model_name=model,
        temperature=temp,
        max_tokens=max_tok,
        timeout=timeout,
    )


def is_llm_configured(api_key: str | None = None) -> bool:
    cfg = get_llm_config(api_key=api_key)
    return bool(cfg.api_key) and cfg.api_key not in ("", "sk-your-key-here", "your-api-key-here")


def mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return "（未配置）"
    return f"{key[:4]}...{key[-4:]}"
