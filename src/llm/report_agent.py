# -*- coding: utf-8 -*-
"""
报告生成 Agent（LangChain 1.x create_agent + Tool Calling）。

流程：获取检测摘要 → 生成基础 Word → 撰写并写入 AI 分析章节。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.llm.llm_config import LLMConfig, get_llm_config, is_llm_configured
from src.llm.prompts import REPORT_AGENT_SYSTEM_PROMPT, REPORT_AGENT_USER_PROMPT
from src.llm.tools import ReportAgentContext, build_report_tools
from src.report.ai_report_generator import build_agent_result_payload
from src.utils.config import PROJECT_ROOT


class ReportAgentError(Exception):
    """Agent 运行错误。"""


def _create_llm(config: LLMConfig) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def _extract_agent_output(response: dict[str, Any]) -> str:
    """从 LangGraph Agent 返回的 messages 中提取最终文本。"""
    messages = response.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                text = "\n".join(p for p in parts if p)
                if text.strip():
                    return text
    return str(response)


def create_report_agent_graph(
    result_dict: dict[str, Any],
    config: LLMConfig | None = None,
):
    """创建 LangChain 1.x Agent 图与共享上下文。"""
    cfg = config or get_llm_config()
    if not is_llm_configured(cfg.api_key):
        raise ReportAgentError(
            "未配置 LLM API Key。请在项目根目录创建 .env 并设置 LLM_API_KEY，"
            "或在 Streamlit 侧边栏填写 API Key。"
        )

    ctx = ReportAgentContext(result_dict=dict(result_dict))
    tools = build_report_tools(ctx)
    llm = _create_llm(cfg)

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=REPORT_AGENT_SYSTEM_PROMPT,
    )
    return graph, ctx


def _image_name_from_result(result_dict: dict[str, Any]) -> str:
    p = Path(result_dict.get("original_path", "unknown.png"))
    return p.name


def run_report_agent(
    result_dict: dict[str, Any],
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    运行报告生成 Agent。

    Returns
    -------
    dict : report_path, agent_output, agent_notes, mode
    """
    cfg = get_llm_config(api_key=api_key, base_url=base_url, model_name=model_name)
    graph, ctx = create_report_agent_graph(result_dict, config=cfg)

    user_input = REPORT_AGENT_USER_PROMPT.format(
        image_name=_image_name_from_result(result_dict),
        detect_time=result_dict.get("detect_time", "未知"),
        model_name=result_dict.get("model_name", "PV-S3"),
        model_version=result_dict.get("model_version", ""),
        confidence_threshold=result_dict.get("confidence_threshold", "N/A"),
    )

    try:
        invoke_kwargs: dict[str, Any] = {}
        if verbose:
            invoke_kwargs["config"] = {"recursion_limit": 25}
        response = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            **invoke_kwargs,
        )
    except Exception as exc:
        raise ReportAgentError(f"Agent 调用失败: {exc}") from exc

    agent_output = _extract_agent_output(response)

    report_path = ctx.report_path
    if not report_path:
        raise ReportAgentError(
            "Agent 未完成报告生成。请检查 API Key、网络与模型名称是否正确。"
            f"\nAgent 输出: {agent_output}"
        )

    try:
        rel = str(Path(report_path).relative_to(PROJECT_ROOT))
    except ValueError:
        rel = report_path

    payload = build_agent_result_payload(report_path, agent_output, ctx.agent_notes)
    payload["report_path_relative"] = rel
    return payload


def run_report_agent_safe(
    result_dict: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """包装 run_report_agent，统一错误结构。"""
    try:
        return {"success": True, **run_report_agent(result_dict, **kwargs)}
    except ReportAgentError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"未知错误: {e}"}
