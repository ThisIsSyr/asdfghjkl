# -*- coding: utf-8 -*-
"""报告生成 Agent 工具集。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool

from src.report.ai_report_generator import append_ai_sections_to_docx
from src.report.report_generator import generate_report


@dataclass
class ReportAgentContext:
    """Agent 工具共享上下文。"""

    result_dict: dict[str, Any]
    report_path: str | None = None
    agent_notes: list[str] = field(default_factory=list)


def _format_summary(result: dict[str, Any]) -> str:
    """将检测结果格式化为 LLM 易读文本。"""
    per_class = result.get("per_class_areas") or {}
    mean_conf = result.get("mean_confidence_per_class") or {}
    summary = {
        "图片信息": {
            "original_path": result.get("original_path"),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "detect_time": result.get("detect_time"),
        },
        "模型信息": {
            "model_name": result.get("model_name"),
            "model_version": result.get("model_version"),
            "mode": result.get("mode"),
            "confidence_threshold": result.get("confidence_threshold"),
        },
        "缺陷统计": {
            "defect_categories": result.get("defect_categories"),
            "defect_area_px": result.get("defect_area"),
            "defect_area_ratio": result.get("defect_area_ratio"),
            "defect_area_ratio_percent": round(float(result.get("defect_area_ratio", 0)) * 100, 4),
            "connected_components": result.get("connected_components"),
            "max_defect_area_px": result.get("max_defect_area"),
            "confidence_score": result.get("confidence_score"),
            "severity_level": result.get("severity_level"),
            "rule_suggestion": result.get("suggestion"),
        },
        "逐类缺陷像素面积": per_class,
        "逐类平均置信度": mean_conf,
        "bbox_count": len(result.get("bboxes") or []),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def build_report_tools(ctx: ReportAgentContext) -> list:
    """根据当前检测上下文构建 LangChain 工具列表。"""

    @tool
    def get_detection_summary() -> str:
        """获取当前光伏板 EL 检测的完整结构化摘要（缺陷面积、类别、严重度、置信度等）。"""
        return _format_summary(ctx.result_dict)

    @tool
    def generate_base_word_report() -> str:
        """生成包含原图、Mask、热图、统计表的基础 Word 检测报告，返回报告文件路径。"""
        path = generate_report(ctx.result_dict)
        ctx.report_path = path
        ctx.agent_notes.append(f"基础报告已生成: {path}")
        return f"基础 Word 报告已生成，路径: {path}"

    @tool
    def append_ai_analysis(
        executive_summary: str,
        defect_analysis: str,
        maintenance_advice: str,
        risk_assessment: str,
        conclusion: str,
    ) -> str:
        """
        将 AI 撰写的五个分析章节追加到已生成的基础 Word 报告中。
        参数均为中文段落文本。
        """
        if not ctx.report_path:
            path = generate_report(ctx.result_dict)
            ctx.report_path = path

        sections = {
            "执行摘要": executive_summary.strip(),
            "缺陷机理与分布分析": defect_analysis.strip(),
            "运维建议详述": maintenance_advice.strip(),
            "风险评估": risk_assessment.strip(),
            "综合结论": conclusion.strip(),
        }
        append_ai_sections_to_docx(ctx.report_path, sections)
        ctx.agent_notes.append("AI 分析章节已写入报告")
        return f"AI 分析章节已追加到报告: {ctx.report_path}"

    @tool
    def get_report_file_path() -> str:
        """返回当前已生成报告的文件路径；若尚未生成则提示先调用 generate_base_word_report。"""
        if ctx.report_path:
            return ctx.report_path
        return "尚未生成报告，请先调用 generate_base_word_report。"

    return [
        get_detection_summary,
        generate_base_word_report,
        append_ai_analysis,
        get_report_file_path,
    ]
