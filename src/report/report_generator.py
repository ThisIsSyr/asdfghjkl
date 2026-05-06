# -*- coding: utf-8 -*-
"""Word 检测报告生成。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.shared import Inches, Pt

from src.utils.config import PROJECT_ROOT, REPORTS_DIR, ensure_directories


def generate_report(result_dict: dict[str, Any]) -> str:
    """
    根据检测结果字典生成 Word 报告。
    文件名：report_图片名_时间.docx
    返回报告绝对路径。
    """
    ensure_directories()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    original_path = Path(result_dict["original_path"])
    if not original_path.is_absolute():
        original_path = PROJECT_ROOT / original_path

    image_name = original_path.name
    stem = original_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"report_{stem}_{ts}.docx"

    doc = Document()
    title = doc.add_heading(
        "基于 PV-S3 半监督语义分割的光伏板 EL 图像缺陷检测报告",
        level=0,
    )
    title.runs[0].font.size = Pt(16)

    doc.add_paragraph(f"项目名称：基于 PV-S3 半监督语义分割的光伏板 EL 图像缺陷检测与智能运维辅助系统")
    doc.add_paragraph(f"检测时间：{result_dict.get('detect_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    doc.add_paragraph(f"图片名称：{image_name}")
    doc.add_paragraph(f"推理模式：{result_dict.get('mode', 'fallback')}（{result_dict.get('model_version', '')}）")

    doc.add_heading("一、图像与分割结果", level=1)

    def add_image_if_exists(rel_key: str, caption: str) -> None:
        rel = result_dict.get(rel_key)
        if not rel:
            return
        p = Path(rel)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            doc.add_paragraph(caption)
            doc.add_picture(str(p), width=Inches(5.5))

    add_image_if_exists("original_path", "原图：")
    add_image_if_exists("mask_path", "缺陷分割 Mask：")
    add_image_if_exists("overlay_path", "原图与 Mask 叠加：")

    doc.add_heading("二、缺陷量化统计", level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "指标"
    hdr[1].text = "数值"

    rows_data = [
        ("缺陷面积（像素）", str(result_dict.get("defect_area", ""))),
        ("缺陷面积占比", f"{float(result_dict.get('defect_area_ratio', 0)) * 100:.4f}%"),
        ("缺陷连通区域数量", str(result_dict.get("connected_components", ""))),
        ("最大缺陷区域面积（像素）", str(result_dict.get("max_defect_area", ""))),
        ("平均置信度", f"{float(result_dict.get('confidence_score', 0)):.4f}"),
        ("严重程度", str(result_dict.get("severity_level", ""))),
        ("缺陷类别（演示）", str(result_dict.get("defect_categories", ""))),
    ]
    for k, v in rows_data:
        row = table.add_row().cells
        row[0].text = k
        row[1].text = v

    doc.add_heading("三、运维建议", level=1)
    doc.add_paragraph(str(result_dict.get("suggestion", "")))

    doc.add_heading("四、检测结论", level=1)
    sev = result_dict.get("severity_level", "正常")
    ratio = float(result_dict.get("defect_area_ratio", 0))
    conclusion = (
        f"综合像素级分割结果，缺陷面积占比为 {ratio * 100:.2f}%，"
        f"严重程度判定为「{sev}」。请结合现场工况参考运维建议执行后续动作。"
    )
    doc.add_paragraph(conclusion)

    doc.save(str(out_path))
    return str(out_path.resolve())
