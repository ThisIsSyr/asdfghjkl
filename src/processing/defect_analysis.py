# -*- coding: utf-8 -*-
"""缺陷 mask 量化分析与严重程度判定。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.utils.config import (
    FALLBACK_MEAN_CONFIDENCE,
    SEVERITY_LIGHT_MAX,
    SEVERITY_MEDIUM_MAX,
    SEVERITY_NORMAL_MAX,
)


@dataclass
class DefectStats:
    defect_area: int
    defect_area_ratio: float
    connected_components: int
    max_defect_area: int
    mean_confidence: float
    severity_level: str
    suggestion: str
    bboxes: list[tuple[int, int, int, int]]  # (x, y, w, h) per component


def _suggestion_for_severity(level: str) -> str:
    m = {
        "正常": "未发现明显缺陷，建议按常规周期巡检。",
        "轻微": "存在轻微缺陷，建议纳入后续巡检重点观察。",
        "中等": "存在较明显缺陷，建议安排人工复核并评估发电效率影响。",
        "严重": "存在严重缺陷，建议尽快进行人工检测、维修或更换组件。",
    }
    return m.get(level, m["正常"])


def base_severity_from_ratio(ratio: float) -> str:
    if ratio < SEVERITY_NORMAL_MAX:
        return "正常"
    if ratio < SEVERITY_LIGHT_MAX:
        return "轻微"
    if ratio < SEVERITY_MEDIUM_MAX:
        return "中等"
    return "严重"


def adjust_severity(
    base: str,
    num_regions: int,
    max_region_ratio: float,
    defect_ratio: float,
) -> str:
    """
    结合连通域数量与最大连通块占比上调严重等级（最多上调一级）。
    """
    order = ["正常", "轻微", "中等", "严重"]
    idx = order.index(base)

    # 碎片缺陷很多：区域数 ≥ 12 且占比 ≥ 0.5%
    if num_regions >= 12 and defect_ratio >= 0.005 and idx < 3:
        idx = min(idx + 1, 3)

    # 最大单块占比很高：单块占全图 ≥ 4%
    if max_region_ratio >= 0.04 and idx < 3:
        idx = min(idx + 1, 3)

    return order[idx]


def analyze_binary_mask(
    mask_uint8: np.ndarray,
    mean_confidence: float | None = None,
) -> DefectStats:
    """
    mask_uint8: 单通道 0/255，与图像同尺寸。
    """
    if mask_uint8.ndim != 2:
        raise ValueError("mask 应为单通道二维数组")

    h, w = mask_uint8.shape[:2]
    total = float(h * w)
    bin_mask = (mask_uint8 > 127).astype(np.uint8)
    defect_area = int(bin_mask.sum())
    defect_ratio = defect_area / total if total > 0 else 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # label 0 为背景
    regions = num_labels - 1
    max_defect_area = 0
    bboxes: list[tuple[int, int, int, int]] = []

    if regions > 0:
        areas = []
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            areas.append(area)
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            bboxes.append((x, y, bw, bh))
        max_defect_area = max(areas)
    else:
        regions = 0

    max_region_ratio = (max_defect_area / total) if total > 0 else 0.0
    base = base_severity_from_ratio(defect_ratio)
    severity = adjust_severity(base, regions, max_region_ratio, defect_ratio)

    conf = mean_confidence if mean_confidence is not None else FALLBACK_MEAN_CONFIDENCE
    if defect_area == 0:
        severity = "正常"
        conf = max(conf, 0.9)

    return DefectStats(
        defect_area=defect_area,
        defect_area_ratio=defect_ratio,
        connected_components=regions,
        max_defect_area=max_defect_area,
        mean_confidence=float(conf),
        severity_level=severity,
        suggestion=_suggestion_for_severity(severity),
        bboxes=bboxes,
    )


def stats_to_result_dict(stats: DefectStats, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "defect_area": stats.defect_area,
        "defect_area_ratio": stats.defect_area_ratio,
        "connected_components": stats.connected_components,
        "max_defect_area": stats.max_defect_area,
        "confidence_score": stats.mean_confidence,
        "severity_level": stats.severity_level,
        "suggestion": stats.suggestion,
        "bboxes": stats.bboxes,
    }
    if extra:
        out.update(extra)
    return out
