# -*- coding: utf-8 -*-
"""项目路径与全局配置。"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录（pv_s3_defect_detection_system/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 目录
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
COLORIZED_DIR = OUTPUT_DIR / "colorized"
CSV_DIR = OUTPUT_DIR / "csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

DB_PATH = DATA_DIR / "detection.db"

# PV-S3 模型权重
MODEL_WEIGHT_PATH = PROJECT_ROOT / "weights" / "pv_s3_best.pth"
MODEL_CONFIG_PATH = PROJECT_ROOT / "weights" / "config.yaml"
PRETRAINED_RESNET_DIR = PROJECT_ROOT / "weights" / "pretrained"

# 推理元信息
MODEL_NAME = "PV-S3"
MODEL_VERSION = "2614-e30"

# 默认置信度阈值（缺陷判定：argmax ≠ 0 且 max_prob ≥ 此值）
DEFAULT_CONFIDENCE_THRESHOLD = 0.90

# fallback 平均置信度
FALLBACK_MEAN_CONFIDENCE = 0.72

# 严重程度阈值（缺陷面积占比）
SEVERITY_NORMAL_MAX = 0.01
SEVERITY_LIGHT_MAX = 0.03
SEVERITY_MEDIUM_MAX = 0.08

# 允许的图片扩展名
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def ensure_directories() -> None:
    """创建项目所需目录。"""
    for d in (
        UPLOAD_DIR,
        MASK_DIR,
        OVERLAY_DIR,
        HEATMAP_DIR,
        COLORIZED_DIR,
        CSV_DIR,
        REPORTS_DIR,
        DATA_DIR,
        DOCS_DIR,
        SAMPLE_IMAGES_DIR,
        PROJECT_ROOT / "weights",
    ):
        d.mkdir(parents=True, exist_ok=True)


def get_relative_to_project(path: Path | str) -> str:
    """返回相对于项目根的路径字符串（用于库存储）。"""
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
