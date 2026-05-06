# -*- coding: utf-8 -*-
"""
PV-S3 兼容推理封装。

真实权重接入说明：
1. 将官方仓库 https://github.com/abj247/PV-S3 中的权重保存到 weights/pv_s3_best.pth
2. 在 load_model() 中取消注释 PyTorch 加载逻辑，并实现与本文件一致的类别数、输入尺寸。
3. predict() 中返回 softmax 或 logits，postprocess_mask() 已支持单通道缺陷类。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.processing.defect_analysis import analyze_binary_mask
from src.utils.config import (
    FALLBACK_MEAN_CONFIDENCE,
    MASK_DIR,
    MODEL_CONFIG_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_WEIGHT_PATH,
    OVERLAY_DIR,
    ensure_directories,
    get_relative_to_project,
)

# 全局可选：真实模型实例
_MODEL = None
_USE_REAL_MODEL = False


def load_model(weight_path: Path | None = None) -> Any:
    """
    加载 PV-S3 权重；若不存在则进入 fallback 模式。
    接入真实模型时：在此构造网络并 load_state_dict。
    """
    global _MODEL, _USE_REAL_MODEL
    wp = Path(weight_path) if weight_path else MODEL_WEIGHT_PATH
    if wp.is_file():
        try:
            # --- 预留：接入 PV-S3 官方结构与权重 ---
            # import torch
            # from pv_s3_net import build_model
            # net = build_model(...)
            # ckpt = torch.load(wp, map_location="cpu")
            # net.load_state_dict(ckpt.get("model", ckpt))
            # net.eval()
            # _MODEL = net
            # _USE_REAL_MODEL = True
            # return _MODEL

            # 权重文件存在但尚未实现加载逻辑时，仍使用 fallback，避免误加载损坏文件
            _MODEL = None
            _USE_REAL_MODEL = False
        except Exception:
            _MODEL = None
            _USE_REAL_MODEL = False
    else:
        _MODEL = None
        _USE_REAL_MODEL = False
    return _MODEL


def preprocess_image(image_bgr: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """归一化/缩放等预处理；返回 RGB float32 [H,W,3] 供模型使用。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_image 需要 BGR 三通道图")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if target_size is not None:
        tw, th = target_size
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    return rgb


def predict(preprocessed_rgb: np.ndarray) -> np.ndarray:
    """
    返回与输入同尺寸的缺陷概率图或 argmax 类别图。
    真实模型：应返回 [H,W] 或 [H,W,C] 的 numpy。
    """
    if _USE_REAL_MODEL and _MODEL is not None:
        import torch

        with torch.no_grad():
            # x = torch.from_numpy(preprocessed_rgb).permute(2,0,1).unsqueeze(0)
            # logits = _MODEL(x)
            # prob = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
            # return prob
            pass
    return _fallback_predict_from_rgb(preprocessed_rgb)


def _fallback_predict_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """传统图像处理模拟缺陷响应（0~1 浮点）。"""
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = 255 - thr
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.bitwise_or(edges, dark)
    combined = cv2.GaussianBlur(combined, (3, 3), 0)
    prob = combined.astype(np.float32) / 255.0
    prob = np.clip(prob * 1.1, 0.0, 1.0)
    return prob


def postprocess_mask(
    pred: np.ndarray,
    orig_hw: tuple[int, int],
    threshold: float = 0.35,
) -> tuple[np.ndarray, float]:
    """
    将模型输出转为 0/255 单通道 mask，并缩放到原图大小。
    返回 (mask_uint8, mean_confidence)。
    """
    h, w = orig_hw
    if pred.ndim == 3:
        pred = pred.argmax(axis=-1).astype(np.float32)
        pred = pred / max(pred.max(), 1.0)
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_prob = pred.astype(np.float32)
    binary = (mask_prob >= threshold).astype(np.uint8) * 255
    if binary.any():
        mean_conf = float(mask_prob[binary > 0].mean())
    else:
        mean_conf = FALLBACK_MEAN_CONFIDENCE * 0.5
    return binary, mean_conf


def _save_visuals(
    image_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    stem: str,
) -> tuple[str, str]:
    ensure_directories()
    mask_path = MASK_DIR / f"{stem}_mask.png"
    overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
    cv2.imwrite(str(mask_path), mask_uint8)
    overlay = image_bgr.copy()
    color = np.zeros_like(overlay)
    color[:, :, 2] = mask_uint8  # 红色缺陷
    overlay = cv2.addWeighted(overlay, 0.65, color, 0.35, 0)
    cv2.imwrite(str(overlay_path), overlay)
    return get_relative_to_project(mask_path), get_relative_to_project(overlay_path)


def run_inference(image_path: str | Path) -> dict[str, Any]:
    """
    端到端推理。返回字典包含路径、统计与建议。
    """
    load_model()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在: {path}")

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {path}")

    h, w = image_bgr.shape[:2]
    rgb = preprocess_image(image_bgr)
    pred = predict(rgb)
    mask_uint8, mean_conf = postprocess_mask(pred, (h, w))

    stats = analyze_binary_mask(mask_uint8, mean_confidence=mean_conf)

    stem = path.stem
    mask_rel, overlay_rel = _save_visuals(image_bgr, mask_uint8, stem)

    image_rel = get_relative_to_project(path.resolve())

    defect_categories = "裂纹/暗斑/潜在异常区域(演示)"
    if stats.defect_area == 0:
        defect_categories = "无明显缺陷"

    result: dict[str, Any] = {
        "original_path": image_rel,
        "mask_path": mask_rel,
        "overlay_path": overlay_rel,
        "defect_categories": defect_categories,
        "defect_area": stats.defect_area,
        "defect_area_ratio": stats.defect_area_ratio,
        "connected_components": stats.connected_components,
        "max_defect_area": stats.max_defect_area,
        "confidence_score": stats.mean_confidence,
        "severity_level": stats.severity_level,
        "suggestion": stats.suggestion,
        "image_width": w,
        "image_height": h,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION if not _USE_REAL_MODEL else "pv-s3-weight",
        "bboxes": stats.bboxes,
        "mode": "pv-s3" if _USE_REAL_MODEL else "fallback",
    }
    return result


def get_model_status() -> dict[str, Any]:
    """用于界面展示当前是否加载真实权重。"""
    load_model()
    return {
        "weight_exists": MODEL_WEIGHT_PATH.is_file(),
        "config_exists": MODEL_CONFIG_PATH.is_file(),
        "using_real_model": _USE_REAL_MODEL,
        "weight_path": str(MODEL_WEIGHT_PATH),
    }
