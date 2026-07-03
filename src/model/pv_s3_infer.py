# -*- coding: utf-8 -*-
"""
PV-S3 语义分割推理封装。

使用 Teacher 1 分支（encoder1 + decoder1）进行推理，
通过置信度阈值过滤低置信度像素，输出 4 张图：
  原图 | 分类预测图 | 置信度热图 | 叠加图
"""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.processing.defect_analysis import analyze_binary_mask
from src.processing.image_utils import imread_bgr, imwrite_bgr
from src.utils.config import (
    COLORIZED_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    HEATMAP_DIR,
    MASK_DIR,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_WEIGHT_PATH,
    OVERLAY_DIR,
    PRETRAINED_RESNET_DIR,
    ensure_directories,
    get_relative_to_project,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
NUM_CLASSES = 5

EL_CLASS_NAMES: dict[int, str] = {
    0: "背景",
    1: "裂纹 (Crack)",
    2: "栅线中断 (Interconnect)",
    3: "接触不良 (Contact)",
    4: "腐蚀 (Corrosion)",
}

# 分类预测图调色板：索引 0-4，每类 3 通道 RGB
PV_S3_PALETTE: list[int] = [0] * (NUM_CLASSES * 3)
# 0=背景黑, 1=裂纹红, 2=栅线中断绿, 3=接触不良蓝, 4=腐蚀黄
_preset = [
    (0, 0, 0),
    (220, 20, 60),
    (50, 205, 50),
    (30, 144, 255),
    (255, 215, 0),
]
for i, (r, g, b) in enumerate(_preset):
    PV_S3_PALETTE[i * 3 + 0] = r
    PV_S3_PALETTE[i * 3 + 1] = g
    PV_S3_PALETTE[i * 3 + 2] = b

# ---------------------------------------------------------------------------
# 全局模型
# ---------------------------------------------------------------------------
_MODEL = None
_USE_REAL_MODEL = False


# ===================================================================
# 模型加载
# ===================================================================


def load_model(weight_path: Path | None = None) -> Any:
    """加载 PV-S3 EntireModel 并注入权重。"""
    global _MODEL, _USE_REAL_MODEL

    import torch
    from src.model.pv_s3_net.entire_model import EntireModel

    wp = Path(weight_path) if weight_path else MODEL_WEIGHT_PATH
    if not wp.is_file():
        raise FileNotFoundError(
            f"PV-S3 权重文件不存在: {wp}\n"
            f"请将训练好的 .pth 放到 weights/pv_s3_best.pth"
        )

    config = {"resnet": 50, "semi": True, "data_h_w": [0, 0]}
    model = EntireModel(
        num_classes=NUM_CLASSES,
        config=config,
        sup_loss=None,
        cons_w_unsup=None,
        ignore_index=0,
    )

    ckpt = torch.load(wp, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # 处理 DataParallel 包装：去掉 "module." 前缀
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    _MODEL = model
    _USE_REAL_MODEL = True
    return _MODEL


# ===================================================================
# 预处理
# ===================================================================


def preprocess_image(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """BGR → RGB float32 [0,1]，可选缩放。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_image 需要 BGR 三通道图")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if target_size is not None:
        tw, th = target_size
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    return rgb


# ===================================================================
# 推理核心
# ===================================================================


def predict(preprocessed_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Teacher 1 推理。

    Returns
    -------
    class_map : np.ndarray  [H, W] int (0-4)
    conf_map  : np.ndarray  [H, W] float32 (0-1)
    prob_full : np.ndarray  [5, H, W] float32
    """
    import torch
    import torch.nn.functional as F

    if not _USE_REAL_MODEL or _MODEL is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")

    x = torch.from_numpy(preprocessed_rgb).permute(2, 0, 1).unsqueeze(0)
    device = next(_MODEL.parameters()).device
    x = x.to(device)

    H, W = x.shape[2], x.shape[3]
    up_h, up_w = ceil(H / 8) * 8, ceil(W / 8) * 8
    x = F.interpolate(x, size=(up_h, up_w), mode="bilinear", align_corners=True)

    with torch.no_grad():
        logits = _MODEL.decoder1(
            _MODEL.encoder1(x),
            data_shape=[up_h, up_w],
        )
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
    prob = torch.softmax(logits, dim=1)[0]  # [5, H, W]

    class_map = prob.argmax(dim=0).cpu().numpy().astype(np.uint8)   # [H, W]
    conf_map = prob.max(dim=0).values.cpu().numpy().astype(np.float32)  # [H, W]
    prob_full = prob.cpu().numpy()

    return class_map, conf_map, prob_full


# ===================================================================
# 后处理：置信度阈值 → 二值 mask
# ===================================================================


def class_map_to_binary(
    class_map: np.ndarray,
    conf_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    根据分类图 + 置信度阈值生成二值 mask。

    条件：argmax 类别 ≠ 0（背景）且置信度 ≥ threshold。

    Returns: uint8 [H, W] 0/255
    """
    is_defect = (class_map > 0) & (conf_map >= threshold)
    return is_defect.astype(np.uint8) * 255


def per_class_areas(class_map: np.ndarray, conf_map: np.ndarray, threshold: float) -> dict[str, int]:
    """统计各类别「高置信度缺陷」像素面积。"""
    areas: dict[str, int] = {}
    for cid in range(1, NUM_CLASSES):
        mask = (class_map == cid) & (conf_map >= threshold)
        name = EL_CLASS_NAMES.get(cid, f"class_{cid}")
        areas[name] = int(mask.sum())
    return areas


def per_class_mean_confidence(
    class_map: np.ndarray,
    conf_map: np.ndarray,
) -> dict[str, float]:
    """统计各类别（含背景）的平均置信度。"""
    means: dict[str, float] = {}
    for cid in range(NUM_CLASSES):
        mask = class_map == cid
        name = EL_CLASS_NAMES.get(cid, f"class_{cid}")
        if mask.any():
            means[name] = float(conf_map[mask].mean())
        else:
            means[name] = 0.0
    return means


# ===================================================================
# 可视化保存
# ===================================================================


def _colorize_mask(class_map: np.ndarray, palette: list[int]) -> np.ndarray:
    """将 [H,W] 类别图映射为 RGB uint8 [H,W,3] 彩色图。"""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(NUM_CLASSES):
        r = palette[cid * 3 + 0]
        g = palette[cid * 3 + 1]
        b = palette[cid * 3 + 2]
        mask = class_map == cid
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b
    return rgb


def _save_heatmap(conf_map: np.ndarray, save_path: Path) -> None:
    """保存置信度热图（matplotlib），兼容 Unicode 路径。"""
    from io import BytesIO

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_map, cmap="jet", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Confidence")
    plt.axis("off")
    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(buf.getvalue())


def _save_visuals(
    image_bgr: np.ndarray,
    class_map: np.ndarray,
    conf_map: np.ndarray,
    binary_mask: np.ndarray,
    stem: str,
) -> dict[str, str]:
    """
    保存 4 张输出图。

    Returns: dict with keys original_path, mask_path, colorized_mask_path,
              heatmap_path, overlay_path（均为相对路径字符串）。
    """
    ensure_directories()
    COLORIZED_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 原图
    orig_path = MASK_DIR / f"{stem}_img.png"
    imwrite_bgr(orig_path, image_bgr)

    # 2) 二值 mask（兼容下游）
    mask_path = MASK_DIR / f"{stem}_mask.png"
    imwrite_bgr(mask_path, binary_mask)

    # 3) 分类彩色预测图
    colored = _colorize_mask(class_map, PV_S3_PALETTE)
    colored_path = COLORIZED_DIR / f"{stem}_pred.png"
    imwrite_bgr(colored_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    # 4) 置信度热图
    heatmap_path = HEATMAP_DIR / f"{stem}_conf.png"
    _save_heatmap(conf_map, heatmap_path)

    # 5) 叠加图：分类彩色图 40% 透明度覆盖原图
    overlay_bgr = image_bgr.astype(np.float32)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR).astype(np.float32)
    overlay_blend = cv2.addWeighted(overlay_bgr, 0.6, colored_bgr, 0.4, 0)
    overlay_blend = overlay_blend.clip(0, 255).astype(np.uint8)
    overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
    imwrite_bgr(overlay_path, overlay_blend)

    return {
        "original_path": get_relative_to_project(orig_path),
        "mask_path": get_relative_to_project(mask_path),
        "colorized_mask_path": get_relative_to_project(colored_path),
        "heatmap_path": get_relative_to_project(heatmap_path),
        "overlay_path": get_relative_to_project(overlay_path),
    }


# ===================================================================
# 端到端推理入口
# ===================================================================


def run_inference(
    image_path: str | Path,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """
    端到端推理。优先进程内执行；若 PyTorch DLL 冲突则尝试子进程。
    """
    try:
        return _run_inference_in_process(image_path, confidence_threshold)
    except OSError as exc:
        msg = str(exc)
        if "1114" in msg or "c10.dll" in msg.lower() or "dll" in msg.lower():
            from src.model.inference_runner import run_inference_via_subprocess

            return run_inference_via_subprocess(image_path, confidence_threshold)
        raise


def _run_inference_in_process(
    image_path: str | Path,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """原进程内推理逻辑。"""
    if not _USE_REAL_MODEL:
        load_model()

    thresh = confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在: {path}")

    image_bgr = imread_bgr(path)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {path}")

    h, w = image_bgr.shape[:2]
    rgb = preprocess_image(image_bgr)

    # 推理
    class_map, conf_map, _prob_full = predict(rgb)

    # 置信度阈值 → 二值 mask
    binary_mask = class_map_to_binary(class_map, conf_map, thresh)

    # 各类面积 & 置信度
    cls_areas = per_class_areas(class_map, conf_map, thresh)
    cls_means = per_class_mean_confidence(class_map, conf_map)

    # 缺陷统计（基于二值 mask）
    mean_conf = float(conf_map[binary_mask > 0].mean()) if binary_mask.any() else 0.0
    stats = analyze_binary_mask(binary_mask, mean_confidence=mean_conf)

    # 保存图像
    stem = path.stem
    paths = _save_visuals(image_bgr, class_map, conf_map, binary_mask, stem)

    # 缺陷类别汇总
    defect_names = [k for k, v in cls_areas.items() if v > 0]
    defect_categories = " / ".join(defect_names) if defect_names else "无明显缺陷"

    # 平均置信度
    confidence_score = float(conf_map.mean()) if conf_map.size > 0 else 0.0

    result: dict[str, Any] = {
        # 原有兼容字段
        "original_path": paths["original_path"],
        "mask_path": paths["mask_path"],
        "overlay_path": paths["overlay_path"],
        "defect_categories": defect_categories,
        "defect_area": stats.defect_area,
        "defect_area_ratio": stats.defect_area_ratio,
        "connected_components": stats.connected_components,
        "max_defect_area": stats.max_defect_area,
        "confidence_score": confidence_score,
        "severity_level": stats.severity_level,
        "suggestion": stats.suggestion,
        "image_width": w,
        "image_height": h,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "bboxes": stats.bboxes,
        "mode": "pv-s3",
        # PV-S3 新增字段
        "colorized_mask_path": paths["colorized_mask_path"],
        "heatmap_path": paths["heatmap_path"],
        "per_class_areas": cls_areas,
        "mean_confidence_per_class": cls_means,
        "confidence_threshold": thresh,
    }
    return result


# ===================================================================
# 状态查询
# ===================================================================


def get_model_status() -> dict[str, Any]:
    """用于 UI 展示模型加载状态。"""
    weight_exists = MODEL_WEIGHT_PATH.is_file()
    pretrained_exists = PRETRAINED_RESNET_DIR.joinpath("resnet50.pth").is_file()
    return {
        "weight_exists": weight_exists,
        "pretrained_exists": pretrained_exists,
        "using_real_model": _USE_REAL_MODEL,
        "weight_path": str(MODEL_WEIGHT_PATH),
    }
