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
    DEFAULT_BLUR_RADIUS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_D,
    DEFAULT_ELONGATION_TH,
    DEFAULT_MIN_AREA_GRID,
    DEFAULT_T,
    HEATMAP_DIR,
    LOGITS_DIFF_DIR,
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
    """BGR → RGB float32 [0,1]，可选缩放，应用 ImageNet 归一化。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("preprocess_image 需要 BGR 三通道图")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if target_size is not None:
        tw, th = target_size
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    # ImageNet 归一化（与原始 DataLoader _val_augmentation 一致）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb


# ===================================================================
# 推理核心
# ===================================================================


def predict(
    preprocessed_rgb: np.ndarray,
    tta: bool = False,
    scales: str = "1.0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Teacher 1 推理。支持可选的 TTA（4 种翻转平均）与多尺度融合。

    Returns
    -------
    class_map : np.ndarray  [H, W] int (0-4)
    conf_map  : np.ndarray  [H, W] float32 (0-1)
    prob_full : np.ndarray  [5, H, W] float32
    logits_raw : np.ndarray [5, H, W] float32  softmax 前的原始 logits
    """
    import torch
    import torch.nn.functional as F

    if not _USE_REAL_MODEL or _MODEL is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")

    x = torch.from_numpy(preprocessed_rgb).permute(2, 0, 1).unsqueeze(0)
    device = next(_MODEL.parameters()).device
    x = x.to(device)

    H, W = x.shape[2], x.shape[3]

    # 8 像素对齐（与原项目 data = F.interpolate(data, size=(up_sizes[0], up_sizes[1])) 一致）
    up_h, up_w = ceil(H / 8) * 8, ceil(W / 8) * 8
    x_padded = F.interpolate(x, size=(up_h, up_w), mode="bilinear", align_corners=True)

    if tta or scales != "1.0":
        x_orig = x  # 保留原尺寸用于多尺度缩放
        preds: list[torch.Tensor] = []
        scale_vals = [float(s.strip()) for s in scales.split(",")]
        with torch.no_grad():
            for s in scale_vals:
                model_h = max(8, ceil(H * s / 8) * 8)
                model_w = max(8, ceil(W * s / 8) * 8)
                data_s = x_padded if s == 1.0 else F.interpolate(
                    x_orig, size=(model_h, model_w), mode="bilinear", align_corners=True,
                )
                if tta:
                    # 原图
                    feat = _MODEL.encoder1(data_s)
                    out = _MODEL.decoder1(feat, data_shape=[model_h, model_w])
                    out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
                    preds.append(out)
                    # 水平翻转
                    x_f = torch.flip(data_s, dims=[3])
                    feat = _MODEL.encoder1(x_f)
                    out = _MODEL.decoder1(feat, data_shape=[model_h, model_w])
                    out = F.interpolate(torch.flip(out, dims=[3]), size=(H, W), mode="bilinear", align_corners=True)
                    preds.append(out)
                    # 垂直翻转
                    x_f = torch.flip(data_s, dims=[2])
                    feat = _MODEL.encoder1(x_f)
                    out = _MODEL.decoder1(feat, data_shape=[model_h, model_w])
                    out = F.interpolate(torch.flip(out, dims=[2]), size=(H, W), mode="bilinear", align_corners=True)
                    preds.append(out)
                    # 双向翻转
                    x_f = torch.flip(data_s, dims=[2, 3])
                    feat = _MODEL.encoder1(x_f)
                    out = _MODEL.decoder1(feat, data_shape=[model_h, model_w])
                    out = F.interpolate(torch.flip(out, dims=[2, 3]), size=(H, W), mode="bilinear", align_corners=True)
                    preds.append(out)
                else:
                    feat = _MODEL.encoder1(data_s)
                    out = _MODEL.decoder1(feat, data_shape=[model_h, model_w])
                    out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
                    preds.append(out)
        logits = torch.stack(preds).mean(dim=0)
    else:
        up_h, up_w = ceil(H / 8) * 8, ceil(W / 8) * 8
        x = F.interpolate(x, size=(up_h, up_w), mode="bilinear", align_corners=True)
        with torch.no_grad():
            logits = _MODEL.decoder1(
                _MODEL.encoder1(x),
                data_shape=[up_h, up_w],
            )
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)

    logits_raw = logits[0].detach()  # [5, H, W] before softmax
    prob = torch.softmax(logits, dim=1)[0]  # [5, H, W]

    class_map = prob.argmax(dim=0).cpu().numpy().astype(np.uint8)
    conf_map = prob.max(dim=0).values.cpu().numpy().astype(np.float32)
    prob_full = prob.cpu().numpy()

    return class_map, conf_map, prob_full, logits_raw


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


# ===================================================================
# 三阶段融合后处理（替代置信度阈值过滤）
# ===================================================================


def stage1_logits_diff(
    logits_raw: torch.Tensor,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    阶段1：Logits 差值热力图 + 二值掩码 M1。

    取 top1 - top2 作为模型确信度指标，差值 ≥ T 的区域标记为潜在缺陷。

    Parameters
    ----------
    logits_raw : torch.Tensor  [5, H, W]  softmax 前的原始 logits
    T : float                  Logits diff 阈值

    Returns
    -------
    diff_map : np.ndarray  [H, W] float32  top1 - top2 差值图
    mask_M1  : np.ndarray  [H, W] uint8   0/255 二值掩码
    """
    import torch

    logits_sorted, _ = torch.sort(logits_raw, dim=0, descending=True)
    diff = logits_sorted[0] - logits_sorted[1]  # [H, W]
    diff_np = diff.cpu().numpy()
    mask = (diff >= T).byte().cpu().numpy() * 255
    return diff_np, mask


def stage2_local_deviation(
    preprocessed_rgb: np.ndarray,
    D: float,
    blur_radius: int,
    elongation_th: float,
    min_area_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    阶段2：局部灰度偏差 + 栅线过滤。

    使用模型输入的 RGB float32 [0,1] 图像（即 preprocess_image 的输出），
    与原始 inference.py 的 restore_transform → grayscale 保持一致。

    Parameters
    ----------
    preprocessed_rgb : np.ndarray  [H, W, 3] float32 [0,1] RGB（模型输入前）
    D                : float       灰度偏差阈值（越低越敏感）
    blur_radius      : int         高斯模糊半径
    elongation_th    : float       栅线长宽比阈值（越高越宽松）
    min_area_grid    : int         栅线最小面积（像素）

    Returns
    -------
    deviation     : np.ndarray  [H, W] float32  灰度偏差图
    mask_M2_raw   : np.ndarray  [H, W] uint8    原始偏差掩码
    grid_mask     : np.ndarray  [H, W] uint8    检测到的栅线
    mask_M2_clean : np.ndarray  [H, W] uint8    去栅线后的掩码
    """
    # 反 ImageNet 归一化 → [0,1] → [0,255] 灰度图（与 restore_transform 一致）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb_01 = (preprocessed_rgb * std + mean).clip(0, 1)
    gray_uint8 = (rgb_01 * 255.0).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(gray_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 与 PIL ImageFilter.GaussianBlur(radius=blur_radius) 保持一致
    # PIL: sigma = radius, kernel = 2*radius + 1
    ksize = 2 * blur_radius + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=blur_radius)
    deviation = np.abs(gray - blurred)

    mask_M2_raw = (deviation >= D).astype(np.uint8) * 255

    # 连通域分析 → 删除长条状栅线
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_M2_raw, connectivity=8,
    )
    grid_mask = np.zeros_like(mask_M2_raw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        elongation = max(w, h) / (min(w, h) + 1e-6)
        if elongation > elongation_th and area > min_area_grid:
            grid_mask[labels == i] = 255

    mask_M2_clean = mask_M2_raw.copy()
    mask_M2_clean[grid_mask > 0] = 0

    return deviation, mask_M2_raw, grid_mask, mask_M2_clean


def stage3_merge(
    mask_M1: np.ndarray,
    mask_M2_clean: np.ndarray,
    deviation: np.ndarray,
    D: float,
    diff_map: np.ndarray,
    T_high: float,
) -> np.ndarray:
    """
    阶段3：加权掩码合并 → 最终二值掩码。

    规则：
    - M2_clean 异常区域 → 使用 M1 的正常阈值 T
    - M2_dark 背景区域 → 使用较高的 T_high 兜底（T + 2）

    Returns
    -------
    mask_final : np.ndarray  [H, W] uint8  0/255 最终二值掩码
    """
    mask_M2_dark = (deviation < D).astype(np.uint8) * 255
    mask_high_conf = (diff_map >= T_high).astype(np.uint8) * 255

    mask_final = np.zeros_like(mask_M1)
    mask_final[mask_M2_clean > 0] = mask_M1[mask_M2_clean > 0]
    mask_final[mask_M2_dark > 0] = mask_high_conf[mask_M2_dark > 0]

    return mask_final


def per_class_areas(class_map: np.ndarray, mask_final: np.ndarray) -> dict[str, int]:
    """统计各类别缺陷像素面积（基于最终融合掩码 mask_final 过滤）。"""
    areas: dict[str, int] = {}
    for cid in range(1, NUM_CLASSES):
        mask = (class_map == cid) & (mask_final > 0)
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


def _save_heatmap(
    data: np.ndarray,
    save_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    label: str = "Confidence",
) -> None:
    """保存热图（matplotlib），兼容 Unicode 路径。"""
    from io import BytesIO

    vmin = vmin if vmin is not None else float(data.min())
    vmax = vmax if vmax is not None else float(data.max())

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label=label)
    plt.axis("off")
    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(buf.getvalue())


def _save_logits_diff_heatmap(diff_map: np.ndarray, save_path: Path) -> None:
    """保存 logits 差值热图（自动缩放到实际数值范围）。"""
    vmax = max(float(diff_map.max()), 1.0)
    _save_heatmap(diff_map, save_path, vmin=0.0, vmax=vmax, label="Logits diff")


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
    T: float | None = None,
    D: float | None = None,
    blur_radius: int | None = None,
    elongation_th: float | None = None,
    min_area_grid: int | None = None,
    tta: bool = False,
    scales: str = "1.0",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    端到端推理。优先进程内执行；若 PyTorch DLL 冲突则尝试子进程。

    Parameters
    ----------
    image_path : 输入 EL 图像路径。
    T : Logits diff 主阈值（默认 7）。
    D : 局部灰度偏差阈值（默认 15）。
    blur_radius : 高斯模糊半径（默认 15）。
    elongation_th : 栅线长宽比阈值（默认 25）。
    min_area_grid : 栅线最小面积像素（默认 150）。
    tta : 是否启用 TTA（4 种翻转平均）。
    scales : 多尺度推理，逗号分隔（如 "0.75,1.0,1.25"）。
    **kwargs : 兼容旧版 confidence_threshold 参数（忽略）。
    """
    try:
        return _run_inference_in_process(
            image_path,
            T=T,
            D=D,
            blur_radius=blur_radius,
            elongation_th=elongation_th,
            min_area_grid=min_area_grid,
            tta=tta,
            scales=scales,
        )
    except OSError as exc:
        msg = str(exc)
        if "1114" in msg or "c10.dll" in msg.lower() or "dll" in msg.lower():
            from src.model.inference_runner import run_inference_via_subprocess

            return run_inference_via_subprocess(image_path)
        raise


def _run_inference_in_process(
    image_path: str | Path,
    T: float | None = None,
    D: float | None = None,
    blur_radius: int | None = None,
    elongation_th: float | None = None,
    min_area_grid: int | None = None,
    tta: bool = False,
    scales: str = "1.0",
) -> dict[str, Any]:
    """原进程内推理逻辑（三阶段融合）。"""
    if not _USE_REAL_MODEL:
        load_model()

    T_val = T if T is not None else DEFAULT_T
    D_val = D if D is not None else DEFAULT_D
    blur_val = blur_radius if blur_radius is not None else DEFAULT_BLUR_RADIUS
    elong_val = elongation_th if elongation_th is not None else DEFAULT_ELONGATION_TH
    min_area_val = min_area_grid if min_area_grid is not None else DEFAULT_MIN_AREA_GRID
    T_high_val = T_val + 2

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在: {path}")

    image_bgr = imread_bgr(path)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {path}")

    orig_h, orig_w = image_bgr.shape[:2]
    h, w = orig_h, orig_w

    rgb = preprocess_image(image_bgr)

    # 推理（返回 logits_raw 供阶段1使用）
    class_map, conf_map, _prob_full, logits_raw = predict(rgb, tta=tta, scales=scales)

    # ===== 三阶段融合后处理 =====
    # 阶段1: Logits 差值 → M1
    diff_map, mask_M1 = stage1_logits_diff(logits_raw, T_val)

    # 阶段2: 局部灰度偏差 → M2_clean（基于模型输入图像，与原始 inference.py 对齐）
    deviation, _mask_M2_raw, _grid_mask, mask_M2_clean = stage2_local_deviation(
        rgb, D_val, blur_val, elong_val, min_area_val,
    )

    # 阶段3: 加权合并 → final mask
    binary_mask = stage3_merge(mask_M1, mask_M2_clean, deviation, D_val, diff_map, T_high_val)

    # 各类面积 & 置信度（基于最终 mask）
    cls_areas = per_class_areas(class_map, binary_mask)
    cls_means = per_class_mean_confidence(class_map, conf_map)

    # 缺陷统计（基于二值 mask）
    mean_conf = float(conf_map[binary_mask > 0].mean()) if binary_mask.any() else 0.0
    stats = analyze_binary_mask(binary_mask, mean_confidence=mean_conf)

    # 重新加载原图用于可视化保存
    image_bgr = imread_bgr(path)
    stem = path.stem
    paths = _save_visuals(image_bgr, class_map, conf_map, binary_mask, stem)

    # 保存 logits_diff 热图（自动缩放到数据实际范围, 不截断）
    try:
        diff_map_big = cv2.resize(diff_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        logits_diff_path = LOGITS_DIFF_DIR / f"{stem}_logits_diff.png"
        _save_logits_diff_heatmap(diff_map_big, logits_diff_path)
        paths["logits_diff_path"] = get_relative_to_project(logits_diff_path)
    except Exception:
        paths["logits_diff_path"] = ""

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
        "logits_diff_path": paths["logits_diff_path"],
        "colorized_mask_path": paths["colorized_mask_path"],
        "heatmap_path": paths["heatmap_path"],
        "per_class_areas": cls_areas,
        "mean_confidence_per_class": cls_means,
        "confidence_threshold": T_val,
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
