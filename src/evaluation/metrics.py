# -*- coding: utf-8 -*-
"""二分类 mask 评价指标（缺陷为正类）。"""
from __future__ import annotations

import numpy as np


def _binarize(arr: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    if arr.dtype != np.bool_:
        return (arr > threshold).astype(np.uint8)
    return arr.astype(np.uint8)


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, threshold: float = 127.0) -> tuple[int, int, int, int]:
    """返回 TP, FP, FN, TN（像素级）。"""
    p = _binarize(pred, threshold).flatten()
    g = _binarize(gt, threshold).flatten()
    if p.shape != g.shape:
        raise ValueError("预测与真值尺寸不一致")
    tp = int(np.logical_and(p == 1, g == 1).sum())
    fp = int(np.logical_and(p == 1, g == 0).sum())
    fn = int(np.logical_and(p == 0, g == 1).sum())
    tn = int(np.logical_and(p == 0, g == 0).sum())
    return tp, fp, fn, tn


def calculate_precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def calculate_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def calculate_iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def calculate_miou(ious: list[float]) -> float:
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def evaluate_masks(pred: np.ndarray, gt: np.ndarray, threshold: float = 127.0) -> dict[str, float]:
    """一次性计算常用指标。"""
    tp, fp, fn, tn = confusion_matrix(pred, gt, threshold)
    prec = calculate_precision(tp, fp)
    rec = calculate_recall(tp, fn)
    f1 = calculate_f1(prec, rec)
    iou = calculate_iou(tp, fp, fn)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou,
        "miou": iou,
    }
