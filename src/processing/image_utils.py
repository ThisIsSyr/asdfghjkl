# -*- coding: utf-8 -*-
"""图像读写与保存工具。"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.config import ALLOWED_IMAGE_EXTS, UPLOAD_DIR, ensure_directories


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """
    读取 BGR 图像，兼容 Windows 中文/Unicode 路径。
    OpenCV 的 cv2.imread 无法直接读取含非 ASCII 字符的路径。
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except OSError:
        return None


def imwrite_bgr(path: str | Path, image_bgr: np.ndarray) -> bool:
    """写入 BGR 图像，兼容 Windows Unicode 路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def validate_image_filename(name: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTS


def save_uploaded_file(uploaded_file, subfolder: str | None = None) -> Path:
    """
    将 Streamlit UploadedFile 保存到 uploads/。
    返回保存后的绝对路径。
    """
    ensure_directories()
    target_dir = UPLOAD_DIR / subfolder if subfolder else UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest.resolve()


def copy_image_to_uploads(src_path: str | Path) -> Path:
    """复制外部路径图片到 uploads。"""
    ensure_directories()
    src = Path(src_path)
    dest = UPLOAD_DIR / src.name
    shutil.copy2(src, dest)
    return dest.resolve()


def get_image_size(path: str | Path) -> tuple[int, int]:
    """返回 (width, height)。"""
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def ensure_rgb_uint8(path: str | Path) -> "tuple":
    """读取为 RGB numpy，供推理使用。延迟导入 numpy 避免循环。"""
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)
