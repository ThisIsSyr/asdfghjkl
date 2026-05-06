# -*- coding: utf-8 -*-
"""图像读写与保存工具。"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from src.utils.config import ALLOWED_IMAGE_EXTS, UPLOAD_DIR, ensure_directories


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
