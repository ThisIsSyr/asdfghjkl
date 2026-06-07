# -*- coding: utf-8 -*-
"""
独立子进程推理入口：在干净进程中**最先**加载 PyTorch，规避 Streamlit 主进程 DLL 冲突。

用法:
  python scripts/inference_worker.py <image_path> [confidence_threshold]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 必须在任何其他 heavy 库之前设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 先加载 PyTorch，再加载 OpenCV / 项目模块
try:
    import torch  # noqa: F401 — 固定 DLL 加载顺序
except OSError as exc:
    print(json.dumps({"ok": False, "error": f"PyTorch 加载失败: {exc}"}, ensure_ascii=False))
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "缺少 image_path 参数"}, ensure_ascii=False))
        sys.exit(1)

    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        from src.model.pv_s3_infer import run_inference

        kwargs = {"confidence_threshold": threshold} if threshold is not None else {}
        result = run_inference(image_path, **kwargs)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
