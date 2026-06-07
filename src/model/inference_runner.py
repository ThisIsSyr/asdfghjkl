# -*- coding: utf-8 -*-
"""通过子进程运行推理（子进程中优先加载 PyTorch）。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKER = PROJECT_ROOT / "scripts" / "inference_worker.py"


def run_inference_via_subprocess(
    image_path: str | Path,
    confidence_threshold: float | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    if not WORKER.is_file():
        raise RuntimeError(f"推理 worker 不存在: {WORKER}")

    cmd = [sys.executable, str(WORKER), str(Path(image_path).resolve())]
    if confidence_threshold is not None:
        cmd.append(str(confidence_threshold))

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if not stdout:
        raise RuntimeError(f"推理子进程无输出。stderr: {stderr}")

    # worker 最后一行应为 JSON
    line = stdout.splitlines()[-1]
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"推理子进程返回非 JSON: {stdout[:500]}") from exc

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "子进程推理失败"))
    return payload["result"]
