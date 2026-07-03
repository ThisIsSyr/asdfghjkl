# -*- coding: utf-8 -*-
"""PyTorch 环境检测与修复提示。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "inference_worker.py"


def _apply_dll_env() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def check_torch_in_process() -> tuple[bool, str]:
    """当前进程能否 import torch。"""
    _apply_dll_env()
    try:
        import torch  # noqa: F401

        return True, ""
    except OSError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def check_torch_subprocess() -> tuple[bool, str]:
    """子进程（固定 import 顺序）能否 import torch。"""
    if not WORKER_SCRIPT.is_file():
        return False, "缺少 scripts/inference_worker.py"
    code = (
        "import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; "
        "import torch; print('ok')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        )
        if proc.returncode == 0 and "ok" in proc.stdout:
            return True, ""
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"exit code {proc.returncode}"
    except Exception as exc:
        return False, str(exc)


def get_torch_status() -> dict:
    ok_proc, err_proc = check_torch_in_process()
    ok_sub, err_sub = check_torch_subprocess()
    return {
        "in_process_ok": ok_proc,
        "in_process_error": err_proc,
        "subprocess_ok": ok_sub,
        "subprocess_error": err_sub,
        "python_executable": sys.executable,
        "any_ok": ok_proc or ok_sub,
    }


FIX_GUIDE = """
**PyTorch DLL 加载失败（WinError 1114）— 修复步骤**

当前 Anaconda 环境中的 PyTorch 无法加载 `c10.dll`，需重建环境（任选一种）：

**方案 A（推荐）：新建 conda 环境**
```powershell
conda create -n pv_s3 python=3.11 -y
conda activate pv_s3
conda install pytorch torchvision cpuonly -c pytorch -y
cd 项目目录
pip install -r requirements.txt
streamlit run app.py
```

**方案 B：降级并重装 PyTorch（仍在 base 环境）**
```powershell
pip uninstall torch torchvision -y
pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

**方案 C：安装 VC++ 运行库**  
下载安装 [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist)

修复后请**完全关闭** Streamlit 再重新 `streamlit run app.py`。
"""
