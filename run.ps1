# 一键启动 Streamlit（设置 DLL 相关环境变量）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

Write-Host "检测 PyTorch ..."
python -c "import torch; print('PyTorch OK:', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "PyTorch 加载失败！请运行: .\scripts\setup_conda_env.ps1" -ForegroundColor Red
    Write-Host "或查看 docs/fix_pytorch_windows.md"
    exit 1
}

Write-Host "启动 Streamlit ..."
streamlit run app.py
