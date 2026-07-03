# 新建独立 conda 环境并安装 PyTorch（推荐修复 WinError 1114）
$ErrorActionPreference = "Stop"

Write-Host "将创建 conda 环境: pv_s3 (Python 3.11 + PyTorch CPU)" -ForegroundColor Cyan

conda create -n pv_s3 python=3.11 -y
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "安装 PyTorch (conda) ..."
conda install -n pv_s3 pytorch torchvision cpuonly -c pytorch -y
if ($LASTEXITCODE -ne 0) { exit 1 }

$proj = Split-Path $PSScriptRoot -Parent
Write-Host "安装项目依赖 ..."
conda run -n pv_s3 pip install -r "$proj\requirements.txt"

Write-Host ""
Write-Host "完成！请使用以下命令启动：" -ForegroundColor Green
Write-Host "  conda activate pv_s3"
Write-Host "  cd `"$proj`""
Write-Host "  streamlit run app.py"
