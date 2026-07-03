# Windows PyTorch c10.dll 错误修复指南

## 现象

```
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。
Error loading ".../torch/lib/c10.dll" or one of its dependencies.
```

## 原因

当前 **Anaconda base** 环境中的 PyTorch 安装损坏或与 Intel MKL/OpenMP 冲突，与项目代码无关。

## 推荐修复（新建 conda 环境）

在项目目录 PowerShell 中执行：

```powershell
.\scripts\setup_conda_env.ps1
conda activate pv_s3
streamlit run app.py
```

## 备选：降级 PyTorch

```powershell
pip uninstall torch torchvision -y
pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

## 验证

```powershell
python -c "import torch; print('OK', torch.__version__)"
```

输出 `OK 2.x.x` 后再启动 Streamlit。

## 其他

- 安装 [VC++ 2015-2022 x64 运行库](https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist)
- 修复后务必 **Ctrl+C 完全退出** 旧 Streamlit 进程再重启
