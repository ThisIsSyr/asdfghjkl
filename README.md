# 基于 PV-S3 半监督语义分割的光伏板 EL 图像缺陷检测与智能运维辅助系统

## 项目简介

本项目面向光伏板生产检测与电站运维场景，构建一套 **可演示的完整 AI 应用系统**：用户上传电致发光（EL）图像后，系统进行缺陷区域分割、量化统计、严重程度评估、运维建议生成，并支持历史记录、批量检测、Word 报告导出与分割指标评价。

## 项目背景

光伏组件在长期运行中易出现裂纹、栅线中断、接触不良、腐蚀与暗斑等缺陷。传统人工查看 EL 图效率低、主观性强。本项目以 **PV-S3** 半监督语义分割思路为核心（参考 [PV-S3](https://github.com/abj247/PV-S3)），在课程设计周期内优先保证 **系统闭环可运行**：在无官方权重时通过 **Fallback 传统图像处理** 生成演示用分割结果，权重接入后仅需扩展 `src/model/pv_s3_infer.py` 中的加载与 `predict` 逻辑。

## 功能模块

| 模块 | 说明 |
|------|------|
| 图像上传 | 单张/批量，`jpg/jpeg/png`，保存至 `uploads/` |
| 模型推理 | 统一接口 `run_inference()`；真实 PV-S3 或 Fallback |
| 可视化 | Mask、原图、叠加图输出至 `outputs/` |
| 缺陷分析 | 面积、占比、连通域、最大缺陷、BBox、严重程度 |
| 运维建议 | 按严重程度生成文本建议 |
| Web 展示 | Streamlit 多页面导航 |
| 数据库 | SQLite：`data/detection.db` |
| 报告 | python-docx 生成至 `reports/` |
| 指标评价 | Precision / Recall / F1 / IoU（预测与真值 Mask） |

## 技术栈

- Python 3.10+
- PyTorch（预留真实模型推理）
- OpenCV、PIL、NumPy
- Streamlit
- SQLite
- Pandas、Matplotlib
- python-docx

## 项目目录结构

```
pv_s3_defect_detection_system/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── model/pv_s3_infer.py
│   ├── processing/
│   ├── database/db.py
│   ├── report/report_generator.py
│   ├── evaluation/metrics.py
│   └── utils/config.py
├── uploads/
├── outputs/masks/ overlays/ csv/
├── reports/
├── data/detection.db（运行后自动生成）
├── weights/（可选，放置 PV-S3 权重）
└── sample_images/
```

## 安装方法

```bash
cd pv_s3_defect_detection_system
pip install -r requirements.txt
```

接入真实 PV-S3 时需安装 PyTorch（本仓库默认 Fallback **不依赖** torch，便于环境快速就绪）：

```bash
pip install -r requirements-optional.txt
```

也可参考 [PyTorch 官网](https://pytorch.org/) 选择与本机 CUDA 匹配的轮子。

## 运行方法

```bash
streamlit run app.py
```

浏览器将打开本地页面；侧边栏可切换：首页、单张检测、批量检测、历史记录、模型说明、指标评价。

## 使用说明

1. **单张检测**：上传图片 → 开始检测 → 查看原图/Mask/叠加图与指标 → 可保存数据库、生成 Word 报告。  
2. **批量检测**：多文件上传 → 批量检测 → 查看表格与分布图 → 下载 CSV → 可将本批结果写入数据库。  
3. **历史记录**：按严重程度筛选；可按 `result_id` 删除记录。  
4. **指标评价**：上传预测 Mask 与真实 Mask（尺寸一致），计算 Precision、Recall、F1、IoU。

## PV-S3 模型接入说明

1. 克隆官方仓库 [PV-S3](https://github.com/abj247/PV-S3)，按说明训练或获取权重。  
2. 将权重保存为例如 `weights/pv_s3_best.pth`（可在 `src/utils/config.py` 中修改 `MODEL_WEIGHT_PATH`）。  
3. 编辑 `src/model/pv_s3_infer.py`：  
   - 在 `load_model()` 中构建网络并 `load_state_dict`；  
   - 将 `_USE_REAL_MODEL` 置为 `True`；  
   - 在 `predict()` 中完成张量前向，输出与图像同尺寸的类别或缺陷概率图。  
4. `postprocess_mask()` 已支持概率图与 resize；类别图可使用 argmax 分支。

## Fallback 演示模式说明

当未实现真实加载逻辑或未启用 `_USE_REAL_MODEL` 时，系统使用 **灰度 + Canny/阈值/形态学** 等方法合成近似缺陷响应，再二值化得到 Mask。该模式 **不代表真实 PV-S3 精度**，仅用于课设演示与流程验证。

## 数据库设计

- **image_info**：图片元数据（路径、尺寸、上传时间）  
- **detection_result**：检测结果（Mask/叠加路径、面积、占比、连通域、置信度、严重程度、建议等）  
- **report_info**：报告路径与生成时间（关联 `result_id`）

详细字段见 `src/database/db.py` 中 `init_db()`。

## 报告生成说明

`src/report/report_generator.py` 中 `generate_report(result_dict)` 使用 **python-docx** 写入标题、统计表、运维建议与结论，并嵌入原图、Mask、叠加图（路径存在于项目内时）。报告文件名：`report_<图片名>_<时间>.docx`。

## 项目亮点

1. 完整 AI 应用系统（上传—推理—分析—展示—存储—报告），非单一脚本。  
2. 基于 PV-S3 半监督语义分割思想，接口预留清晰。  
3. 像素级缺陷分割展示（Mask + Overlay）。  
4. 缺陷量化与严重程度、运维建议。  
5. 批量检测与 CSV、图表分布。  
6. SQLite 历史与 Word 报告。  
7. 分割指标评价（需双 Mask）。  
8. 适合课程设计答辩演示。

## 后续优化方向

- 接入官方 PV-S3 训练流程与真实权重，替换 Fallback。  
- 增加验证集与 mIoU 批量评估、混淆矩阵可视化。  
- 可选接入大模型 API，对结构化结果做自然语言报告增强（不参与核心分割）。  
- 前端美化与权限管理。

## 课程设计答辩说明

演示建议顺序：展示首页技术路线 → **单张上传** 展示三图与指标 → **保存记录** → **历史页** 查询 → **批量检测** 展示表格与柱状图/直方图 → **指标评价** 上传双 Mask → **Word 报告** 打开查看。强调：**核心缺陷来自 PV-S3/Fallback 分割与规则统计，大模型非必需。**

---

运行命令示例：

```bash
pip install -r requirements.txt
streamlit run app.py
```
