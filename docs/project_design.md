# 课程设计文档 — 项目设计说明

## 一、项目需求分析

### 1.1 业务需求

- 对光伏板 EL 图像进行**像素级缺陷区域**识别，并输出可解释结果。  
- 对缺陷进行**面积、数量、占比、严重程度**等量化，支撑运维决策。  
- 提供 **Web 交互**、**历史追溯**、**报告导出**，形成可演示的完整系统。

### 1.2 功能需求

| 编号 | 需求描述 |
|------|----------|
| R1 | 支持单张、批量上传 jpg/jpeg/png |
| R2 | 调用 PV-S3 或 Fallback 生成分割 Mask 与叠加图 |
| R3 | 统计缺陷面积、占比、连通域、最大缺陷、外接框 |
| R4 | 按规则输出严重程度与维护建议 |
| R5 | SQLite 存储图片与检测、报告元数据 |
| R6 | 自动生成 Word 报告 |
| R7 | 支持预测/真值 Mask 的 Precision、Recall、F1、IoU |

### 1.3 非功能需求

- 无权重时系统仍可用（Fallback）。  
- 代码模块化，便于后续接入真实 PV-S3。  
- 界面清晰，适合作业答辩演示。

## 二、系统总体设计

系统采用 **B/S 式单进程架构**：**Streamlit** 作为表示层，**Python 业务层** 完成推理、分析与持久化，**SQLite** 作本地存储。推理层通过 `run_inference()` 统一对外，内部根据是否加载成功 PV-S3 选择真实网络或 Fallback。

```
用户 → Streamlit → run_inference / 分析 / 报告
                    ↓
              outputs/  reports/  data/detection.db
```

## 三、功能模块设计

| 模块 | 主要文件 | 职责 |
|------|----------|------|
| 配置与目录 | `src/utils/config.py` | 路径、阈值、ensure_directories |
| 推理 | `src/model/pv_s3_infer.py` | load/preprocess/predict/postprocess/run_inference |
| 图像工具 | `src/processing/image_utils.py` | 保存上传、读尺寸 |
| 缺陷分析 | `src/processing/defect_analysis.py` | 连通域、严重度、建议 |
| 数据库 | `src/database/db.py` | 三表 CRUD、联合查询、删除 |
| 报告 | `src/report/report_generator.py` | python-docx 生成报告 |
| 指标 | `src/evaluation/metrics.py` | 像素级混淆矩阵与 IoU 等 |
| 界面 | `app.py` | 多页面与图表 |

## 四、模型算法设计（逻辑）

### 4.1 真实 PV-S3（规划）

- 输入 EL 图，经归一化与尺寸对齐后送入分割网络（如 DeepLabv3+ 类骨干）。  
- 输出每像素类别或缺陷类概率；后处理为二值 Mask 并缩放到原图。

### 4.2 Fallback（当前默认可用）

- 灰度化、Canny 边缘、Otsu/反相暗区、形态学操作，融合为**伪概率图**，再阈值化。  
- 仅用于**流程与界面演示**，不代表真实检测性能。

## 五、数据库设计

- **image_info(image_id, image_name, image_path, upload_time, image_width, image_height)**  
- **detection_result(result_id, image_id, model_*, detect_time, mask_path, overlay_path, 统计字段, severity, suggestion)**  
- **report_info(report_id, result_id, report_path, generate_time)**  

`detection_result.image_id` 外键引用 `image_info`；`report_info.result_id` 引用 `detection_result`。

## 六、系统流程设计

1. 用户上传 → 保存 `uploads/`。  
2. `run_inference`：读图 → 预处理 → 预测 → 后处理 Mask → 写 `outputs/masks`、`overlays`。  
3. `analyze_binary_mask`：连通域、严重度、建议。  
4. 页面展示；可选写入 DB、生成 `reports/*.docx`。  
5. 批量结果可导出 CSV 与分布图。

## 七、关键技术说明

- **Streamlit**：快速构建多页与组件。  
- **OpenCV 连通域**：`connectedComponentsWithStats` 统计区域与 BBox。  
- **严重度规则**：以面积占比为主，结合区域数与最大块占比上调一级。  
- **python-docx**：插入图片与表格生成报告。

## 八、测试方案

| 用例 | 预期 |
|------|------|
| 单张 jpg 检测 | 生成三图，指标为合理数值 |
| 无缺陷简单图 | 面积近 0，严重度为正常 |
| 保存与历史 | 库中可查询、可删除 |
| 双 Mask 指标 | 与手算小图一致 |
| 批量多图 | 表格与 CSV 行数一致 |

## 九、项目进度安排（参考）

| 阶段 | 内容 |
|------|------|
| 第 1–2 周 | 需求确认、环境搭建、Fallback 跑通 |
| 第 3–4 周 | 数据库、单页检测、报告 |
| 第 5–6 周 | 批量、图表、指标页 |
| 第 7 周 | 文档、PPT、答辩演练 |

## 十、总结

本设计在**不依赖**官方权重的情况下实现完整业务闭环，并通过 `pv_s3_infer.py` **预留**真实 PV-S3 接入点，兼顾课程设计**可演示性**与**可扩展性**。
