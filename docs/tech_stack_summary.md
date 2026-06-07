# 技术栈总结与后续工作

## 一、技术栈总览

| 层次 | 技术 | 在本项目中的作用 |
|------|------|------------------|
| **编程语言** | Python 3.10+ | 全栈实现语言 |
| **深度学习** | PyTorch 2.x | PV-S3 `EntireModel` 加载与 GPU/CPU 推理 |
| **视觉骨干** | ResNet50 + DeepLabv3+ 解码器 | PV-S3 Teacher 分支 encoder1/decoder1 |
| **半监督思想** | Mean Teacher（PV-S3 原论文） | 训练阶段学生/教师一致性；推理用 Teacher1 |
| **语义分割** | 5 类像素分类 | 背景 / 裂纹 / 栅线中断 / 接触不良 / 腐蚀 |
| **图像处理** | OpenCV、NumPy、PIL | 读写、调色板可视化、连通域、阈值 mask |
| **可视化** | Matplotlib | 置信度热图、批量统计柱状图/直方图 |
| **Web 前端** | Streamlit | 多页导航、上传、滑块阈值、结果展示 |
| **数据库** | SQLite | `image_info` / `detection_result` / `report_info` |
| **数据分析** | Pandas | 批量结果表、CSV 导出 |
| **文档生成** | python-docx | 规则模板 Word 报告（图+表+建议） |
| **Agent 框架** | LangChain + langchain-openai | Tool-Calling Agent 驱动智能报告 |
| **大模型接入** | OpenAI 兼容 API | DeepSeek / OpenAI / 通义千问等（`base_url` + `api_key`） |
| **配置管理** | python-dotenv + `.env` | API Key、模型名、Base URL 集中配置 |

## 二、系统架构（当前）

```
用户上传 EL 图
    ↓
Streamlit UI（阈值滑块、单张/批量）
    ↓
pv_s3_infer.run_inference()
    ├─ EntireModel (Teacher1) → class_map + conf_map
    ├─ 置信度阈值 → 二值 mask
    ├─ defect_analysis → 面积/严重度/建议
    └─ 输出 5 图：原图/分类图/热图/叠加/Mask
    ↓
SQLite 持久化 + CSV 批量导出
    ↓
报告生成（二选一）
    ├─ 规则模板：report_generator.generate_report()
    └─ AI Agent：report_agent.run_report_agent()
            ├─ Tool: get_detection_summary
            ├─ Tool: generate_base_word_report
            └─ Tool: append_ai_analysis → ai_report_generator
```

## 三、Agent 模块说明

| 文件 | 职责 |
|------|------|
| `src/llm/llm_config.py` | API Key、Base URL、模型名；读取 `.env` |
| `src/llm/prompts.py` | System / User 提示词模板 |
| `src/llm/tools.py` | 4 个 LangChain Tool + `ReportAgentContext` |
| `src/llm/report_agent.py` | `create_tool_calling_agent` + `AgentExecutor` |
| `src/report/ai_report_generator.py` | 向 Word 追加「智能分析报告」章节 |
| `.env.example` | 配置模板，复制为 `.env` 即可使用 |

**设计原则：** 大模型**不修改**分割结果，只基于结构化 JSON 撰写自然语言分析。

## 四、快速启用 Agent

```bash
cd asdfghjkl-main
copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY
pip install langchain langchain-openai langchain-core python-dotenv
streamlit run app.py
```

侧边栏填写 API Key 后，在「单张图像检测」点击 **🤖 Agent 生成智能报告**。

## 五、后续工作清单（建议优先级）

### P0 — 答辩前必做
- [ ] 准备 5~10 张代表性 EL 样例（含不同严重度）放入 `sample_images/`
- [ ] 填写 `.env` 并实测 Agent 报告生成全流程
- [ ] 在验证集上跑 **mIoU / F1**，截图填入 PPT
- [ ] 统一答辩演示脚本：单张检测 → 调阈值 → Agent 报告 → 打开 Word

### P1 — 功能完善
- [ ] 批量检测页增加「批量 Agent 报告」队列（当前仅单张）
- [ ] 数据库 `report_info` 区分 `rule` / `ai_agent` 报告类型字段
- [ ] Agent 失败时自动降级为规则模板报告 + 提示
- [ ] 多类别 mask 的 per-class IoU 评价（不仅是二值缺陷）

### P2 — 模型与算法
- [ ] 对比不同置信度阈值下的 PR 曲线，确定默认阈值依据
- [ ] 测试集可视化：错分区域叠加展示
- [ ] 可选：Student 分支或模型集成提升边界精度

### P3 — 工程与展示
- [ ] README 补充 Agent 章节与截图
- [ ] Docker / 一键启动脚本
- [ ] 报告模板美化（封面、页眉、学校 Logo）
- [ ] 历史记录页支持在线预览/下载 Agent 报告

### P4 — 可选加分
- [ ] LangSmith / 本地日志记录 Agent 工具调用链（答辩展示「可观测性」）
- [ ] RAG：接入光伏运维标准 PDF 作为检索增强
- [ ] 部署为内网服务（FastAPI + Streamlit 分离）

## 六、答辩话术要点

1. **核心检测**来自 PV-S3 语义分割 + 规则后处理，可复现、可量化。  
2. **Agent** 仅做报告增强，工具链可审计，避免大模型「幻觉」篡改检测数据。  
3. **置信度阈值**体现工程化思维：精度与召回可 trade-off。  
4. 展示 **五类分割彩图 + 热图 + Word 报告**，体现完整 AI 应用闭环。
