# -*- coding: utf-8 -*-
"""
Streamlit 主应用：光伏板 EL 图像缺陷检测与智能运维辅助系统。
运行：在项目根目录执行  streamlit run app.py
"""
from __future__ import annotations

# 修复 PyTorch + OpenCV OpenMP 冲突（必须在其他 import 之前设置）
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

# 尽早加载 PyTorch，降低与 OpenCV/Matplotlib 的 DLL 冲突概率
_TORCH_PRELOAD_OK = False
_TORCH_PRELOAD_ERROR = ""
try:
    import torch as _torch  # noqa: F401

    _TORCH_PRELOAD_OK = True
except OSError as _e:
    _TORCH_PRELOAD_ERROR = str(_e)
except Exception as _e:
    _TORCH_PRELOAD_ERROR = str(_e)

import json
import sys
from datetime import datetime
from pathlib import Path

# 保证以脚本方式运行时能解析 src 包
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.database.db import (
    delete_record,
    get_all_records,
    init_db,
    insert_detection_result,
    insert_image_info,
    insert_report_info,
    resolve_project_path,
)
from src.evaluation.metrics import evaluate_masks
from src.model.pv_s3_infer import get_model_status, run_inference
from src.processing.image_utils import get_image_size, save_uploaded_file, validate_image_filename
from src.llm.llm_config import DEFAULT_BASE_URL, DEFAULT_MODEL, get_llm_config, is_llm_configured, mask_api_key
from src.report.report_generator import generate_report
from src.utils.config import CSV_DIR, PROJECT_ROOT, ensure_directories
from src.utils.torch_check import FIX_GUIDE, get_torch_status


def _run_report_agent_safe(*args, **kwargs):
    """仅在点击 Agent 报告时加载 LangChain，避免拖慢整站导航。"""
    from src.llm.report_agent import run_report_agent_safe

    return run_report_agent_safe(*args, **kwargs)

# 页面配置
st.set_page_config(
    page_title="光伏板 EL 缺陷检测系统",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()
init_db()

if "torch_status" not in st.session_state:
    st.session_state.torch_status = get_torch_status()


def _show_torch_warning_if_needed() -> None:
    ts = st.session_state.get("torch_status") or {}
    if ts.get("any_ok"):
        return
    st.error("⚠️ **PyTorch 无法加载，模型推理不可用**")
    err = ts.get("in_process_error") or ts.get("subprocess_error") or _TORCH_PRELOAD_ERROR
    if err:
        st.code(err[:800])
    st.markdown(FIX_GUIDE)
    if st.button("重新检测 PyTorch 环境", key="recheck_torch"):
        st.session_state.torch_status = get_torch_status()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()


def render_llm_sidebar() -> dict:
    """侧边栏 LLM 配置（会话级，可覆盖 .env）。"""
    env_cfg = get_llm_config()
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = env_cfg.api_key or ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = env_cfg.base_url
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = env_cfg.model_name

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Agent 配置")
    st.sidebar.caption(
        f"默认 API：`{DEFAULT_BASE_URL}` | 模型：`{DEFAULT_MODEL}`\n\n"
        "在 `.env` 中配置 `LLM_API_KEY` 可永久生效；也可在此临时填写。"
    )
    st.sidebar.text_input(
        "LLM API Key",
        type="password",
        key="llm_api_key",
        help="DeepSeek / OpenAI / 通义等 OpenAI 兼容接口",
    )
    st.sidebar.text_input("API Base URL", key="llm_base_url")
    st.sidebar.text_input("模型名称", key="llm_model")

    if "llm_use_rag" not in st.session_state:
        st.session_state.llm_use_rag = False
    st.sidebar.checkbox(
        "📚 启用光伏领域知识检索 (RAG)",
        key="llm_use_rag",
        help="勾选后 Agent 生成报告时将自动检索知识库",
    )

    api_key = st.session_state.llm_api_key
    base_url = st.session_state.llm_base_url
    model_name = st.session_state.llm_model
    configured = is_llm_configured(api_key)
    st.sidebar.caption(
        f"状态：{'✅ 已配置 ' + mask_api_key(api_key) if configured else '❌ 未配置 API Key'}"
    )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
        "configured": configured,
        "use_rag": st.session_state.llm_use_rag,
    }


def page_home():
    st.title("基于 PV-S3 半监督语义分割的光伏板 EL 图像缺陷检测与智能运维辅助系统")
    st.markdown(
        """
### 项目背景
光伏组件在制造与长期运行中易出现裂纹、栅线中断、接触不良、腐蚀与暗斑等缺陷，会降低发电效率并带来安全隐患。
传统依赖人工查看 EL（电致发光）图像的方式主观性强、效率低。

### 系统功能
- 单张 / 批量上传 EL 图像并自动分割缺陷区域  
- 生成 **Mask**、**叠加图**，并进行面积与连通域统计  
- **严重程度**评估与**运维建议**  
- **SQLite** 历史记录与 **Word** 检测报告  
- **LangChain Agent** 智能撰写报告分析章节（可选，需 API Key）  
- **像素级指标评价**（上传预测与真值 Mask）  

### 技术路线
用户上传 → 预处理 → **PV-S3 / Fallback** 推理 → Mask 与叠加图 → 连通域与量化分析 →  
严重程度与建议 → 可视化展示 → 数据库持久化 → 报告导出  

### 项目亮点
1. 完整的 AI 应用闭环，而非单一脚本推理  
2. 预留 **PV-S3** 权重接入；无权重时 **Fallback** 仍可全流程演示  
3. 像素级缺陷分割与运维决策支持  
        """
    )


def page_single():
    st.header("单张图像检测")
    _show_torch_warning_if_needed()
    status = get_model_status()
    weight_status = (
        "✅ 已加载" if status["using_real_model"]
        else ("⚠️ 权重文件存在但未加载" if status["weight_exists"] else "❌ 未放置权重")
    )
    st.caption(
        f"模型权重：{weight_status} | "
        f"预训练ResNet：{'✅' if status.get('pretrained_exists') else '❌'} | "
        f"推理模式：`{'PV-S3' if status['using_real_model'] else '未加载'}`"
    )

    # 置信度阈值滑块
    st.markdown("**⚙️ 置信度阈值调节**")
    st.caption("只统计模型置信度 ≥ 阈值的像素为缺陷。阈值越高，检测越严格。")
    conf_threshold = st.select_slider(
        "置信度阈值",
        options=[0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999],
        value=0.90,
    )
    st.caption(f"当前阈值：**{conf_threshold}**")

    up = st.file_uploader("上传 EL 图像（jpg/jpeg/png）", type=["jpg", "jpeg", "png"])
    run_btn = st.button("开始检测", type="primary")

    if run_btn and up is not None:
        if not validate_image_filename(up.name):
            st.error("仅支持 jpg、jpeg、png")
            return
        ts = st.session_state.get("torch_status") or {}
        if not ts.get("any_ok"):
            st.error("PyTorch 环境异常，无法推理。请按上方修复指南重建环境后重试。")
            return
        try:
            saved = save_uploaded_file(up)
            with st.spinner(f"正在 PV-S3 推理（阈值={conf_threshold}）..."):
                res = run_inference(saved, confidence_threshold=conf_threshold)
                res["detect_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["last_single"] = res
                st.session_state["last_single_path"] = str(saved)
            st.success("检测完成")
        except Exception as e:
            st.error(f"检测失败：{e}")
            return

    res = st.session_state.get("last_single")
    if not res:
        st.info("请上传图片并点击「开始检测」。")
        return

    # 5 图展示：上排3张 + 下排2张，强制等大
    st.markdown(
        """
        <style>
        .stImage img {
            max-height: 350px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    orig = resolve_project_path(res["original_path"], PROJECT_ROOT)
    colorized = resolve_project_path(res.get("colorized_mask_path", ""), PROJECT_ROOT)
    heatmap = resolve_project_path(res.get("heatmap_path", ""), PROJECT_ROOT)
    over = resolve_project_path(res["overlay_path"], PROJECT_ROOT)
    mask = resolve_project_path(res["mask_path"], PROJECT_ROOT)

    # 上排：原图 | 分类预测图 | 置信度热图
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.subheader("原图")
        if orig.is_file():
            st.image(Image.open(orig), use_container_width=True)
    with r1c2:
        st.subheader("分类预测图")
        if colorized and colorized.is_file():
            st.image(Image.open(colorized), use_container_width=True)
    with r1c3:
        st.subheader("置信度热图")
        if heatmap and heatmap.is_file():
            st.image(Image.open(heatmap), use_container_width=True)

    # 下排：叠加图 | 二值Mask（用 offset columns 居中）
    _, r2c1, r2c2, _ = st.columns([1, 3, 3, 1])
    with r2c1:
        st.subheader("叠加图")
        if over.is_file():
            st.image(Image.open(over), use_container_width=True)
    with r2c2:
        st.subheader("二值 Mask")
        if mask.is_file():
            st.image(Image.open(mask), use_container_width=True)

    # 缺陷总体指标
    st.markdown("---")
    st.subheader("缺陷量化统计")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("缺陷面积", f"{res['defect_area']} px")
    m2.metric("面积占比", f"{res['defect_area_ratio'] * 100:.2f}%")
    m3.metric("连通区域数", f"{res['connected_components']}")
    m4.metric("严重程度", res["severity_level"])

    # 逐类面积
    per_class = res.get("per_class_areas") or {}
    if per_class:
        st.subheader("各类缺陷像素面积")
        cls_cols = st.columns(len(per_class))
        for i, (cls_name, area) in enumerate(per_class.items()):
            with cls_cols[i]:
                st.metric(cls_name, f"{area} px")

    st.markdown(f"**缺陷类别：** {res['defect_categories']}")
    st.markdown(f"**维护建议：** {res['suggestion']}")
    st.markdown(f"**全局平均置信度：** {res['confidence_score']:.4f}")
    st.markdown(f"**使用置信度阈值：** {res.get('confidence_threshold', 'N/A')}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("保存检测记录到数据库"):
            try:
                w, h = get_image_size(orig)
                img_id = insert_image_info(
                    image_name=orig.name,
                    image_path=str(res["original_path"]),
                    image_width=w,
                    image_height=h,
                )
                rid = insert_detection_result(
                    image_id=img_id,
                    model_name=res.get("model_name", ""),
                    model_version=res.get("model_version", ""),
                    mask_path=res["mask_path"],
                    overlay_path=res["overlay_path"],
                    defect_area=res["defect_area"],
                    defect_area_ratio=res["defect_area_ratio"],
                    connected_components=res["connected_components"],
                    max_defect_area=res["max_defect_area"],
                    confidence_score=res["confidence_score"],
                    severity_level=res["severity_level"],
                    suggestion=res["suggestion"],
                    colorized_mask_path=res.get("colorized_mask_path", ""),
                    heatmap_path=res.get("heatmap_path", ""),
                    per_class_stats=json.dumps(res.get("per_class_areas", {}), ensure_ascii=False),
                )
                st.session_state["last_saved_result_id"] = rid
                st.success(f"已保存，result_id={rid}")
            except Exception as e:
                st.error(f"保存失败：{e}")
    with col_b:
        if st.button("生成 Word 报告（规则模板）"):
            try:
                rep = dict(res)
                rep.setdefault("detect_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                path = generate_report(rep)
                rid = st.session_state.get("last_saved_result_id")
                if rid:
                    try:
                        rel_rep = str(Path(path).relative_to(PROJECT_ROOT))
                    except ValueError:
                        rel_rep = path
                    insert_report_info(rid, rel_rep)
                st.success(f"报告已生成：{path}")
            except Exception as e:
                st.error(f"报告生成失败：{e}")

    with col_c:
        llm_cfg = st.session_state.get("llm_sidebar", {})
        if st.button("🤖 Agent 生成智能报告", type="secondary"):
            if not llm_cfg.get("configured"):
                st.error("请先在侧边栏配置 LLM API Key，或复制 .env.example 为 .env 后填写。")
            else:
                try:
                    rep = dict(res)
                    rep.setdefault("detect_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    with st.spinner("Agent 正在分析检测结果并撰写报告（约 30~90 秒）..."):
                        out = _run_report_agent_safe(
                            rep,
                            api_key=llm_cfg.get("api_key"),
                            base_url=llm_cfg.get("base_url"),
                            model_name=llm_cfg.get("model_name"),
                            use_rag=llm_cfg.get("use_rag", False),
                        )
                    if out.get("success"):
                        path = out["report_path"]
                        st.session_state["last_agent_report"] = out
                        rid = st.session_state.get("last_saved_result_id")
                        if rid:
                            rel_rep = out.get("report_path_relative") or path
                            insert_report_info(rid, rel_rep)
                        st.success(f"AI Agent 报告已生成：{path}")
                        with st.expander("查看 Agent 输出摘要"):
                            st.write(out.get("agent_output", ""))
                            if out.get("agent_notes"):
                                st.caption(" | ".join(out["agent_notes"]))
                        rag_sources = out.get("rag_sources")
                        if rag_sources:
                            with st.expander("📚 RAG 知识库检索来源", expanded=False):
                                for i, src in enumerate(rag_sources, 1):
                                    st.markdown(f"**{i}. [{src['score']:.4f}] {src['source']}**")
                                    st.caption(src['content'][:300])
                    else:
                        st.error(out.get("error", "Agent 生成失败"))
                except Exception as e:
                    st.error(f"Agent 报告生成失败：{e}")


def page_batch():
    st.header("批量图像检测")
    _show_torch_warning_if_needed()

    # 置信度阈值
    conf_threshold = st.select_slider(
        "置信度阈值",
        options=[0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999],
        value=0.90,
    )
    st.caption(f"当前阈值：**{conf_threshold}**")

    files = st.file_uploader(
        "一次选择多张图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if st.button("批量检测", type="primary") and files:
        rows = []
        prog = st.progress(0.0)
        for i, f in enumerate(files):
            if not validate_image_filename(f.name):
                continue
            try:
                saved = save_uploaded_file(f, subfolder="batch")
                res = run_inference(saved, confidence_threshold=conf_threshold)
                res["file_name"] = f.name
                rows.append(res)
            except Exception as e:
                rows.append({"file_name": f.name, "error": str(e)})
            prog.progress((i + 1) / len(files))
        st.session_state["batch_rows"] = rows
        st.success("批量检测完成")

    rows = st.session_state.get("batch_rows")
    if not rows:
        st.info("请上传多张图片并点击「批量检测」。")
        return

    df_data = []
    for r in rows:
        if "error" in r:
            df_data.append({"文件名": r["file_name"], "错误": r["error"]})
            continue
        df_data.append(
            {
                "文件名": r.get("file_name", ""),
                "缺陷面积": r["defect_area"],
                "面积占比%": round(r["defect_area_ratio"] * 100, 4),
                "连通域数": r["connected_components"],
                "最大缺陷面积": r["max_defect_area"],
                "严重程度": r["severity_level"],
                "模式": r.get("mode", ""),
                "阈值": r.get("confidence_threshold", ""),
            }
        )
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

    ok_df = df[df["严重程度"].notna()] if "严重程度" in df.columns else df
    if not ok_df.empty and "严重程度" in ok_df.columns:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        vc = ok_df["严重程度"].value_counts()
        order = ["正常", "轻微", "中等", "严重"]
        vc = vc.reindex(order).fillna(0).astype(int)
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        vc.plot(kind="bar", ax=ax1, color=colors)
        ax1.set_title("严重程度分布")
        st.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ratios = pd.to_numeric(ok_df["面积占比%"], errors="coerce").dropna()
        if not ratios.empty:
            ax2.hist(ratios, bins=min(15, max(5, len(ratios))), color="#3498db", edgecolor="white")
        ax2.set_title("缺陷面积占比分布（%）")
        ax2.set_xlabel("面积占比 %")
        st.pyplot(fig2)
        plt.close(fig2)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载 CSV", data=csv_bytes, file_name=csv_name, mime="text/csv")

    if st.button("将本批结果写入数据库"):
        try:
            for r in rows:
                if "error" in r:
                    continue
                orig = resolve_project_path(r["original_path"], PROJECT_ROOT)
                w, h = get_image_size(orig)
                img_id = insert_image_info(
                    image_name=orig.name,
                    image_path=r["original_path"],
                    image_width=w,
                    image_height=h,
                )
                insert_detection_result(
                    image_id=img_id,
                    model_name=r.get("model_name", ""),
                    model_version=r.get("model_version", ""),
                    mask_path=r["mask_path"],
                    overlay_path=r["overlay_path"],
                    defect_area=r["defect_area"],
                    defect_area_ratio=r["defect_area_ratio"],
                    connected_components=r["connected_components"],
                    max_defect_area=r["max_defect_area"],
                    confidence_score=r["confidence_score"],
                    severity_level=r["severity_level"],
                    suggestion=r["suggestion"],
                    colorized_mask_path=r.get("colorized_mask_path", ""),
                    heatmap_path=r.get("heatmap_path", ""),
                    per_class_stats=json.dumps(r.get("per_class_areas", {}), ensure_ascii=False),
                )
            st.success("已写入数据库")
        except Exception as e:
            st.error(f"写入失败：{e}")


def page_history():
    st.header("历史检测记录")
    filt = st.selectbox("严重程度筛选", ["全部", "正常", "轻微", "中等", "严重"])
    records = get_all_records(None if filt == "全部" else filt)
    if not records:
        st.info("暂无记录")
        return

    # 简化显示：只展示关键列
    display_cols = [
        "result_id", "image_name", "detect_time", "severity_level",
        "defect_area", "defect_area_ratio", "connected_components",
        "confidence_score", "model_version",
    ]
    df = pd.DataFrame(records)
    cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)

    # 展开查看逐类详情
    with st.expander("查看详细记录（含逐类面积）"):
        st.dataframe(df, use_container_width=True)

    del_id = st.number_input("输入要删除的 result_id", min_value=1, step=1)
    if st.button("删除该记录"):
        if delete_record(int(del_id)):
            st.success("已删除")
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()


def page_ai_agent():
    st.header("AI Agent 智能报告")
    st.markdown(
        """
本模块使用 **LangChain Tool-Calling Agent**，在 PV-S3 结构化检测结果基础上，
自动调用工具生成 Word 报告并撰写专业分析章节。

**Agent 工具链：**
1. `get_detection_summary` — 读取缺陷面积、类别、严重度等结构化数据  
2. `generate_base_word_report` — 生成含原图/Mask/热图的基础报告  
3. `append_ai_analysis` — 写入执行摘要、缺陷分析、运维建议、风险评估、结论  
4. `get_report_file_path` — 返回报告路径  

**配置方式：** 侧边栏填写 API Key，或在项目根目录创建 `.env`（参考 `.env.example`）。
        """
    )
    llm_cfg = st.session_state.get("llm_sidebar", {})
    st.info(
        f"当前 API：`{llm_cfg.get('base_url', DEFAULT_BASE_URL)}` | "
        f"模型：`{llm_cfg.get('model_name', DEFAULT_MODEL)}` | "
        f"Key：{mask_api_key(llm_cfg.get('api_key', ''))}"
    )
    res = st.session_state.get("last_single")
    if not res:
        st.warning("请先在「单张图像检测」完成一次检测，再在此生成 Agent 报告。")
        return
    use_rag = llm_cfg.get("use_rag", False)
    if use_rag:
        st.info("📚 RAG 知识增强已启用 — Agent 将检索光伏领域知识库辅助分析")
    if st.button("使用 Agent 生成报告", type="primary"):
        if not llm_cfg.get("configured"):
            st.error("请先在侧边栏配置 LLM API Key。")
            return
        rep = dict(res)
        rep.setdefault("detect_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with st.spinner("Agent 运行中..."):
            out = _run_report_agent_safe(
                rep,
                api_key=llm_cfg.get("api_key"),
                base_url=llm_cfg.get("base_url"),
                model_name=llm_cfg.get("model_name"),
                use_rag=llm_cfg.get("use_rag", False),
            )
        if out.get("success"):
            st.success(f"报告路径：{out['report_path']}")
            st.markdown("**Agent 最终输出：**")
            st.write(out.get("agent_output", ""))
            if out.get("agent_notes"):
                st.caption(" | ".join(out["agent_notes"]))
            rag_sources = out.get("rag_sources")
            if rag_sources:
                with st.expander("📚 RAG 知识库检索来源", expanded=False):
                    for i, src in enumerate(rag_sources, 1):
                        st.markdown(f"**{i}. [{src['score']:.4f}] {src['source']}**")
                        st.caption(src['content'][:300])
        else:
            st.error(out.get("error"))


def page_rag():
    """RAG 知识库管理页面。"""
    st.header("📚 RAG 知识库管理")
    st.markdown("""
光伏缺陷检测领域知识库，为 AI Agent 提供专业参考资料。
知识库文档位于 `data/knowledge/`，向量库位于 `data/chroma_db/`。
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("知识库状态")
        kb_dir = Path("data/knowledge")
        md_files = list(kb_dir.glob("*.md")) if kb_dir.exists() else []
        pdf_files = list(kb_dir.glob("*.pdf")) if kb_dir.exists() else []
        st.metric("知识文档数", len(md_files) + len(pdf_files))
        if md_files or pdf_files:
            with st.expander("查看文档列表"):
                for f in md_files:
                    st.caption(f"📄 {f.name}")
                for f in pdf_files:
                    st.caption(f"📑 {f.name}")

        chroma_dir = Path("data/chroma_db")
        chroma_exists = chroma_dir.exists() and any(chroma_dir.iterdir())
        chroma_count = 0
        if chroma_exists:
            try:
                import sqlite3
                conn = sqlite3.connect(str(chroma_dir / "chroma.sqlite3"))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM embeddings")
                chroma_count = cur.fetchone()[0]
                conn.close()
            except Exception:
                pass
        chroma_real = chroma_exists and chroma_count > 0
        status_text = f"✅ 已构建 ({chroma_count} 条)" if chroma_real else ("⚠️ 空库，需构建" if chroma_exists else "❌ 未构建")
        st.metric("向量库状态", status_text)

    with col2:
        st.subheader("构建/重建知识库")
        st.caption("运行知识库灌入脚本，将 Markdown 文档向量化存入 ChromaDB。")
        llm_cfg = st.session_state.get("llm_sidebar", {})

        if st.button("🔨 构建知识库", type="primary"):
            if not llm_cfg.get("configured"):
                st.error("请先在侧边栏配置 LLM API Key（用于生成 Embedding）。")
            else:
                with st.spinner("正在分块、向量化文档……"):
                    try:
                        import subprocess
                        script = Path("scripts/build_knowledge_base.py")
                        cmd = [
                            sys.executable, str(script), "--force",
                            "--api-key", llm_cfg.get("api_key", ""),
                            "--base-url", llm_cfg.get("base_url", ""),
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        if result.returncode == 0:
                            st.success("知识库构建完成！")
                            st.code(result.stdout)
                        else:
                            st.error(f"构建失败")
                            st.code(result.stderr or result.stdout)
                    except Exception as e:
                        st.error(f"构建出错: {e}")

        if st.button("🔍 检查知识库"):
            try:
                import subprocess
                script = Path("scripts/build_knowledge_base.py")
                cmd = [sys.executable, str(script), "--check"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                st.code(result.stdout)
            except Exception as e:
                st.error(f"检查失败: {e}")

    st.markdown("---")
    st.subheader("使用说明")
    st.markdown("""
1. 在侧边栏配置 LLM API Key
2. 点击「构建知识库」完成文档向量化
3. 在侧边栏勾选「启用光伏领域知识检索」
4. 运行 Agent 生成报告时，将自动检索相关知识
    """)


def page_docs():
    st.header("模型与技术说明")
    st.markdown(
        """
### PV-S3 简介
PV-S3 面向光伏缺陷分割，结合半监督学习与语义分割骨干（如 DeepLabv3+），在有限标注下利用大量未标注 EL 图像提升泛化能力。
仓库参考：[PV-S3](https://github.com/abj247/PV-S3)

### 半监督学习
通过 Mean Teacher、一致性正则等机制，让模型在标注稀缺时仍能从未标注数据中学习稳定特征。

### 语义分割
对图像每个像素分类，输出与输入同分辨率的类别图，适合裂纹、暗斑等形状不规则缺陷。

### Mean Teacher
维护学生与教师网络，教师由学生参数滑动平均得到，对无标签数据施加预测一致性约束。

### DeepLabv3+
采用空洞卷积与 ASPP 模块，兼顾多尺度上下文与边界细节，是常见的分割骨干之一。

### 本系统推理流程（概念）
上传图像 → 预处理（缩放/归一化）→ 网络前向 → softmax / argmax → 后处理（阈值、连通域）→ 量化与可视化。

### LangChain Agent 智能报告
大模型**不参与**像素级缺陷分割，仅对 PV-S3 结构化输出进行自然语言分析与报告增强。
使用 Tool-Calling Agent 调用 `generate_base_word_report`、`append_ai_analysis` 等工具。

### 技术亮点
- 真实 PV-S3 五类语义分割 + 置信度阈值过滤  
- LangChain Agent 工具链自动生成 Word 智能分析章节  
- 业务侧量化指标与规则/AI 双模式报告  
        """
    )


def page_metrics():
    st.header("模型指标评价")
    st.caption("上传**预测 Mask** 与**真实 Mask**（单通道二值图，尺寸需一致）")
    p_file = st.file_uploader("预测 Mask", type=["png", "jpg", "jpeg"])
    g_file = st.file_uploader("真实 Mask", type=["png", "jpg", "jpeg"])
    if st.button("计算指标") and p_file and g_file:
        try:
            pred = np.array(Image.open(p_file).convert("L"))
            gt = np.array(Image.open(g_file).convert("L"))
            if pred.shape != gt.shape:
                st.error("两张 Mask 尺寸不一致")
                return
            m = evaluate_masks(pred, gt)
            st.json(m)
            st.success(
                f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, "
                f"F1={m['f1']:.4f}, IoU={m['iou']:.4f}"
            )
        except Exception as e:
            st.error(f"计算失败：{e}")


def main():
    pages = {
        "首页": page_home,
        "单张图像检测": page_single,
        "批量图像检测": page_batch,
        "历史检测记录": page_history,
        "AI 智能报告": page_ai_agent,
        "模型与技术说明": page_docs,
        "模型指标评价": page_metrics,
        "RAG 知识库": page_rag,
    }
    st.sidebar.title("导航")
    page_names = list(pages.keys())
    choice = st.sidebar.radio("选择页面", page_names, key="main_nav_page")
    st.session_state["llm_sidebar"] = render_llm_sidebar()
    pages[choice]()


if __name__ == "__main__":
    main()
