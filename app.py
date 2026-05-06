# -*- coding: utf-8 -*-
"""
Streamlit 主应用：光伏板 EL 图像缺陷检测与智能运维辅助系统。
运行：在项目根目录执行  streamlit run app.py
"""
from __future__ import annotations

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
from src.report.report_generator import generate_report
from src.utils.config import CSV_DIR, PROJECT_ROOT, ensure_directories

# 页面配置
st.set_page_config(
    page_title="光伏板 EL 缺陷检测系统",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()
init_db()


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
    status = get_model_status()
    st.caption(
        f"当前模型权重：{'已找到文件' if status['weight_exists'] else '未放置'} "
        f"| 推理模式：`{status['using_real_model'] and 'PV-S3 真实推理' or 'Fallback 演示'}`"
    )

    up = st.file_uploader("上传 EL 图像（jpg/jpeg/png）", type=["jpg", "jpeg", "png"])
    run_btn = st.button("开始检测", type="primary")

    if run_btn and up is not None:
        if not validate_image_filename(up.name):
            st.error("仅支持 jpg、jpeg、png")
            return
        try:
            saved = save_uploaded_file(up)
            with st.spinner("正在推理与生成可视化..."):
                res = run_inference(saved)
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

    c1, c2, c3 = st.columns(3)
    orig = resolve_project_path(res["original_path"], PROJECT_ROOT)
    mask = resolve_project_path(res["mask_path"], PROJECT_ROOT)
    over = resolve_project_path(res["overlay_path"], PROJECT_ROOT)
    with c1:
        st.subheader("原图")
        if orig.is_file():
            st.image(Image.open(orig), use_container_width=True)
    with c2:
        st.subheader("缺陷 Mask")
        if mask.is_file():
            st.image(Image.open(mask), use_container_width=True)
    with c3:
        st.subheader("叠加图")
        if over.is_file():
            st.image(Image.open(over), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("缺陷面积", f"{res['defect_area']} px")
    m2.metric("面积占比", f"{res['defect_area_ratio'] * 100:.2f}%")
    m3.metric("连通区域数", f"{res['connected_components']}")
    m4.metric("严重程度", res["severity_level"])

    st.markdown(f"**维护建议：** {res['suggestion']}")
    st.markdown(f"**平均置信度：** {res['confidence_score']:.4f}")

    col_a, col_b = st.columns(2)
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
                )
                st.session_state["last_saved_result_id"] = rid
                st.success(f"已保存，result_id={rid}")
            except Exception as e:
                st.error(f"保存失败：{e}")
    with col_b:
        if st.button("生成 Word 报告"):
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


def page_batch():
    st.header("批量图像检测")
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
                res = run_inference(saved)
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
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)

    del_id = st.number_input("输入要删除的 result_id", min_value=1, step=1)
    if st.button("删除该记录"):
        if delete_record(int(del_id)):
            st.success("已删除")
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()


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

### 技术亮点
- 统一推理接口，便于替换为官方 PV-S3 权重  
- Fallback 保证课设演示不中断  
- 业务侧量化指标与报告自动生成  
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
        "模型与技术说明": page_docs,
        "模型指标评价": page_metrics,
    }
    st.sidebar.title("导航")
    choice = st.sidebar.radio("选择页面", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
