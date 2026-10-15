"""minivllm Streamlit 课表。默认占用 8501。

在仓库根目录启动：
  streamlit run exercises/hub.py

它只负责"选课 + 给出运行命令"；真正的动态演示是各章的 app_XX_*.py，
另开一个终端用 `streamlit run <脚本>` 跑（见页面里的命令块）。
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "exercises"

CHAPTERS = [
    ("ch01", "第 1 章 · LLM 推理基础", "01–06"),
    ("ch02", "第 2 章 · KV Cache 与 PagedAttention", "07–13"),
    ("ch03", "第 3 章 · Continuous Batching 与调度", "14–20"),
    ("ch04", "第 4 章 · 模型执行与 CUDA Graph", "21–27"),
    ("ch05", "第 5 章 · 量化", "28–34"),
    ("ch06", "第 6 章 · 分布式并行", "35–41"),
    ("ch07", "第 7 章 · Attention Kernel", "42–47"),
    ("ch08", "第 8 章 · 端到端部署", "48–50"),
    ("ch09", "第 9 章 · Triton", "51–60"),
    ("ch10", "第 10 章 · AI 编译器", "61–70"),
    ("ch11", "第 11 章 · 昇腾全栈", "71–100"),
]


def list_apps(chapter: str) -> list[Path]:
    folder = EXERCISES / chapter
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("app_*.py") if p.name != "app_common.py")


st.set_page_config(page_title="minivllm 实验台", layout="wide")
st.title("minivllm · 交互实验台")
st.caption("本机 Jupyter Lab → http://localhost:8888  ·  本页是 Streamlit 课表（8501）")

c1, c2, c3 = st.columns(3)
c1.metric("练习册", "100 课")
c2.metric("演示脚本", f"{sum(len(list_apps(ch)) for ch, _, _ in CHAPTERS)} 个")
c3.metric("架构文档", "docs 8 篇")

st.markdown(
    """
**两条用法**

1. Jupyter：打开 `exercises/chXX/NN_*.ipynb`，内核选 `python3`（即你装了 ipykernel 的那个环境）。
2. 演示：在下面选一课，复制命令到另一个终端执行；演示脚本各自占用 8502 端口，本课表留在 8501。
"""
)

chapter_labels = {f"{title}（{span}）": ch for ch, title, span in CHAPTERS}
picked = st.selectbox("章节", list(chapter_labels))
chapter = chapter_labels[picked]
apps = list_apps(chapter)

if not apps:
    st.warning(f"`exercises/{chapter}` 下没有 `app_*.py`。")
    st.stop()

rel_apps = [str(p.relative_to(ROOT)).replace("\\", "/") for p in apps]
choice = st.selectbox("演示脚本", rel_apps)
st.code(
    f"streamlit run {choice} --server.port 8502",
    language="bash",
)

nb_guess = EXERCISES / chapter
st.markdown(f"对应 notebook 目录：`exercises/{chapter}/`")
nbs = sorted(nb_guess.glob("*.ipynb"))
if nbs:
    st.write(" · ".join(f"`{p.name}`" for p in nbs[:12]) + (" …" if len(nbs) > 12 else ""))
