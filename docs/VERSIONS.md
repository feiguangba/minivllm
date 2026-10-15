# 版本策略

教学仓库和 vLLM 运行时解耦。

| 层 | 版本 | 说明 |
|---|---|---|
| 练习册 / Streamlit | 不绑定 vLLM 包 | 用 `exercises/vllm_real.py`（TinyGPT），CPU 可跑完全部 100 课 |
| docs 架构文档 | vLLM **V1** · 标注为 v0.23.0-dev | 路径约定 `v1/engine/...`，见 `docs/01_system_architecture.md` 文首 |
| 本地环境 | `requirements.txt`（Python 3.12） | `pip install -r requirements.txt`；torch 单独安装（cu128 或 CPU 版），见 README「快速开始」 |
| Binder | `.binder/environment.yml` | 云端 CPU 环境，浏览器里免安装体验 |

## 和 V0 的关系

本仓库文档从一开始就按 V1 多进程拓扑写（前端 / EngineCore / Worker + ZMQ）。没有单独的 V0 源码树。若对照网上 2024 年博客里的 `LLMEngine` 单进程循环，以 `docs/` 为准。

## 本机 GPU 备注

作者环境为 RTX 5060 Laptop（sm_120），torch 装 `cu128` 轮子后 `torch.cuda.is_available()` 为 True。

没有 GPU 时 ch01–ch03 与 `minivllm` 纯 CPU 即可跑通；依赖 CUDA 的章节（ch07 Attention Kernel、ch09 Triton 等）会走 CPU/模拟回退路径或提示缺少 CUDA。Windows 上 vLLM 本身不原生支持，真机 `vllm serve` 请走 WSL2 / Linux（见 `exercises/ch08/48_serve_model.ipynb`）。
