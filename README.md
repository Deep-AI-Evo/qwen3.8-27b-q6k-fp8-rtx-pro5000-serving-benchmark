# ⚡ Qwen3.8-27B 部署方案实测报告

> **llama.cpp (Q6_K) vs vLLM (FP8) 全面对比 · TTFT / Prefill / Decode / 并发**
> 基于 2026-08-15 实机测试，所有数据均可复现（测试脚本附于仓库）

![Platform](https://img.shields.io/badge/Platform-Windows%2010-0078d6)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%205000%2072GB-76b900)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b9692-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.26.0%2Bcu132-purple)
![CUDA](https://img.shields.io/badge/CUDA-13.3-green)
![Tested](https://img.shields.io/badge/Tested-2026--08--15-yellow)

---

## 📌 核心结论（TL;DR）

| # | 结论 | 依据 |
|---|---|---|
| 1 | **agent 场景选 vLLM FP8** | TTFT 全尺寸碾压（1.4×~2.75×），而 TTFT 决定 agent 体感 |
| 2 | **长文本生成选 llama.cpp Q6_K** | decode 在 200K 上下文快 51%，权重质量最高 |
| 3 | **q8_0 KV 是白赚的** | 仅 ~1% 速度代价，省 50% KV 显存，并发 2→4 |
| 4 | **并发越高 GPU 越值** | 两并发总体吞吐比单流高 ~70%（batch 效率） |
| 5 | **NVFP4 在 Windows 暂不可用** | 工具链问题链（CUDA 13.3 × MSVC 14.44 C2719），详见 §8 |

---

## 🖥️ 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 5000 **72GB** Blackwell (sm_120, 73415 MiB) |
| CPU | Intel Core Ultra 7 270K Plus |
| 内存 | 127 GiB |
| 系统 | Windows 10 x64 (10.0.26200) |
| llama.cpp | b9692 (CUDA 13.3 构建) |
| vLLM | 0.26.0+cu132 (Python 3.12 + torch 2.11 cu130) |

## 🚀 三种部署方案

| 方案 | 模型 | 权重大小 | KV cache | 并发 | 实测显存 |
|---|---|---|---|---|---|
| **llama.cpp Q6_K** | Q6_K_XL.gguf + mmproj | 24.1 GiB | q8_0 | 4×256K | 64.2 GiB |
| **vLLM FP8** | 官方 FP8 (safetensors) | ~29 GB | 默认 | 动态 | 0.90 util |
| **vLLM NVFP4** ⚠️ | unsloth NVFP4 (safetensors) | 21.8 GiB | — | — | 部署受阻 (§8) |

> ⚠️ NVFP4 模型已完整下载校验，但 Windows 工具链无法编译其 FP4 内核，**未测出速度数据**，问题记录见 §8。

---

## 📏 测试方法论

- **prefill/TTFT**：OpenAI API 流式请求，`TTFT = 请求发出 → 首个 token`，`prefill t/s = prompt_tokens / TTFT`；每轮使用**随机文本**杜绝缓存命中
- **decode**：流式生成期间 tokens/s（剔除首 token）
- **纯内核 prefill**：llama-bench（不经 HTTP 层）
- **并发 profile**：N 线程同时发请求分别计时；`总体吞吐 = Σtokens / 窗口时间`
- 所有对比均在 GPU 空闲（无其他服务）状态下进行；关键数据 3 轮取均值

---

## 📊 测试结果

### 4.1 KV Cache 格式：f16 vs q8_0（llama.cpp，200K 上下文，各 3 轮均值）

| 指标 | f16 KV | q8_0 KV | 差异 |
|:---|---:|---:|---:|
| prefill 200K | 873.04 t/s | 862.00 t/s | **-1.26%** |
| decode @200K | 39.94 t/s | 39.68 t/s | **-0.66%** |
| 显存（2×256K） | 58.0 GiB | 44.9 GiB | **省 13.4 GiB** 🎉 |

> **结论：q8_0 KV 以 ~1% 速度代价换取 50% KV 显存，并发从 2 提到 4。**

### 4.2 单并发 Prefill / TTFT（同口径）

| Prompt | llama.cpp Q6_K | vLLM FP8 | 胜者 |
|:---|---:|---:|:---:|
| 2.7K TTFT | 3.60s | 2.55s | vLLM |
| 2.7K prefill | 749 t/s | 1,033 t/s | vLLM +38% |
| 37K TTFT | 23.8s | 10.2s | **vLLM 2.3×** |
| 37K prefill | 1,576 t/s | 3,671 t/s | **vLLM +133%** |
| 232K TTFT | 302s | 110s | **vLLM 2.7×** |
| 232K prefill | 768 t/s | 2,114 t/s | **vLLM +175%** |

> 注：llama.cpp 每请求有 ~2.5s 固定开销（调度/分词），小 prompt 的 prefill 速度被稀释。纯内核 prefill（llama-bench）：2.7K=2112 t/s、37K=1822 t/s——符合 attention 成本随上下文增长的理论。

### 4.3 单并发 Decode

| 上下文 | llama.cpp Q6_K | vLLM FP8 | 差距 |
|:---|---:|---:|:---:|
| ~2K | 39.7 t/s | 37.3 t/s | -6%（平手） |
| **200K** | 39.9 t/s | 26.4 t/s | **llama.cpp +51%** |

> decode 在常规上下文基本持平；llama.cpp 仅在极端长上下文（200K+）明显领先。

### 4.4 两并发 Decode Profile

| 指标 | llama.cpp Q6_K | vLLM FP8 |
|:---|---:|---:|
| 单流基线（1 并发） | 39.9 t/s | 37.3 t/s |
| 两并发·平均每流 | 34.1 t/s | 32.5 t/s |
| **两并发·总体吞吐** | **68.2 t/s** | 64.7 t/s |

> 💡 **关键发现**：两并发总体吞吐（~68 t/s）比单流（~40 t/s）高 **~70%** —— 批处理效率提升，对 agent 多请求场景是重大利好。

### 4.5 模型质量参考（量化格式）

| 格式 | 精度 | 相对 FP16 损失 |
|:---|---:|---|
| FP16/BF16 | 16-bit | 基线 |
| FP8 (e4m3) | 8-bit FP | ~0.2-0.5%，基本无损 |
| Q6_K | ~6.5-bit int | ~0.5-1%，近无损 |
| NVFP4 | 4-bit FP (e2m1) | ~1-2%，可测但不明显 |

---

## 🤖 Agent 场景：选谁？

| agent 工作负载特征 | vLLM FP8 | llama.cpp Q6_K |
|:---|:---:|:---:|
| 大量短请求（工具调用） | ✅ TTFT 快 28% | ⚠️ 每请求 ~2.5s 开销地板 |
| 子 agent 并发 | ✅ 动态批处理 | 固定槽位（4-8） |
| RAG 中长片段 (10-50K) | ✅ prefill 快 2.3× | 慢 |
| 长文档 (100K+) | ✅ prefill 快 2.75× | ❌ 最弱项 |
| 持续生成/长文写作 | ⚠️ decode -7% | ✅ decode 略快 |

**一句话**：agent 的体感等待 = TTFT，而 TTFT 之争 vLLM 全胜 → **vLLM 更适合 agent**；llama.cpp 在质量/长生成/极简运维上保留优势。

**实践建议**：
- vLLM：`--max-num-seqs` 提到 4-6，启动后发预热请求（首个请求冷启动可达 280s）
- llama.cpp：`--ctx-size 1048576 --parallel 4 --cache-type-k q8_0 --cache-type-v q8_0`
- agent 的 RAG 文档建议切片到 10-50K（200K prefill 需 110-302s，物理瓶颈）

---

## ⚠️ 踩坑记录（方法论陷阱）

1. **vLLM 启动后首个请求极慢**（280s+）—— CUDA graph/torch.compile 冷启动，需预热
2. **prefix cache 污染**——相同前缀的 prompt 会命中缓存，测得虚高 prefill（5794 t/s 是假的，真实 2114 t/s）
3. **GPU 抢占污染**——两服务同跑时 vLLM 降级运行，数据不可用
4. **vLLM 随机 seed**——思考模式行为不稳定，复现需固定 seed
5. **llama.cpp 小请求固定开销** ~2.5s/请求（TTFT 地板，与 prompt 长度无关）

---

## 🔧 NVFP4 部署受阻记录

模型已从 hf-mirror 完整下载（21.8 GiB，1968 权重条目校验通过），但 **vLLM-on-Windows 工具链问题链**导致 FP4 硬件内核无法编译：

| # | 问题 | 根因 | 处置 |
|:---:|---|---|---|
| 1 | C2719 "128 对齐形参" | CUDA 13.3 cudafe stub × MSVC 14.44 不兼容 | ❌ 无可用 MSVC 中间版本 |
| 2 | 14.29 无 C++20 | flashinfer JIT 内核需 C++17+ | ✅ 改 JIT 为 c++17 绕过 |
| 3 | cudafe++ 崩溃 | NVCC_APPEND_FLAGS 强推 c++20 | ❌ 放弃 |
| 4 | ParallelLMHead 缺属性 | vLLM 0.26 bug | ✅ 本地补丁（2 文件） |
| 5 | flashinfer 导入遮蔽 | 引擎子进程 sys.path 异常 | ❌ 补丁引发循环导入 |

**建议**：Linux 容器部署 NVFP4（无此问题链）；或等待 vLLM Windows 修复。本地补丁均有 `.bak` 备份且不影响 FP8 部署。

---

## 📁 仓库结构

```
qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark/
├── README.md              # 本报告
├── scripts/
│   ├── prefill_test.py    # prefill / TTFT 测试（随机文本防缓存）
│   ├── conc_test.py       # N 并发 decode profile 测试
│   └── start_qwen38q6_vision.bat  # llama.cpp Q6_K 部署脚本
```

## 🔄 复现方法

```bash
# prefill/TTFT（llama.cpp 或 vLLM 均可）
python scripts/prefill_test.py <API_BASE_URL> <model_name> <label>

# N 并发 decode profile
python scripts/conc_test.py <API_BASE_URL> <model_name> <N> <max_tokens> <label>
```

---

*测试与文档：Deep-AI-Evo · 模型：Qwen3.8-27B (Apache-2.0) · 数据基于单机实测，不同环境可能略有差异*
