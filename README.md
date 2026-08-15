# ⚡ Qwen3.8-27B 部署方案实测报告

> **llama.cpp (Q6_K) vs vLLM (FP8 vs NVFP4) 三方对比 · TTFT / Prefill / Decode / 并发**
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
| 1 | **decode 王者：vLLM NVFP4** | 单流 49.6 t/s（比 Q6_K 快 25%）；两并发总体 88.8 t/s |
| 2 | **prefill 王者：vLLM FP8** | 200K prefill 2114 t/s（比 NVFP4 快 54%） |
| 3 | **agent 场景首选：vLLM（FP8 或 NVFP4）** | TTFT 全尺寸领先，决定 agent 体感 |
| 4 | **llama.cpp 的价值：质量 + 极简运维** | Q6_K 权重精度最高（~6.5-bit），单 exe 部署 |
| 5 | **q8_0 KV 是白赚的** | 仅 ~1% 速度代价，省 50% KV 显存，并发 2→4 |
| 6 | **并发越高 GPU 越值** | 两并发总体吞吐比单流高 ~70%（batch 效率） |

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

| 方案 | 模型 | 权重 | KV cache | 后端 | 部署难度 |
|---|---|---|---|---|---|
| **llama.cpp Q6_K** | Q6_K_XL.gguf + mmproj | 24.1 GiB | q8_0 | 原生 CUDA | ⭐ 双击即用 |
| **vLLM FP8** | 官方 FP8 | ~29 GB | 默认 | flashinfer | ⭐⭐ 开箱即用 |
| **vLLM NVFP4** | unsloth NVFP4 | 21.8 GiB | 默认 | marlin(mxfp4) | ⭐⭐⭐⭐ 需 3 个本地补丁 (§8) |

---

## 📏 测试方法论

- **prefill/TTFT**：OpenAI API 流式请求，`TTFT = 请求发出 → 首个 token`，`prefill t/s = prompt_tokens / TTFT`；每轮使用**随机文本**杜绝缓存命中
- **decode**：流式生成期间 tokens/s（剔除首 token）
- **并发 profile**：N 线程同时发请求分别计时；`总体吞吐 = Σtokens / 窗口时间`
- 所有对比均在 GPU 空闲（无其他服务）状态下进行；关键数据多轮取均值

---

## 📊 测试结果

### 4.1 单并发 Prefill / TTFT

| Prompt | llama.cpp Q6_K | vLLM FP8 | vLLM NVFP4 |
|:---|---:|---:|---:|
| 2.7K TTFT | 3.60s | 2.55s | 2.90s |
| 2.7K prefill | 749 t/s | 1,033 t/s | 915 t/s |
| 37K TTFT | 23.8s | 10.2s | 15.5s |
| 37K prefill | 1,576 t/s | **3,671 t/s** | 2,417 t/s |
| 232K TTFT | 302s | **110s** | 169s |
| 232K prefill | 768 t/s | **2,114 t/s** | 1,373 t/s |

> 注：llama.cpp 每请求有 ~2.5s 固定开销（调度/分词），小 prompt 的 prefill 速度被稀释。纯内核 prefill（llama-bench）：2.7K=2112 t/s、37K=1822 t/s——符合 attention 成本随上下文增长的理论。

### 4.2 单并发 Decode

| 上下文 | llama.cpp Q6_K | vLLM FP8 | vLLM NVFP4 |
|:---|---:|---:|---:|
| ~11K | 39.7 t/s | 37.3 t/s | **49.6 t/s** 🏆 |
| ~150-200K | 39.9 t/s | 26.4 t/s | **42.1 t/s** 🏆 |

> 💡 **NVFP4 的 FP4 张量核优势兑现**：decode 全面登顶——短上下文比 Q6_K 快 25%，长上下文比 FP8 快 60%。

### 4.3 两并发 Decode Profile

| 指标 | llama.cpp Q6_K | vLLM FP8 | vLLM NVFP4 |
|:---|---:|---:|---:|
| 单流基线 | 39.9 t/s | 37.3 t/s | **49.6 t/s** |
| 两并发·平均每流 | 34.1 t/s | 32.5 t/s | **45.0 t/s** |
| **两并发·总体吞吐** | 68.2 t/s | 64.7 t/s | **88.8 t/s** 🏆 |

> 💡 两并发总体吞吐（88.8 t/s）比单流（49.6 t/s）高 **~80%** —— 批处理效率提升，对 agent 多请求场景是重大利好。

### 4.4 KV Cache 格式：f16 vs q8_0（llama.cpp，200K 上下文，3 轮均值）

| 指标 | f16 KV | q8_0 KV | 差异 |
|:---|---:|---:|---:|
| prefill 200K | 873.04 t/s | 862.00 t/s | **-1.26%** |
| decode @200K | 39.94 t/s | 39.68 t/s | **-0.66%** |
| 显存（2×256K） | 58.0 GiB | 44.9 GiB | **省 13.4 GiB** 🎉 |

> **结论：q8_0 KV 以 ~1% 速度代价换取 50% KV 显存，并发从 2 提到 4。**

### 4.5 模型质量参考（量化格式）

| 格式 | 精度 | 相对 FP16 损失 |
|:---|---:|---|
| FP16/BF16 | 16-bit | 基线 |
| FP8 (e4m3) | 8-bit FP | ~0.2-0.5%，基本无损 |
| Q6_K | ~6.5-bit int | ~0.5-1%，近无损 |
| NVFP4 | 4-bit FP (e2m1) | ~1-2%，可测但不明显 |

---

## 🤖 Agent 场景：选谁？

| agent 工作负载特征 | vLLM FP8 | vLLM NVFP4 | llama.cpp Q6_K |
|:---|:---:|:---:|:---:|
| 大量短请求（工具调用） | ✅ TTFT 最快 | ✅ TTFT 接近 | ⚠️ ~2.5s 开销地板 |
| 子 agent 并发生成 | ✅ 总体 64.7 | ✅ **总体 88.8** | ✅ 总体 68.2 |
| RAG 长片段 prefill | ✅ **最快** | 中 | 慢 |
| 长文写作 decode | ⚠️ 26-37 t/s | ✅ **42-50 t/s** | ✅ 39-40 t/s |
| 权重质量 | 近无损 | 4-bit（~1-2% 损失） | **最高** |

**一句话**：**vLLM NVFP4 是综合最优解**——decode 最快、并发吞吐最高、权重最小（21.8 GiB）；FP8 仅在长文档 prefill 单项领先；llama.cpp Q6_K 在质量与运维简单性上保留价值。

**实践建议**：
- NVFP4/FP8：`--max-num-seqs` 提到 4-6，启动后发预热请求（首个请求冷启动可达 280s）
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

## 🔧 NVFP4 部署记录（Windows，已解决 ✅）

模型从 hf-mirror 下载（21.8 GiB，1968 权重条目校验通过）。vLLM-on-Windows 工具链问题链，**通过 3 个本地补丁全部解决**：

| # | 问题 | 根因 | 解决 |
|:---:|---|---|---|
| 1 | C2719 "128 对齐形参" | CUDA 13.3 cudafe stub × MSVC 14.44 不兼容 | 换 marlin 后端（预编译内核，无需 JIT） |
| 2 | ParallelLMHead 缺 `output_size_per_partition` | vLLM 0.26 bug（marlin 路径） | 补丁 ① `marlin_utils_fp8.py`：size_k_first 自适应 |
| 3 | 同上（forward 路径） | 同上 | 补丁 ② `scaled_mm/marlin.py`：用 `num_embeddings_per_partition`/`embedding_dim` 兜底 |
| 4 | `from flashinfer import` 解析到 vLLM 内部模块 | 引擎子进程 sys.path 遮蔽真实 flashinfer | 补丁 ③ `site-packages/sitecustomize.py`：进程启动时预导入真包 |

**部署命令**：`scripts/run_vllm_nvfp4_bench.bat`（marlin 后端 + 3 补丁后）

**性能备注**：NVFP4 走 marlin(mxfp4) 后端（非 flashinfer FP4 JIT 路径），实测速度见 §4——decode 全面领先，FP4 张量核优势完整兑现。

---

## 📁 仓库结构

```
qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark/
├── README.md              # 本报告
├── scripts/
│   ├── prefill_test.py    # prefill / TTFT 测试（随机文本防缓存）
│   ├── conc_test.py       # N 并发 decode profile 测试
│   ├── start_qwen38q6_vision.bat  # llama.cpp Q6_K 部署
│   └── run_vllm_nvfp4_bench.bat   # vLLM NVFP4 部署（marlin）
```

## 🔄 复现方法

```bash
# prefill/TTFT（三个服务均可）
python scripts/prefill_test.py <API_BASE_URL> <model_name> <label>

# N 并发 decode profile
python scripts/conc_test.py <API_BASE_URL> <model_name> <N> <max_tokens> <label>
```

---

## 🔗 相关仓库

- [RTX PRO 6000 FP8 vs NVFP4 benchmark](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark) — 含 [DGX Spark / PRO 5000 / PRO 6000 三设备横向对比](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
- [DGX Spark NVFP4 部署教程](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)

---

*测试与文档：Deep-AI-Evo · 模型：Qwen3.8-27B (Apache-2.0) · 数据基于单机实测，不同环境可能略有差异*
