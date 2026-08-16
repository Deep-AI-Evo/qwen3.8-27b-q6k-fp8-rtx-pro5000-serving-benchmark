# ⚡ Qwen3.8-27B 部署方案实测报告

> 🌐 **English** | [Switch to English](README.en.md)

> **llama.cpp (Q6_K) vs vLLM (FP8 vs NVFP4) 三方对比 · TTFT / Prefill / Decode / 并发**
> 基于 2026-08-15 实机测试，所有数据均可复现（测试脚本附于仓库）
>
> 📊 **三设备横向对比请看这里**：[Qwen3.8-27B 跨设备横向对比（DGX Spark / RTX PRO 5000 / RTX PRO 6000）](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)

![Platform](https://img.shields.io/badge/Platform-Windows%2010-0078d6)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%205000%2072GB-76b900)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b9692-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.26.0%2Bcu132-purple)
![CUDA](https://img.shields.io/badge/CUDA-13.3-green)
![Tested](https://img.shields.io/badge/Tested-2026--08--15-yellow)

---

## 📑 目录

- [核心结论（TL;DR）](#-核心结论tldr)
- [测试环境与三种部署方案](#%EF%B8%8F-测试环境)
- [测试结果（prefill/decode/并发/KV）](#-测试结果)
- [Agent 场景选型](#-agent-场景选谁)
- [踩坑记录](#-踩坑记录方法论陷阱)
- [跨设备横向对比（DGX Spark / PRO 5000 / PRO 6000）](#-跨设备横向对比dgx-spark--rtx-pro-5000--rtx-pro-6000) ⭐ 在文末

> ⭐ 想看 **DGX Spark / RTX PRO 5000 / RTX PRO 6000** 三设备谁强？文末有完整的[三设备横向对比](#-跨设备横向对比dgx-spark--rtx-pro-5000--rtx-pro-6000)（含 PRO 6000 FP8 全档位数据）。


---

## 📌 核心结论（TL;DR）

| # | 结论 | 依据 |
|---|---|---|
| 1 | **decode 王者：vLLM NVFP4 + MTP** | 单流 63.6 t/s（比 Q6_K 快 60%）；两并发总体 112.6 t/s |
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

| 方案 | 模型 | 权重 | KV cache | 后端 | MTP 投机解码 | 部署难度 |
|:---|:---|---:|:---|:---|:---|:---|
| **llama.cpp Q6_K** | Q6_K_XL.gguf + mmproj | 24.1 GiB | q8_0 | 原生 CUDA | `draft-mtp`（单流 +56%，长上下文无净损失） | ⭐ 双击即用 |
| **vLLM FP8** | 官方 FP8 | 28.8 GiB | 默认 | flashinfer | 支持（短中程 +16~35%，长上下文 -42% ⚠️） | ⭐⭐ 开箱即用 |
| **vLLM NVFP4** | unsloth NVFP4 | 21.8 GiB | 默认 | marlin(mxfp4) | 支持（**全场景加速**，decode 之王） | ⭐⭐⭐⭐ 需 3 个本地补丁（见 [NVFP4 部署记录](#-nvfp4-部署记录windows-已解决)） |

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

### 4.4 MTP 投机解码（Multi-Token Prediction，1 层 nextn）

> 启用方式：`--speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'`
>（vLLM 自动从同 checkpoint 加载 `mtp.*` 权重作为 drafter）

| 场景 | Q6_K 无 MTP | **Q6_K + MTP** | FP8 无 MTP | **FP8 + MTP** | NVFP4 无 MTP | **NVFP4 + MTP** |
|:---|---:|---:|---:|---:|---:|---:|
| 单流 ~11K | 39.7 | **61.9 (+56%)** | 37.3 | 43.2 (+16%) | 49.6 | **63.6 (+28%)** 🏆 |
| 两并发·平均每流 | 34.1 | 25.5 (-25%) ❌ | 32.5 | 44.0 (+35%) | 45.0 | **56.3 (+25%)** |
| 两并发·总体 | 68.2 | 42.3 (-38%) ❌ | 64.7 | 87.0 (+34%) | 88.8 | **112.6 (+27%)** 🏆 |
| 长上下文 ~148K | 39.9 | **40.0 (±0%)** | 26.4 | **15.3 (-42%)** ❌ | 42.1 | **57.8 (+37%)** 🏆 |

> ⚠️ **重要发现**：MTP 在 FP8 上**长上下文反而大幅变慢**（-42%，复测两次确认），
> 疑似 FP8 路径下 drafter 在长上下文的 KV/SSM 状态处理有缺陷（接受率坍塌，吞吐≈基线一半）；
> 而 **NVFP4 (marlin) 路径的 MTP 在长上下文依然 +37%**。
> 结论：**NVFP4 + MTP 是 decode 的终极形态**；FP8 + MTP 仅适合短/中上下文；llama.cpp MTP 单流大加速但并发退化。
>
> ⚠️ **跨设备复现警告（2026-08-16 更新）**：PRO 6000 仓库用**完全相同的软件栈**
> （vLLM 0.26.0 + marlin + n=1 + 相同启动参数）在 Linux 上复测，148K decode 仅 21.4 t/s
> （-53%），**未能复现本机 +37%**；n 值 / 后端 / vLLM 版本 / 调度参数逐一排除，
> 且其接受率全程健康（71~100%）——崩盘机制是 MTP 每步开销失控而非接受率坍塌。
> 差异指向 OS/驱动层（Windows vs Linux）。在更多设备验证前，57.8 t/s @148K 应视为
> **本机（Windows）单设备结果**，其他平台用前务必自测。详见
> [横向对比 MTP 专题](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)。

### 4.5 MTP 对 prefill / TTFT 的影响

> MTP 投机解码主要影响 decode；prefill 侧草稿模型是额外开销，实测影响很小：

| Prompt | Q6_K | Q6_K+MTP | FP8 | FP8+MTP | NVFP4 | NVFP4+MTP |
|:---|---:|---:|---:|---:|---:|---:|
| 2.7K prefill | 749 t/s | 706 (-6%) | 1033 | 1064 (+3%) | 915 | 931 (+2%) |
| 37K prefill | 1576 | 1556 (-1%) | 3671 | 3543 (-3%) | 2417 | 2325 (-4%) |
| 232K prefill | 768 | 734 (-4%) | 2114 | 2012 (-5%) | 1373 | 1305 (-5%) |
| 232K TTFT | 302s | 316s | 110s | 115s | 169s | 178s |

**结论：MTP 对 prefill 的影响在 ±5% 以内（TTFT 多出的部分主要是草稿模型 prefill），
prefill 排行不变：FP8 > NVFP4 > Q6_K。MTP 的全部收益/损失都发生在 decode 侧。**

### 4.6 MTP 长上下文衰减：根因与解决方案（社区调研）

> 对应 vLLM GitHub Issue **#47602**（[链接](https://github.com/vllm-project/vllm/issues/47602)）：
> *"Native MTP draft acceptance rate decays with total context length"*，与我们实测完全吻合。

**衰减曲线（issue 作者实测，Qwen3.6-27B，MTP n=6）**：
| 上下文 | MTP n=6 | 无 MTP | Δ |
|:---|---:|---:|---:|
| 2K | 138.0 t/s | 60.4 t/s | +129% |
| 8K | 88.3 | 60.1 | +47% |
| 16K | 51.0 | 59.1 | **-14%** |
| 30K | 26.8 | 54.6 | **-51%** |

接受率（position-1）：2K 时 93.5% → 30K 时 72.1%；整体平均 64.9% → 39.1%。

**根因（社区共识，跨引擎验证）**：
- MTP 草稿头只有 **1 层**（`mtp_num_hidden_layers=1`），目标模型有 64 层
- 短上下文：目标最终 hidden state 已编码足够信息，1 层草稿头够用
- 长上下文：要生成与远处上下文一致的续写，需要**长程多跳 attention**——单层草稿头没有这个容量，猜测越来越偏离 64 层目标的真实分布
- DeepSeek 官方在 DSpark 中正是因此弃用 MTP-1（"as context grows, acceptance of draft tokens decreases"）
- **不是 vLLM 实现 bug**：llama.cpp (Vulkan/AMD) + vLLM (CUDA/NVIDIA) 上同样复现（issue 评论 + 本机实测）

**解决方案 / 规避**：
| 方案 | 效果 | 备注 |
|---|---|---|
| **vLLM NVFP4 + MTP** | 长上下文无衰减（+37%） | 本机（Windows）实测；⚠️ PRO 6000（Linux）同栈复测 -53%，指向 OS/驱动层差异，见 §4.4 复现警告 |
| **llama.cpp draft-mtp（n_max=4）** | 单流 +56%，长上下文持平（**无净损失**） | 本机实测；两并发下反而退化（-38%） |
| vLLM FP8 + MTP | 短程 +16%，长程 -42% | 长上下文**禁用** MTP 更优 |
| `num_speculative_tokens_per_batch_size` | 按**批大小**动态调 n | 不是按上下文长度，无法直接解决 |
| 等待 vLLM 实现按长度分桶开关 | issue #47602 提案中 | 尚未合并 |

**实践建议**：长上下文场景（>16K）在本机（Windows）可用 NVFP4+MTP，其他平台（Linux 实测未复现）
建议关闭 MTP；短中程 MTP 收益明显。

### 4.7 KV Cache 格式：f16 vs q8_0（llama.cpp，200K 上下文，3 轮均值）

| 指标 | f16 KV | q8_0 KV | 差异 |
|:---|---:|---:|---:|
| prefill 200K | 873.04 t/s | 862.00 t/s | **-1.26%** |
| decode @200K | 39.94 t/s | 39.68 t/s | **-0.66%** |
| 显存（2×256K） | 58.0 GiB | 44.9 GiB | **省 13.4 GiB** 🎉 |

> **结论：q8_0 KV 以 ~1% 速度代价换取 50% KV 显存，并发从 2 提到 4。**

### 4.8 模型质量参考（量化格式）

| 格式 | 精度 | 相对 FP16 损失 |
|:---|---:|---|
| FP16/BF16 | 16-bit | 基线 |
| FP8 (e4m3) | 8-bit FP | ~0.2-0.5%，基本无损 |
| Q6_K | ~6.5-bit int | ~0.5-1%，近无损 |
| NVFP4 | 4-bit FP (e2m1) | ~1-2%，可测但不明显 |

---

---

## 🤖 Agent 场景：选谁？

| agent 工作负载特征 | vLLM FP8 | vLLM NVFP4 | llama.cpp Q6_K |
|:---|:---:|:---:|:---:|
| 大量短请求（工具调用） | ✅ TTFT 最快 | ✅ TTFT 接近 | ⚠️ ~2.5s 开销地板 |
| 子 agent 并发生成 | ✅ 总体 64.7 | ✅ **总体 88.8** | ✅ 总体 68.2 |
| RAG 长片段 prefill | ✅ **最快** | 中 | 慢 |
| 长文写作 decode | ⚠️ 26-43 t/s（+MTP 短中程 43，长程 15 ❌） | ✅ **58-64 t/s（+MTP 全场景加速）** | ✅ 39-40 t/s |
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
│   ├── run_vllm_nvfp4_bench.bat   # vLLM NVFP4 部署（marlin）
│   ├── run_vllm_fp8_mtp_bench.bat    # vLLM FP8 + MTP 投机解码
│   └── run_vllm_nvfp4_mtp_bench.bat  # vLLM NVFP4 + MTP 投机解码
```

## 🔄 复现方法

```bash
# prefill/TTFT（三个服务均可）
python scripts/prefill_test.py <API_BASE_URL> <model_name> <label>

# N 并发 decode profile
python scripts/conc_test.py <API_BASE_URL> <model_name> <N> <max_tokens> <label>
```

---

## 🆚 跨设备横向对比（DGX Spark / RTX PRO 5000 / RTX PRO 6000）

> 摘要自 [三设备横向对比](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)（完整档位与测试方法见该文档）。
> ⚠️ 口径差异：三台设备 vLLM 版本 / MTP 设置 / 操作系统不完全相同，数量级参考意义大于精确对比。

**单并发 decode（tok/s）**

| 上下文 | DGX Spark NVFP4+MTP×3 | PRO 5000 FP8 无MTP → +MTP（本仓库） | PRO 5000 NVFP4 无MTP → +MTP（本仓库） | PRO 5000 Q6_K 无MTP → +MTP（本仓库） | PRO 6000 NVFP4 无MTP → +MTP(n=2) | PRO 6000 Q6_K |
|:---|---:|---:|---:|---:|---:|---:|
| 短（~1-11K） | ~21 | 37.3 → 43.2 | 49.6 → 63.6 | 39.7 → 61.9 | 58.6 → **100.2** | 55.4 |
| ~148K | — | 26.4 → 15.3 ❌ | 42.1 → **57.8** ✅（PRO 6000 同栈复测仅 21.4 ❌，见 §4.4） | 39.9 → 40.0 | ≈46* → 21.4 ❌ | — |
| ~200K | 14.2 | 26.4（无MTP） | 42.1（无MTP） | 39.9 | **43.7** → 18.2 ❌ | 35.7 |

\* PRO 6000 无 MTP 148K 未单测，为 100K/200K 插值。

**单并发 prefill ~200K（tok/s）/ TTFT（s）**

| DGX Spark NVFP4 | PRO 5000 FP8（本仓库） | PRO 5000 Q6_K（本仓库） | **PRO 6000 FP8+MTP** | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K |
|---|---|---|---|---|---|
| 840 / 244s | 2,114 / 110s | 768 / 302s | 3,869 / 45.8s | **4,447 / 39.9s** | 1,676 / 105.9s |

**并发 decode 聚合吞吐（tok/s）**

| 并发 | DGX Spark NVFP4 | PRO 5000 FP8（本仓库） | PRO 5000 NVFP4（本仓库） | **PRO 5000 NVFP4+MTP（本仓库）** | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K（4 槽） |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.0 | 37.3 | 49.6 | **63.6** | 78.8 | 95.0 | 49.6 |
| 2 | 22.7 | 64.7 | 88.8 | **112.6** | 137.0 | 151.3 | 92.8 |
| 4 | 44.0 | — | — | — | 299.3 | 346.5 | 158.5 |
| 8 | 77.7 | — | — | — | 556.2 | **654.1** | — |

> 注：PRO 5000 仅测到 2 并发（72GB 卡，4 并发需降低单槽上下文）；PRO 6000 96GB 可到 8 并发。

一句话：**PRO 6000 在 prefill/TTFT/并发上全面领先；长上下文 decode 的最高报告值是本仓库的
NVFP4+MTP 57.8 t/s @148K（Windows），但 PRO 6000（Linux）用完全相同软件栈复测仅 21.4 t/s——
n 值/后端/版本/调度参数逐一排除、接受率全程健康，差异指向 OS/驱动层**。
FP8+MTP 在两台设备长上下文均为负优化——用前实测自己的配置（详见本仓库 §4.4/§4.5）。

---

## 🔗 相关仓库

- [RTX PRO 6000 FP8 vs NVFP4 benchmark](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark) — 含 [DGX Spark / PRO 5000 / PRO 6000 三设备横向对比](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
- [DGX Spark NVFP4 部署教程](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)

---

*测试与文档：Deep-AI-Evo · 模型：Qwen3.8-27B (Apache-2.0) · 数据基于单机实测，不同环境可能略有差异*
