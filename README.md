# ⚡ Qwen3.8-27B 部署方案实测报告

> 🌐 **English** | [Switch to English](README.en.md)

> **llama.cpp / vLLM / SGLang 三方引擎 × Q6_K / FP8 / NVFP4 / FP4 四种权重 × MTP / DSPARK 两种投机解码**
> 基于 2026-08-15~17 实机测试，所有数据可复现（测试脚本附于仓库）
>
> 📊 **三设备横向对比**：[DGX Spark / RTX PRO 5000 / RTX PRO 6000](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)

![Platform](https://img.shields.io/badge/Platform-Windows%2010-0078d6)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%205000%2072GB-76b900)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b9692-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.26.0%2Bcu132-purple)
![SGLang](https://img.shields.io/badge/SGLang-qwen38.27b-blue)
![Tested](https://img.shields.io/badge/Tested-2026--08--17-yellow)

---

## 📌 核心结论（TL;DR）

| # | 结论 | 关键数字 |
|---|---|---|
| 1 | **综合王者：SGLang + NVFP4 + DSPARK** | decode 85.2 t/s（短）/ 71.4（200K）；两并发 147.6 t/s；prefill 6000+ t/s |
| 2 | **DSPARK 解决 MTP 长上下文衰减** | 200K decode 71.4 t/s 不衰减（vLLM FP8+MTP 仅 15.3 ❌） |
| 3 | **prefill 差距是引擎级的** | SGLang 6000+ t/s = vLLM 的 6 倍 = llama.cpp 的 2 倍 |
| 4 | **Windows 原生首选 vLLM**（无需 Docker） | NVFP4+MTP 63.6 t/s，开箱即用 |
| 5 | **质量/极简运维：llama.cpp Q6_K** | ~6.5-bit 近无损，单 exe 部署 |
| 6 | **FP4 GGUF 是 llama.cpp 的提速器** | NVFP4 GGUF+MTP 68.1 t/s（Q6_K 的 1.7 倍） |
| 7 | **锁 270W 无性能损失** | 实测峰值功耗 244W < 270W，温度 73°C 健康 |
| 8 | **显存纪律**：0.80 util / 单容器 | 超分导致 GUI/RDP 卡死（实测踩坑） |

---

## 🖥️ 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 5000 **72GB** Blackwell (sm_120, 73415 MiB, 270W 锁功率) |
| CPU | Intel Core Ultra 7 270K Plus |
| 系统 | Windows 10 x64 (10.0.26200) + WSL2 + Docker Desktop |
| llama.cpp | b9692 (CUDA 13.3, 原生 FP4) |
| vLLM | 0.26.0+cu132 (Python 3.12 + torch 2.11 cu130) |
| SGLang | lmsysorg/sglang:qwen38-27b 镜像 (Docker) |

## 🚀 部署方案全景

| 方案 | 权重 | 投机解码 | 部署方式 | 显存 |
|:---|:---|---:|:---|:---:|
| **SGLang NVFP4+DSPARK** | RadixArk NVFP4 20.4 GiB | DSPARK (γ=7) | Docker | ~57 GiB |
| **SGLang FP8+DSPARK** | 官方 FP8 28.8 GiB | DSPARK (γ=7) | Docker | ~65 GiB |
| **vLLM NVFP4+MTP** | unsloth NVFP4 21.8 GiB | MTP (n=1) | 原生 bat | 57 GiB (0.80) |
| vLLM FP8+MTP | 官方 FP8 28.8 GiB | MTP (n=1) | 原生 bat | ~65 GiB |
| **llama.cpp FP4 GGUF+MTP** | esatapedico GGUF 18.3 GiB | draft-mtp | 原生 bat | ~55 GiB |
| llama.cpp Q6_K | Q6_K_XL 24.1 GiB | draft-mtp | 原生 bat | ~50 GiB |

> SGLang 需 Docker（Windows 下）；vLLM/llama.cpp 可原生运行。
> ⚠️ SGLang 与 vLLM 不能同时跑（显存超分），详见 §5。

---

## 📏 测试方法论

- **prefill/TTFT**：流式 API，`TTFT = 请求发出→首 token`，`prefill t/s = prompt_tokens/TTFT`；随机文本防缓存
- **decode**：流式生成 tokens/s（剔除首 token）
- **并发**：N 线程同时请求；`总体 = Σtokens / 窗口`
- SGLang 容器内测速（Docker 端口转发在 WSL2 下可能失效）
- 所有对比 GPU 空闲状态；关键数据多轮取均值

---

## 📊 测试结果（全方案主对比）

### 4.1 Prefill / TTFT

| Prompt | SGLang NVFP4+DSPARK | SGLang FP8+DSPARK | vLLM NVFP4+MTP | vLLM FP8+MTP | llama.cpp FP4+MTP | llama.cpp Q6_K |
|:---|---:|---:|---:|---:|---:|---:|
| 2.7K | **5,388-6,772** | 4,858-5,030 | 931 | 1,064 | 706 | 749 |
| 37K | **5,221-5,249** | 4,306-4,309 | 2,325 | 3,543 | 1,556 | 1,576 |
| 232K | 1,900 | — | 1,305 | 2,012 | 734 | 768 |

> SGLang 的 prefill 优势是引擎级（混合 GDN 优化 + chunked 2048）：**2-6 倍于其他引擎**。

### 4.2 Decode（单流）

| 上下文 | SGLang NVFP4+DSPARK | SGLang FP8+DSPARK | llama.cpp FP4+MTP | vLLM NVFP4+MTP | vLLM FP8+MTP | llama.cpp Q6_K+MTP |
|:---|---:|---:|---:|---:|---:|---:|
| ~11K | **85.2** 🏆 | 64.3 | 68.1 | 63.6 | 43.2 | 61.9 |
| ~148-200K | **71.4** 🏆 | 52.7 | 42.2 | 57.8 | 15.3 ❌ | 40.0 |

> **DSPARK 是唯一长上下文不衰减的投机方案**（置信度头动态控制草稿长度）；MTP-1 在长上下文接受率坍塌（vLLM FP8+MTP 只剩 15.3）。

### 4.3 两并发 Decode Profile

| 指标 | SGLang NVFP4+DSPARK | vLLM NVFP4+MTP | SGLang FP8+DSPARK | vLLM FP8+MTP | llama.cpp FP4+MTP |
|:---|---:|---:|---:|---:|---:|
| 平均每流 | **76.8** | 56.3 | 55.7 | 44.0 | 35.6 |
| 总体吞吐 | **147.6** | 112.6 | 101.8 | 87.0 | 66.1 |

### 4.4 MTP vs DSPARK 投机解码对比（含社区调研）

| 维度 | MTP (n=1) | **DSPARK (γ=7)** |
|:---|:---|:---|
| 草稿来源 | checkpoint 内 1 层头 | 独立 1.36B 草稿 + 目标辅助特征 |
| 短上下文收益 | +16~28% | **+72%（NVFP4 49.6→85.2）** |
| 长上下文 (200K) | 衰减（FP8 路径 -42% ❌） | **不衰减（71.4 t/s）** |
| 草稿大小 | 内嵌 | 2.53 GiB |
| 根因 | 1 层头容量不足（[vLLM #47602](https://github.com/vllm-project/vllm/issues/47602)） | 置信度头动态调整草稿数，长上下文保持接受率 |

### 4.5 KV Cache 与显存

| 配置 | KV 格式 | 显存 | 说明 |
|:---|:---|:---:|:---|
| llama.cpp q8_0 KV | 8-bit | 省 50% | 速度损失 ~1%（实测 200K 3 轮均值） |
| vLLM fp8_e4m3 KV | 8-bit | KV 池容量翻倍 | fp8_e5m2 与 fp8 检查点不兼容 ⚠️ |
| SGLang mamba ratio 0.7 | — | — | 控制 KV/mamba 状态池分配 |

**显存纪律（实测教训）**：
- vLLM `--gpu-memory-utilization 0.80`：总占用 57 GiB，GUI 余量 14.7 GiB（0.90 导致界面/RDP 卡死）
- SGLang 容器**一次只跑一个**（两个容器 2×0.8 超分，FP8 被压到 available 0）
- 270W 功率限制：实测峰值 244W 未触顶，速度/温度无影响（73°C）

### 4.6 量化格式质量参考

| 格式 | 精度 | 相对 FP16 |
|:---|---:|---|
| FP16/BF16 | 16-bit | 基线 |
| FP8 (e4m3) | 8-bit | ~0.2-0.5% |
| Q6_K | ~6.5-bit | ~0.5-1% |
| NVFP4 | 4-bit FP | ~1-2% |
| FP4 GGUF | 4-bit | ~1-2% |

---

## 🤖 Agent 场景选型（重写）

| 场景 | 首选 | 关键数字 | 备选 |
|:---|:---|:---|:---|
| 交互/工具调用（TTFT 敏感） | **SGLang NVFP4+DSPARK** | TTFT 0.4-0.5s @2.7K | vLLM NVFP4+MTP |
| 高并发子 agent | **SGLang NVFP4+DSPARK** | 2 并发 147.6 t/s | vLLM NVFP4+MTP |
| 长文档 decode（>30K） | **SGLang NVFP4+DSPARK** | 71.4 t/s @200K | vLLM NVFP4+MTP 57.8 |
| RAG 灌入（prefill 敏感） | **SGLang（任一）** | 6000+ t/s | vLLM FP8 |
| 质量优先/离线 | llama.cpp Q6_K | 近无损 | vLLM FP8 |
| 不想装 Docker | vLLM NVFP4+MTP | 63.6 t/s 开箱即用 | — |

**一句话**：**SGLang NVFP4+DSPARK 是当前最优**；追求简单运维选 vLLM NVFP4+MTP；追求极致质量选 llama.cpp Q6_K。

---

## ⚠️ 踩坑记录（方法论陷阱 + 部署坑）

1. **vLLM 首请求冷启动 280s+** → 启动后预热
2. **prefix cache 污染测速**（5794 t/s 是假的）
3. **GPU 超分卡死**：vLLM 0.90 / 双 SGLang 容器 → GUI/RDP 冻结（DWM 在独显）
4. **WSL2 CUDA IPC 失效**：SGLang 需 `--mm-feature-transport cpu`
5. **Docker 端口转发失效**：容器内 `docker exec` 测速（`MSYS_NO_PATHCONV=1`）
6. **bat 中文注释 GBK 闪退** → 纯 ASCII
7. **后台任务启动服务进 Session 0 杀不掉** → 双击启动
8. **fp8_e5m2 KV 与 fp8 检查点不兼容** → 用 fp8_e4m3
9. **llama.cpp 每请求 ~2.5s 固定开销**（TTFT 地板）

---

## 🔧 NVFP4 部署记录（Windows）

vLLM marlin 路径需 3 个本地补丁（sitecustomize / marlin_utils_fp8 / marlin）；SGLang 走 Docker 无此问题。
**完整 SGLang 部署教程**：[docs/SGLang-DSPARK-Windows-部署教程.md](docs/SGLang-DSPARK-Windows-部署教程.md)

---

## 📁 仓库结构

```
├── README.md / README.en.md
├── docs/SGLang-DSPARK-Windows-部署教程.md
└── scripts/
    ├── conc_test.py / prefill_test.py      # 测速
    ├── start_qwen38q6_vision.bat           # llama.cpp Q6_K
    ├── start_qwen38_nvfp4_gguf_mtp.bat     # llama.cpp FP4 GGUF+MTP
    ├── start_qwen38_nvfp4_gguf_orig.bat    # llama.cpp FP4 GGUF ORIG
    ├── run_vllm_nvfp4_mtp_bench.bat        # vLLM NVFP4+MTP（0.80 + fp8 KV）
    ├── run_vllm_fp8_mtp_bench.bat          # vLLM FP8+MTP
    ├── start_sglang_nvfp4_dspark.ps1       # SGLang NVFP4+DSPARK
    └── start_sglang_fp8_dspark.ps1         # SGLang FP8+DSPARK
```

## 🔄 复现方法

```bash
# prefill/TTFT、并发 decode（三引擎通用；SGLang 在容器内执行）
python scripts/prefill_test.py <API_BASE_URL> <model_name> <label>
python scripts/conc_test.py <API_BASE_URL> <model_name> <N> <max_tokens> <label>
```

---

## 🆚 跨设备横向对比（DGX Spark / PRO 5000 / PRO 6000）——重写

> 完整版见 [PRO 6000 仓库对比文档](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
> ⚠️ 引擎版本差异大（SGLang / vLLM 0.26 / vLLM 0.21 / llama.cpp），数量级参考

**单流 decode（tok/s）**

| 上下文 | DGX Spark NVFP4+MTP×3 | **PRO 5000 SGLang NVFP4+DSPARK** | PRO 5000 vLLM NVFP4+MTP | PRO 6000 vLLM NVFP4+MTP(n=2) | PRO 6000 Q6_K |
|:---|---:|---:|---:|---:|---:|
| 短 (~1-11K) | ~21 | 85.2 | 63.6 | **100.2** | 55.4 |
| ~148-200K | 14.2 | **71.4** 🏆 | 57.8 | 43.7（无MTP） | 35.7 |

**prefill 200K（tok/s）/ TTFT（s）**

| DGX Spark | **PRO 5000 SGLang NVFP4+DSPARK** | PRO 5000 vLLM FP8 | PRO 6000 vLLM NVFP4+MTP |
|---|---:|---:|---:|
| 840 / 244s | 1,900 / 122s | 2,114 / 110s | **4,447 / 39.9s** |

**并发总体（tok/s）**

| 并发 | PRO 5000 SGLang DSPARK | PRO 6000 vLLM NVFP4+MTP | PRO 6000 Q6_K(4槽) |
|:---:|---:|---:|---:|
| 1 | 85.2 | 95.0 | 49.6 |
| 2 | 147.6 | **151.3** | 92.8 |
| 4 | — | 346.5 | 158.5 |
| 8 | — | **654.1** | — |

**新结论（重写）**：

1. **引擎优化 > 硬件差距**：PRO 5000（72GB）+ SGLang DSPARK 在长上下文 decode（71.4 vs 43.7）**大幅反超** PRO 6000（96GB）+ vLLM——投机解码选对，小卡反超大卡
2. **PRO 6000 保留优势**：短上下文峰值（100.2 vs 85.2）、并发扩展（8 路 654）
3. **DGX Spark 的价值**：128GB 统一内存 + 低功耗桌面形态，绝对速度不在一个量级
4. **预判**：PRO 6000 若也上 SGLang DSPARK，短/长/并发预计全面压制（未测，权重更大 + 引擎更好）
5. PRO 5000 短 prefill 6000+ t/s 为三设备最高（SGLang 引擎优化）

---

## 🔗 相关仓库

- [RTX PRO 6000 benchmark（含三设备横向对比）](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark)
- [DGX Spark NVFP4 部署教程](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)
- [LMSYS Qwen3.8-27B cookbook（SGLang 参考）](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)

---

*测试与文档：Deep-AI-Evo · 模型：Qwen3.8-27B (Apache-2.0) · 单机实测，环境不同结果可能略有差异*
