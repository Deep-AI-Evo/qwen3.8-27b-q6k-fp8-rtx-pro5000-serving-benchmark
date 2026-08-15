# Qwen3.8-27B 部署方案实测报告

**日期**: 2026-08-15
**测试对象**: Qwen3.8-27B (qwen35 架构, 65 层混合注意力: 线性注意力 + 每 4 层一个全注意力, 含 MTP)

## 1. 环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell (sm_120, 73415 MiB) |
| CPU | Intel Core Ultra 7 270K Plus |
| 内存 | 127 GiB |
| OS | Windows 10 x64 (10.0.26200) |
| llama.cpp | b9692 (CUDA 13.3) |
| vLLM | 0.26.0+cu132 (Python 3.12 + torch 2.11 cu130) |

## 2. 部署配置

### 2.1 llama.cpp Q6_K (H:\start_qwen38q6_vision.bat)
- 模型: Qwen3.8-27B-UD-Q6_K_XL.gguf (24.1 GiB, unsloth, qwen35 架构)
- 视觉: mmproj-Qwen3.8-27B-BF16.gguf (931 MB, 从 ModelScope unsloth/Qwen3.8-27B-GGUF 下载)
- KV: q8_0 (--cache-type-k q8_0 --cache-type-v q8_0)
- 并发: 4 槽 × 256K (--ctx-size 1048576 --parallel 4)
- 实测显存: 64.2 GiB / 71.7 GiB

### 2.2 vLLM FP8 (start_qwen38_fp8_vllm.bat, 端口 12346)
- 模型: Qwen3.8-27B-FP8 (官方 FP8, ~29 GB, safetensors)
- vLLM 原生支持 Qwen3_5ForConditionalGeneration
- 配置: --max-model-len 262144 --gpu-memory-utilization 0.90 --max-num-seqs 2

### 2.3 vLLM NVFP4 (部署尝试)
- 模型: unsloth/Qwen3.8-27B-NVFP4 (21.8 GiB, NVFP4 + FP8 混合量化, mxfp4 格式)
- 来源: HuggingFace (hf-mirror.com 下载)
- 部署过程中的关键问题与修复见 §6

## 3. 测试方法论

- **prefill/TTFT**: OpenAI API 流式请求, 从请求发出到首个 token 的时间为 TTFT, prefill t/s = prompt_tokens / TTFT。每轮使用随机文本杜绝 prefix/prompt 缓存命中
- **decode**: 流式生成期间 tokens/s (排除首个 token)
- **llama-bench 纯 prefill**: 不经 HTTP 层的内核级测量
- **两并发 profile**: 2 线程同时发起流式请求, 分别计时; 总体吞吐 = Σtokens / (最晚结束 - 最早首token)
- 所有对比均在 GPU 空闲(无其他服务)状态下进行

## 4. 测试结果

### 4.1 KV cache 格式: f16 vs q8_0 (llama.cpp Q6_K, 200K 上下文, 3 轮均值)

| 指标 | f16 KV | q8_0 KV | 差异 |
|---|---|---|---|
| prefill 200K | 873.04 t/s | 862.00 t/s | -1.26% |
| decode 200K ctx | 39.94 t/s | 39.68 t/s | -0.66% |
| 显存 (2×256K) | 58.0 GiB | 44.9 GiB | 省 13.4 GiB |

结论: q8_0 KV 以 ~1% 速度代价换取 50% KV 显存, 使并发从 2 提高到 4。

### 4.2 单并发: prefill / TTFT (同口径, 随机文本)

| Prompt | llama.cpp Q6_K | vLLM FP8 | 胜者 |
|---|---|---|---|
| 2.7K TTFT | 3.6s (3轮: 3.56/3.63/3.60) | 2.6s (2.71/2.57/2.55) | vLLM |
| 2.7K prefill | 749 t/s | 1033 t/s | vLLM +38% |
| 37K TTFT | 23.8s (23.36/24.15) | 10.2s (10.19/10.20) | vLLM 2.3x |
| 37K prefill | 1576 t/s | 3671 t/s | vLLM +133% |
| 232K TTFT | 302s | 110s | vLLM 2.7x |
| 232K prefill | 768 t/s | 2114 t/s | vLLM +175% |

补充: llama.cpp 每请求有 ~2.3-2.9s 固定开销 (调度/tokenization), 小 prompt 的 prefill 速度被严重稀释。
纯内核 prefill (llama-bench): 2.7K=2112 t/s, 37K=1822 t/s — 符合 attention 成本随上下文增长的理论。

### 4.3 单并发: decode

| 上下文 | llama.cpp Q6_K | vLLM FP8 | 差异 |
|---|---|---|---|
| 短 (~2K) | 39.7 t/s | 37.3 t/s | -6% |
| 长 (200K) | 39.9 t/s (3轮均值) | 26.4 t/s | llama.cpp +51% |

decode 在常规上下文下基本持平; llama.cpp 仅在极端长上下文 (200K+) 下明显领先。

### 4.4 两并发 decode profile

| 指标 | llama.cpp Q6_K | vLLM FP8 |
|---|---|---|
| 单流 (1并发基线) | 39.9 t/s | 37.3 t/s |
| 两并发·平均每流 | 34.1 t/s | 32.5 t/s |
| 两并发·总体吞吐 | 68.2 t/s | 64.7 t/s |
| 两并发 TTFT (~11K, 热) | 2.5s | 3.5s |

关键发现: 两并发总体吞吐 (~68 t/s) 比单流 (~40 t/s) 高 ~70% — batch 效率提升, 对 agent 多请求场景是重大利好。

### 4.5 模型质量对比 (参考)

| 格式 | 精度 | 相对 FP16 损失 |
|---|---|---|
| FP16/BF16 | 16-bit | 基线 |
| FP8 (e4m3) | 8-bit FP | ~0.2-0.5%, 基本无损 |
| Q6_K (当前) | ~6.5-bit int | ~0.5-1%, 近无损 |
| NVFP4 | 4-bit FP (e2m1) | ~1-2%, 可测但不明显 |

## 5. 结论与建议

1. **agent 场景 (大量短请求 + 子 agent 并发): vLLM 更合适**
   - agent 体感 = TTFT, vLLM 全尺寸碾压 (1.4x-2.7x)
   - 短请求固定开销小; 动态批处理适合高并发
   - 建议 --max-num-seqs 4-6, 启动后发预热请求
2. **长文本生成/质量优先: llama.cpp Q6_K**
   - 权重质量最高, decode 在长上下文下更快
3. **并发与显存**
   - llama.cpp: q8_0 KV + 4×256K = 64.2 GiB (实测); 8×128K 也放得下 (~64 GiB)
   - vLLM: 0.90 util 下 KV 池 ~32 GiB (2×256K BF16 或 4×256K FP8 KV)
4. **长上下文 prefill 是物理瓶颈**: 200K prompt 处理需 110-302s, agent 的 RAG 文档建议切片到 10-50K

## 6. 已知问题与踩坑记录

1. **vLLM 启动后首个请求极慢** (280s+): CUDA graph/torch.compile 冷启动, 建议预热
2. **vLLM prefix cache 污染测试**: 相同前缀的 prompt 会命中缓存, 测得虚高 prefill (5794 t/s 是假的, 真实 2114 t/s)
3. **GPU 抢占污染测试**: llama.cpp 与 vLLM 同跑时显存不足, vLLM 降级运行, 数据不可用
4. **vLLM 每次请求随机 seed**: 思考模式行为不稳定, 测试需固定 seed
5. **llama.cpp 小请求固定开销** ~2.5s/请求 (TTFT 地板)
6. **NVFP4 部署问题** (Windows):
   - CUDA 13.3 + MSVC 14.44: C2719 错误 (128 对齐参数, MSVC 过新)
   - MSVC 14.29: 不支持 C++20, flashinfer FP4 JIT 编译失败
   - marlin 后端: vLLM 0.26 bug (ParallelLMHead.output_size_per_partition 缺失), 已本地打补丁
   - 正确方案: 安装 CUDA 13.3 官方支持的 MSVC 14.43 (待验证)

## 7. 附: 部署文件与脚本

- scripts/conc_test.py — 并发 decode profile 测试
- scripts/prefill_test.py — prefill/TTFT 测试
- scripts/start_qwen38q6_vision.bat — llama.cpp Q6_K 部署 (4×256K, q8_0 KV)

## 8. NVFP4 部署实测记录 (未完成 - 工具链受阻)

### 8.1 部署尝试
- 模型: unsloth/Qwen3.8-27B-NVFP4 (21.8 GiB) 已从 hf-mirror 完整下载并校验
- 格式: safetensors, NVFP4(mxfp4) + FP8 混合量化 (config 声明 compressed-tensors)
- 尝试路径: vLLM 0.26.0+cu132, 4 种后端组合

### 8.2 问题链 (vLLM-on-Windows 工具链)
| # | 问题 | 原因 | 状态 |
|---|---|---|---|
| 1 | C2719 "128 对齐的形参将不被对齐" | CUDA 13.3 cudafe 生成的 stub 与 MSVC 14.44 不兼容 (MSVC 过新) | 阻塞 cutlass/flashinfer FP4 JIT |
| 2 | MSVC 14.29 无 C++20 | flashinfer JIT 内核需 C++17+; nvcc 忽略 14.29 的 c++20 标志 | 已通过改 flashinfer JIT 为 c++17 绕过 |
| 3 | cudafe++ 崩溃 (0xC0000409) | NVCC_APPEND_FLAGS 强推 c++20 与 nvcc 冲突 | 放弃该路径 |
| 4 | marlin 后端 ParallelLMHead 缺 output_size_per_partition | vLLM 0.26 bug | 已本地补丁 (marlin_utils_fp8.py + marlin.py) |
| 5 | `from flashinfer import` 解析到 vllm 内部模块 | 引擎子进程 sys.path 含 scaled_mm 目录, 遮蔽真实 flashinfer 包 | 未能定位注入点; 补丁引发循环导入 |

### 8.3 结论
NVFP4 的硬件加速路径 (flashinfer/cutlass FP4 内核) 在**当前 Windows 工具链**下无法编译:
- MSVC 14.44 (VS 2022 17.13+) 与 CUDA 13.3 的 cudafe stub 不兼容 (C2719)
- 机器上无 CUDA 官方支持的中间版本 MSVC (14.43/14.42 安装被安装器静默忽略)
- vLLM 0.26 Windows 版对 NVFP4+FP8 混合模型的 marlin 路径有未修 bug

**建议**: 
1. 在 Linux 容器/服务器上部署 NVFP4 (无此工具链问题)
2. 或等待 vLLM Windows 版修复 + 新 MSVC/CUDA 组合
3. 或继续使用 FP8 (质量几乎无损, 速度已验证)

### 8.4 本地补丁清单 (均已备份 .bak)
- vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py (size_k_first 自适应)
- vllm/model_executor/kernels/linear/scaled_mm/marlin.py (apply_weights 属性兜底)
- flashinfer/jit/*.py (c++20 → c++17, 4 个文件)
- 注意: 以上补丁不影响 FP8 部署 (不同代码路径)

