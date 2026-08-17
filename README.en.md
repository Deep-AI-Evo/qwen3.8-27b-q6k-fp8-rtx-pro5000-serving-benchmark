# ⚡ Qwen3.8-27B Deployment Benchmark Report

> 🌐 **中文** | [切换到中文](README.md)

> **3 engines (llama.cpp / vLLM / SGLang) × 4 weight formats (Q6_K / FP8 / NVFP4 / FP4) × 2 speculative methods (MTP / DSPARK)**
> Measured 2026-08-15~17 on real hardware; reproducible (scripts in repo)
>
> 📊 **Cross-device comparison**: [DGX Spark / RTX PRO 5000 / RTX PRO 6000](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)

![Platform](https://img.shields.io/badge/Platform-Windows%2010-0078d6)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%205000%2072GB-76b900)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b9692-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.26.0%2Bcu132-purple)
![SGLang](https://img.shields.io/badge/SGLang-qwen38.27b-blue)
![Tested](https://img.shields.io/badge/Tested-2026--08--17-yellow)

---

## 📌 Key Findings (TL;DR)

| # | Finding | Key numbers |
|---|---|---|
| 1 | **Overall champion: SGLang + NVFP4 + DSPARK** | decode 85.2 t/s (short) / 71.4 (200K); 2-way 147.6 t/s; prefill 6000+ t/s |
| 2 | **DSPARK fixes MTP long-context decay** | 200K decode 71.4 t/s stable (vLLM FP8+MTP: 15.3 ❌) |
| 3 | **Prefill gap is engine-level** | SGLang 6000+ t/s = 6× vLLM = 2× llama.cpp |
| 4 | **Best native Windows: vLLM** (no Docker) | NVFP4+MTP 63.6 t/s, out of the box |
| 5 | **Quality/minimal ops: llama.cpp Q6_K** | ~6.5-bit near-lossless, single exe |
| 6 | **FP4 GGUF boosts llama.cpp** | NVFP4 GGUF+MTP 68.1 t/s (1.7× Q6_K) |
| 7 | **270W power cap: zero impact** | peak 244W < 270W, 73°C healthy |
| 8 | **Memory discipline**: 0.80 util / one container | oversubscription froze GUI/RDP (measured) |

---

## 🖥️ Test Environment

| Item | Configuration |
|---|---|
| GPU | NVIDIA RTX PRO 5000 **72GB** Blackwell (sm_120, 73415 MiB, 270W capped) |
| CPU | Intel Core Ultra 7 270K Plus |
| OS | Windows 10 x64 (10.0.26200) + WSL2 + Docker Desktop |
| llama.cpp | b9692 (CUDA 13.3, native FP4) |
| vLLM | 0.26.0+cu132 (Python 3.12 + torch 2.11 cu130) |
| SGLang | lmsysorg/sglang:qwen38-27b image (Docker) |

## 🚀 Deployment Matrix

| Option | Weights | Speculative | Deploy | VRAM |
|:---|:---|---:|:---|:---:|
| **SGLang NVFP4+DSPARK** | RadixArk NVFP4 20.4 GiB | DSPARK (γ=7) | Docker | ~57 GiB |
| **SGLang FP8+DSPARK** | Official FP8 28.8 GiB | DSPARK (γ=7) | Docker | ~65 GiB |
| **vLLM NVFP4+MTP** | unsloth NVFP4 21.8 GiB | MTP (n=1) | native bat | 57 GiB (0.80) |
| vLLM FP8+MTP | Official FP8 28.8 GiB | MTP (n=1) | native bat | ~65 GiB |
| **llama.cpp FP4 GGUF+MTP** | esatapedico GGUF 18.3 GiB | draft-mtp | native bat | ~55 GiB |
| llama.cpp Q6_K | Q6_K_XL 24.1 GiB | draft-mtp | native bat | ~50 GiB |

> SGLang requires Docker on Windows; vLLM/llama.cpp run natively.
> ⚠️ Do not run SGLang and vLLM simultaneously (VRAM oversubscription), see §5.

---

## 📏 Methodology

- **prefill/TTFT**: streaming API; `prefill t/s = prompt_tokens/TTFT`; random text to defeat caches
- **decode**: streaming tokens/s (first token excluded)
- **Concurrency**: N threads; `aggregate = Σtokens / window`
- SGLang benchmarked inside the container (Docker port forwarding may fail on WSL2)
- GPU idle during all comparisons; multi-round averages

---

## 📊 Results (full matrix)

### 4.1 Prefill / TTFT (t/s)

| Prompt | SGLang NVFP4+DSPARK | SGLang FP8+DSPARK | vLLM NVFP4+MTP | vLLM FP8+MTP | llama.cpp FP4+MTP | llama.cpp Q6_K |
|:---|---:|---:|---:|---:|---:|---:|
| 2.7K | **5,388-6,772** | 4,858-5,030 | 931 | 1,064 | 706 | 749 |
| 37K | **5,221-5,249** | 4,306-4,309 | 2,325 | 3,543 | 1,556 | 1,576 |
| 232K | 1,900 | — | 1,305 | 2,012 | 734 | 768 |

> SGLang's prefill advantage is engine-level (hybrid GDN optimization + chunked 2048): **2-6× other engines**.

### 4.2 Decode (single stream, t/s)

| Context | SGLang NVFP4+DSPARK | SGLang FP8+DSPARK | llama.cpp FP4+MTP | vLLM NVFP4+MTP | vLLM FP8+MTP | llama.cpp Q6_K+MTP |
|:---|---:|---:|---:|---:|---:|---:|
| ~11K | **85.2** 🏆 | 64.3 | 68.1 | 63.6 | 43.2 | 61.9 |
| ~148-200K | **71.4** 🏆 | 52.7 | 42.2 | 57.8 | 15.3 ❌ | 40.0 |

> **DSPARK is the only speculative method that does not decay at long context** (confidence head adapts draft length); MTP-1 acceptance collapses at long context.

### 4.3 Two-Way Concurrency

| Metric | SGLang NVFP4+DSPARK | vLLM NVFP4+MTP | SGLang FP8+DSPARK | vLLM FP8+MTP | llama.cpp FP4+MTP |
|:---|---:|---:|---:|---:|---:|
| avg per stream | **76.8** | 56.3 | 55.7 | 44.0 | 35.6 |
| aggregate | **147.6** | 112.6 | 101.8 | 87.0 | 66.1 |

### 4.4 MTP vs DSPARK

| Dimension | MTP (n=1) | **DSPARK (γ=7)** |
|:---|:---|:---|
| Draft source | in-checkpoint 1-layer head | separate 1.36B draft + target aux features |
| Short-context gain | +16~28% | **+72% (NVFP4 49.6→85.2)** |
| Long context (200K) | decays (FP8 path -42% ❌) | **stable (71.4 t/s)** |
| Draft size | embedded | 2.53 GiB |
| Root cause | 1-layer head lacks capacity ([vLLM #47602](https://github.com/vllm-project/vllm/issues/47602)) | confidence head keeps acceptance at long context |

### 4.5 KV Cache & VRAM

| Config | KV format | VRAM | Note |
|:---|:---|:---:|:---|
| llama.cpp q8_0 KV | 8-bit | 50% saved | ~1% speed cost (200K, 3-round avg) |
| vLLM fp8_e4m3 KV | 8-bit | 2× pool capacity | fp8_e5m2 incompatible with fp8 checkpoints ⚠️ |
| SGLang mamba ratio 0.7 | — | — | controls KV/mamba state pool split |

**Memory discipline (measured lessons)**:
- vLLM `--gpu-memory-utilization 0.80`: 57 GiB total, 14.7 GiB GUI headroom (0.90 froze GUI/RDP)
- SGLang: **one container at a time** (two containers oversubscribe; FP8 got available=0)
- 270W cap: peak 244W, no speed/temp impact (73°C)

### 4.6 Quantization Quality Reference

| Format | Precision | vs FP16 |
|:---|---:|---|
| FP16/BF16 | 16-bit | baseline |
| FP8 (e4m3) | 8-bit | ~0.2-0.5% |
| Q6_K | ~6.5-bit | ~0.5-1% |
| NVFP4 | 4-bit FP | ~1-2% |
| FP4 GGUF | 4-bit | ~1-2% |

---

## 🤖 Agent Scenarios (rewritten)

| Scenario | First choice | Key numbers | Alternative |
|:---|:---|:---|:---|
| Interactive/tool calls (TTFT) | **SGLang NVFP4+DSPARK** | TTFT 0.4-0.5s @2.7K | vLLM NVFP4+MTP |
| High-concurrency sub-agents | **SGLang NVFP4+DSPARK** | 2-way 147.6 t/s | vLLM NVFP4+MTP |
| Long-doc decode (>30K) | **SGLang NVFP4+DSPARK** | 71.4 t/s @200K | vLLM NVFP4+MTP 57.8 |
| RAG ingest (prefill) | **SGLang (either)** | 6000+ t/s | vLLM FP8 |
| Quality-first/offline | llama.cpp Q6_K | near-lossless | vLLM FP8 |
| No Docker wanted | vLLM NVFP4+MTP | 63.6 t/s out of the box | — |

**One-liner**: **SGLang NVFP4+DSPARK is the current best**; pick vLLM NVFP4+MTP for simplicity, llama.cpp Q6_K for max quality.

---

## ⚠️ Pitfalls (methodology + deployment)

1. vLLM first request cold start 280s+ → warm up
2. Prefix cache contamination in benchmarks
3. **GPU oversubscription freezes GUI/RDP** (DWM runs on dGPU): vLLM 0.90 / dual SGLang containers
4. **WSL2 CUDA IPC broken**: SGLang needs `--mm-feature-transport cpu`
5. **Docker port forwarding fails on WSL2**: benchmark inside container (`docker exec`, `MSYS_NO_PATHCONV=1`)
6. Chinese comments in .bat crash under GBK → keep ASCII
7. Background-started servers land in Session 0, unkillable → double-click to start
8. fp8_e5m2 KV incompatible with fp8 checkpoints → fp8_e4m3
9. llama.cpp ~2.5s fixed per-request overhead (TTFT floor)

---

## 🔧 NVFP4 Deployment Notes (Windows)

vLLM marlin path needs 3 local patches; SGLang via Docker avoids them.
**Full SGLang tutorial**: [docs/SGLang-DSPARK-Windows-部署教程.md](docs/SGLang-DSPARK-Windows-部署教程.md)

---

## 📁 Repository Layout

```
├── README.md / README.en.md
├── docs/SGLang-DSPARK-Windows-部署教程.md
└── scripts/
    ├── conc_test.py / prefill_test.py      # benchmarks
    ├── start_qwen38q6_vision.bat           # llama.cpp Q6_K
    ├── start_qwen38_nvfp4_gguf_mtp.bat     # llama.cpp FP4 GGUF+MTP
    ├── start_qwen38_nvfp4_gguf_orig.bat    # llama.cpp FP4 GGUF ORIG
    ├── run_vllm_nvfp4_mtp_bench.bat        # vLLM NVFP4+MTP (0.80 + fp8 KV)
    ├── run_vllm_fp8_mtp_bench.bat          # vLLM FP8+MTP
    ├── start_sglang_nvfp4_dspark.ps1       # SGLang NVFP4+DSPARK
    └── start_sglang_fp8_dspark.ps1         # SGLang FP8+DSPARK
```

## 🔄 Reproduction

```bash
# prefill/TTFT and concurrency (all engines; SGLang inside container)
python scripts/prefill_test.py <API_BASE_URL> <model_name> <label>
python scripts/conc_test.py <API_BASE_URL> <model_name> <N> <max_tokens> <label>
```

---

## 🆚 Cross-Device Comparison (DGX Spark / PRO 5000 / PRO 6000) — rewritten

> Full doc: [PRO 6000 repo](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
> ⚠️ Engine versions differ (SGLang / vLLM 0.26 / vLLM 0.21 / llama.cpp) — magnitude reference

**Single-stream decode (t/s)**

| Context | DGX Spark NVFP4+MTP×3 | **PRO 5000 SGLang NVFP4+DSPARK** | PRO 5000 vLLM NVFP4+MTP | PRO 6000 vLLM NVFP4+MTP(n=2) | PRO 6000 Q6_K |
|:---|---:|---:|---:|---:|---:|
| Short (~1-11K) | ~21 | 85.2 | 63.6 | **100.2** | 55.4 |
| ~148-200K | 14.2 | **71.4** 🏆 | 57.8 | 43.7 (no MTP) | 35.7 |

**Prefill 200K (t/s) / TTFT (s)**

| DGX Spark | **PRO 5000 SGLang NVFP4+DSPARK** | PRO 5000 vLLM FP8 | PRO 6000 vLLM NVFP4+MTP |
|---|---:|---:|---:|
| 840 / 244s | 1,900 / 122s | 2,114 / 110s | **4,447 / 39.9s** |

**Concurrency aggregate (t/s)**

| Concurrency | PRO 5000 SGLang DSPARK | PRO 6000 vLLM NVFP4+MTP | PRO 6000 Q6_K(4 slots) |
|:---:|---:|---:|---:|
| 1 | 85.2 | 95.0 | 49.6 |
| 2 | 147.6 | **151.3** | 92.8 |
| 4 | — | 346.5 | 158.5 |
| 8 | — | **654.1** | — |

**New conclusions**:

1. **Engine optimization beats hardware gap**: PRO 5000 (72GB) + SGLang DSPARK **beats** PRO 6000 (96GB) + vLLM at long-context decode (71.4 vs 43.7) — choose the right speculative method and a smaller card outruns a bigger one
2. **PRO 6000 retains**: short-context peak (100.2 vs 85.2), concurrency scaling (8-way 654)
3. **DGX Spark's value**: 128GB unified memory + low-power desktop form; not comparable in raw speed
4. **Prediction**: PRO 6000 + SGLang DSPARK would likely dominate all dimensions (untested)
5. PRO 5000 short prefill 6000+ t/s is the highest of all three devices (SGLang engine)

---

## 🔗 Related Repositories

- [RTX PRO 6000 benchmark (incl. 3-device comparison)](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark)
- [DGX Spark NVFP4 deployment tutorial](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)
- [LMSYS Qwen3.8-27B cookbook (SGLang reference)](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)

---

*Tests & docs: Deep-AI-Evo · Model: Qwen3.8-27B (Apache-2.0) · Single-machine measurements; results may vary*
