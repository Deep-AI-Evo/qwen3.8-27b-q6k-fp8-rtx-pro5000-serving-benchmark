# ⚡ Qwen3.8-27B Deployment Benchmark Report

> 🌐 **中文** | [切换到中文](README.md)

> **llama.cpp (Q6_K) vs vLLM (FP8 vs NVFP4) — TTFT / Prefill / Decode / Concurrency**
> Measured on 2026-08-15 on real hardware; all results reproducible (scripts included in this repo)
>
> 📊 **Cross-device comparison (DGX Spark / RTX PRO 5000 / RTX PRO 6000)**:
> [Qwen3.8-27B Cross-Device Benchmark](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)

![Platform](https://img.shields.io/badge/Platform-Windows%2010-0078d6)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%205000%2072GB-76b900)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b9692-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.26.0%2Bcu132-purple)
![CUDA](https://img.shields.io/badge/CUDA-13.3-green)
![Tested](https://img.shields.io/badge/Tested-2026--08--15-yellow)

---

## 📑 Table of Contents

- [TL;DR](#-key-findings-tldr)
- [Test Environment & Deployment Options](#%EF%B8%8F-test-environment)
- [Results (prefill/decode/concurrency/KV)](#-results)
- [Agent Use-Case Selection](#-agent-scenarios-which-one)
- [Pitfalls](#-pitfalls-methodology-traps)
- [Cross-Device Comparison (DGX Spark / PRO 5000 / PRO 6000)](#-cross-device-comparison-dgx-spark--rtx-pro-5000--rtx-pro-6000) ⭐ at the bottom

> ⭐ Which device wins — **DGX Spark / RTX PRO 5000 / RTX PRO 6000**? Full
> [cross-device comparison](#-cross-device-comparison-dgx-spark--rtx-pro-5000--rtx-pro-6000)
> at the bottom (including PRO 6000 FP8 full-tier data).

---

## 📌 Key Findings (TL;DR)

| # | Finding | Evidence |
|---|---|---|
| 1 | **Decode champion: vLLM NVFP4 + MTP** | 63.6 t/s single-stream (60% faster than Q6_K); 112.6 t/s 2-way aggregate |
| 2 | **Prefill champion: vLLM FP8** | 2,114 t/s @200K (54% faster than NVFP4) |
| 3 | **Best for agents: vLLM (FP8 or NVFP4)** | TTFT leads at every context length — TTFT defines agent experience |
| 4 | **llama.cpp value: quality + minimal ops** | Q6_K highest weight precision (~6.5-bit), single-exe deployment |
| 5 | **q8_0 KV is a free win** | ~1% speed cost for 50% KV memory savings; concurrency 2→4 |
| 6 | **Higher concurrency = better GPU utilization** | 2-way aggregate throughput ~70% above single-stream (batching efficiency) |

---

## 🖥️ Test Environment

| Item | Configuration |
|---|---|
| GPU | NVIDIA RTX PRO 5000 **72GB** Blackwell (sm_120, 73415 MiB) |
| CPU | Intel Core Ultra 7 270K Plus |
| RAM | 127 GiB |
| OS | Windows 10 x64 (10.0.26200) |
| llama.cpp | b9692 (CUDA 13.3 build) |
| vLLM | 0.26.0+cu132 (Python 3.12 + torch 2.11 cu130) |

## 🚀 Three Deployment Options

| Option | Model | Weights | KV cache | Backend | MTP Speculative Decoding | Deployment Effort |
|:---|:---|---:|:---|:---|:---|:---|
| **llama.cpp Q6_K** | Q6_K_XL.gguf + mmproj | 24.1 GiB | q8_0 | Native CUDA | `draft-mtp` (+56% single-stream, no net loss at long ctx) | ⭐ Double-click to run |
| **vLLM FP8** | Official FP8 | 28.8 GiB | Default | flashinfer | Supported (+16~35% short/medium, **-42% long ctx** ⚠️) | ⭐⭐ Out of the box |
| **vLLM NVFP4** | unsloth NVFP4 | 21.8 GiB | Default | marlin(mxfp4) | Supported (**accelerates everywhere**, decode king) | ⭐⭐⭐⭐ 3 local patches (see [NVFP4 deployment notes](#-nvfp4-deployment-notes-windows-resolved)) |

---

## 📏 Methodology

- **prefill/TTFT**: OpenAI streaming API — `TTFT = request sent → first token`, `prefill t/s = prompt_tokens / TTFT`; every round uses **random text** to defeat cache hits
- **decode**: tokens/s during streaming generation (first token excluded)
- **Concurrency profile**: N threads send requests simultaneously, timed independently; `aggregate = Σtokens / window time`
- All comparisons run with an **idle GPU** (no other services); key data averaged over multiple rounds

---

## 📊 Results

### 4.1 Single-Stream Prefill / TTFT

| Prompt | llama.cpp Q6_K | vLLM FP8 | vLLM NVFP4 |
|:---|---:|---:|---:|
| 2.7K TTFT | 3.60s | 2.55s | 2.90s |
| 2.7K prefill | 749 t/s | 1,033 t/s | 915 t/s |
| 37K TTFT | 23.8s | 10.2s | 15.5s |
| 37K prefill | 1,576 t/s | **3,671 t/s** | 2,417 t/s |
| 232K TTFT | 302s | **110s** | 169s |
| 232K prefill | 768 t/s | **2,114 t/s** | 1,373 t/s |

> Note: llama.cpp has a ~2.5s fixed per-request overhead (scheduling/tokenization) that dilutes small-prompt prefill. Pure kernel prefill (llama-bench): 2.7K = 2112 t/s, 37K = 1822 t/s — consistent with attention cost scaling with context length.

### 4.2 Single-Stream Decode

| Context | llama.cpp Q6_K | vLLM FP8 | vLLM NVFP4 |
|:---|---:|---:|---:|
| ~11K | 39.7 t/s | 37.3 t/s | **49.6 t/s** 🏆 |
| ~150-200K | 39.9 t/s | 26.4 t/s | **42.1 t/s** 🏆 |

> 💡 **NVFP4's FP4 tensor-core advantage delivers**: decode leads everywhere — 25% faster than Q6_K at short context, 60% faster than FP8 at long context.

### 4.3 Two-Way Concurrency Decode Profile

| Metric | llama.cpp Q6_K | vLLM FP8 | vLLM NVFP4 |
|:---|---:|---:|---:|
| Single-stream baseline | 39.9 t/s | 37.3 t/s | **49.6 t/s** |
| 2-way · avg per stream | 34.1 t/s | 32.5 t/s | **45.0 t/s** |
| **2-way · aggregate** | 68.2 t/s | 64.7 t/s | **88.8 t/s** 🏆 |

> 💡 2-way aggregate (88.8 t/s) is ~80% above single-stream (49.6 t/s) — batching efficiency, a big win for multi-request agent workloads.

### 4.4 MTP Speculative Decoding (Multi-Token Prediction, 1-layer nextn)

> Enable: `--speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'`
> (vLLM automatically loads the `mtp.*` weights from the same checkpoint as drafter)

| Scenario | Q6_K no MTP | **Q6_K + MTP** | FP8 no MTP | **FP8 + MTP** | NVFP4 no MTP | **NVFP4 + MTP** |
|:---|---:|---:|---:|---:|---:|---:|
| Single ~11K | 39.7 | **61.9 (+56%)** | 37.3 | 43.2 (+16%) | 49.6 | **63.6 (+28%)** 🏆 |
| 2-way · avg/stream | 34.1 | 25.5 (-25%) ❌ | 32.5 | 44.0 (+35%) | 45.0 | **56.3 (+25%)** |
| 2-way · aggregate | 68.2 | 42.3 (-38%) ❌ | 64.7 | 87.0 (+34%) | 88.8 | **112.6 (+27%)** 🏆 |
| Long ctx ~148K | 39.9 | **40.0 (±0%)** | 26.4 | **15.3 (-42%)** ❌ | 42.1 | **57.8 (+37%)** 🏆 |

> ⚠️ **Key finding**: MTP is a **net regression at long context on FP8** (-42%, confirmed twice) —
> likely drafter KV/SSM state handling deficiency in the FP8 path (acceptance collapse, throughput ≈ half baseline);
> meanwhile **NVFP4 (marlin) MTP still gains +37% at long context**.
> Conclusion: **NVFP4 + MTP is the ultimate decode configuration**; FP8 + MTP is only for short/medium context;
> llama.cpp MTP gives big single-stream gains but regresses under concurrency.

### 4.5 MTP Impact on Prefill / TTFT

> MTP mainly affects decode; the draft model adds prefill overhead, measured as negligible:

| Prompt | Q6_K | Q6_K+MTP | FP8 | FP8+MTP | NVFP4 | NVFP4+MTP |
|:---|---:|---:|---:|---:|---:|---:|
| 2.7K prefill | 749 t/s | 706 (-6%) | 1033 | 1064 (+3%) | 915 | 931 (+2%) |
| 37K prefill | 1576 | 1556 (-1%) | 3671 | 3543 (-3%) | 2417 | 2325 (-4%) |
| 232K prefill | 768 | 734 (-4%) | 2114 | 2012 (-5%) | 1373 | 1305 (-5%) |
| 232K TTFT | 302s | 316s | 110s | 115s | 169s | 178s |

**MTP impact on prefill is within ±5% (extra TTFT is mostly draft-model prefill);
the prefill ranking is unchanged: FP8 > NVFP4 > Q6_K. All of MTP's gains/losses are on the decode side.**

### 4.6 MTP Long-Context Decay: Root Cause & Solutions (community research)

> Matches vLLM GitHub Issue **#47602** ([link](https://github.com/vllm-project/vllm/issues/47602)):
> *"Native MTP draft acceptance rate decays with total context length"* — consistent with our measurements.

**Decay curve (issue author's data, Qwen3.6-27B, MTP n=6)**:
| Context | MTP n=6 | No MTP | Δ |
|:---|---:|---:|---:|
| 2K | 138.0 t/s | 60.4 t/s | +129% |
| 8K | 88.3 | 60.1 | +47% |
| 16K | 51.0 | 59.1 | **-14%** |
| 30K | 26.8 | 54.6 | **-51%** |

Acceptance rate (position-1): 93.5% @2K → 72.1% @30K; overall average 64.9% → 39.1%.

**Root cause (community consensus, cross-engine verified)**:
- The MTP draft head is only **1 layer** (`mtp_num_hidden_layers=1`) vs the target's 64 layers
- Short context: the target's final hidden state already encodes enough for a 1-layer head to extend a few tokens
- Long context: consistent continuation requires **long-range multi-hop attention** — a single layer lacks the capacity, so draft guesses drift from the 64-layer target's true distribution
- DeepSeek abandoned MTP-1 for DSpark for exactly this reason ("as context grows, acceptance of draft tokens decreases")
- **Not a vLLM implementation bug**: reproduced on llama.cpp (Vulkan/AMD) and vLLM (CUDA/NVIDIA) alike (issue comment + our own measurements)

**Solutions / workarounds**:
| Option | Effect | Notes |
|---|---|---|
| **vLLM NVFP4 + MTP** | No long-context decay (+37%) | Measured locally; mechanism unclear (likely marlin vs flashinfer fp8 path difference) |
| **llama.cpp draft-mtp (n_max=4)** | +56% single-stream, parity at long ctx (**no net loss**) | Measured locally; regresses under 2-way concurrency (-38%) |
| vLLM FP8 + MTP | +16% short, -42% long | **Disable** MTP for long context |
| `num_speculative_tokens_per_batch_size` | Dynamic n by **batch size** | Not length-based; cannot directly solve this |
| Length-bucketed on/off in vLLM | Proposed in issue #47602 | Not yet merged |

**Practical advice**: for long contexts (>16K) either use NVFP4+MTP or disable MTP entirely; short/medium contexts benefit clearly from MTP.

### 4.7 KV Cache Format: f16 vs q8_0 (llama.cpp, 200K context, 3-round averages)

| Metric | f16 KV | q8_0 KV | Δ |
|:---|---:|---:|---:|
| prefill 200K | 873.04 t/s | 862.00 t/s | **-1.26%** |
| decode @200K | 39.94 t/s | 39.68 t/s | **-0.66%** |
| Memory (2×256K) | 58.0 GiB | 44.9 GiB | **13.4 GiB saved** 🎉 |

> **q8_0 KV trades ~1% speed for 50% KV memory — concurrency 2→4.**

### 4.9 Qwen3.8 NVFP4 GGUF Version Comparison (llama.cpp FP4 path)

> Source: [esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF)
> (unsloth NVFP4 converted to GGUF with embedded MTP head; llama.cpp b9692 native FP4 tensor cores)

| Metric | **VERY-HIGH** (18.3 GiB) | **ORIG** (30.9 GiB) |
|:---|---:|---:|
| prefill 2K | **3,559 t/s** | 2,877 t/s |
| prefill 200K | **1,026 t/s** | 985 t/s |
| decode ~2K | **56.7 t/s** | 35.0 t/s |
| decode 200K | **54.9 t/s** | 34.8 t/s |
| decode ~11K +MTP | **68.1 t/s** 🏆 | 54.7 t/s |
| decode 200K +MTP | 42.2 t/s | 36.8 t/s |
| 2-way avg +MTP | **35.6 t/s** | 21.3 t/s |
| 2-way agg +MTP | **66.1 t/s** | 33.9 t/s |
| Deployment | 4×256K ✅ | 2×256K (4×256K exceeds VRAM ⚠️) |

**VERY-HIGH wins everywhere**: decode 30-60% faster, 24% faster prefill, 12.6 GiB less VRAM, double concurrency.
ORIG's only edge is potentially higher fidelity (unverified). **VERY-HIGH + MTP single-stream 68.1 t/s is the fastest of all schemes tested (incl. vLLM NVFP4+MTP at 63.6).**

### 4.8 Quantization Quality Reference

| Format | Precision | Loss vs FP16 |
|:---|---:|---|
| FP16/BF16 | 16-bit | Baseline |
| FP8 (e4m3) | 8-bit FP | ~0.2-0.5%, essentially lossless |
| Q6_K | ~6.5-bit int | ~0.5-1%, near-lossless |
| NVFP4 | 4-bit FP (e2m1) | ~1-2%, measurable but not obvious |

---

## 🤖 Agent Scenarios: Which One?

| Agent workload trait | vLLM FP8 | vLLM NVFP4 | llama.cpp Q6_K |
|:---|:---:|:---:|:---:|
| Many short requests (tool calls) | ✅ Fastest TTFT | ✅ TTFT close | ⚠️ ~2.5s overhead floor |
| Sub-agent concurrent generation | ✅ agg 64.7 | ✅ **agg 88.8** | ✅ agg 68.2 |
| RAG long-doc prefill | ✅ **Fastest** | Medium | Slow |
| Long-form writing decode | ⚠️ 26-43 t/s | ✅ **58-64 t/s (MTP everywhere)** | ✅ 39-40 t/s |
| Weight quality | Near-lossless | 4-bit (~1-2% loss) | **Highest** |

**One-liner**: **vLLM NVFP4 is the best all-rounder** — fastest decode, highest concurrent throughput, smallest weights (21.8 GiB); FP8 wins only long-doc prefill; llama.cpp Q6_K retains value in quality and ops simplicity.

**Practical tips**:
- NVFP4/FP8: raise `--max-num-seqs` to 4-6; send a warm-up request after startup (first request can take 280s+ cold)
- llama.cpp: `--ctx-size 1048576 --parallel 4 --cache-type-k q8_0 --cache-type-v q8_0`
- Slice RAG documents to 10-50K tokens (200K prefill takes 110-302s — a physical bottleneck)

---

## ⚠️ Pitfalls (Methodology Traps)

1. **vLLM's first request after startup is extremely slow** (280s+) — CUDA graph/torch.compile cold start; warm up first
2. **Prefix cache contamination** — prompts sharing a prefix hit the cache, inflating prefill numbers (5794 t/s was fake; real: 2114 t/s)
3. **GPU contention contamination** — running two servers at once degrades vLLM; data unusable
4. **vLLM random seed** — thinking-mode behavior is unstable; fix the seed for reproducibility
5. **llama.cpp fixed per-request overhead** ~2.5s (TTFT floor, independent of prompt length)

---

## 🔧 NVFP4 Deployment Notes (Windows, Resolved ✅)

Model downloaded from hf-mirror (21.8 GiB, 1968 weight entries verified). The vLLM-on-Windows toolchain issue chain was **fully resolved with 3 local patches**:

| # | Problem | Root cause | Fix |
|:---:|---|---|---|
| 1 | C2719 "128-aligned param" | CUDA 13.3 cudafe stub × MSVC 14.44 incompatibility | Switch to marlin backend (prebuilt kernels, no JIT) |
| 2 | ParallelLMHead lacks `output_size_per_partition` | vLLM 0.26 bug (marlin path) | Patch ① `marlin_utils_fp8.py`: size_k_first adaptive |
| 3 | Same (forward path) | Same | Patch ② `scaled_mm/marlin.py`: fall back to `num_embeddings_per_partition`/`embedding_dim` |
| 4 | `from flashinfer import` resolved to a vLLM internal module | Engine subprocess sys.path shadows the real flashinfer | Patch ③ `site-packages/sitecustomize.py`: pre-import the real package at process start |

**Deployment command**: `scripts/run_vllm_nvfp4_bench.bat` (marlin backend + 3 patches)

**Performance note**: NVFP4 runs on the marlin(mxfp4) backend (not the flashinfer FP4 JIT path); measured speeds in §4 — decode leads across the board, FP4 tensor-core advantage fully delivered.

---

## 📁 Repository Layout

```
qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark/
├── README.md / README.en.md    # This report (中文 / English)
├── scripts/
│   ├── prefill_test.py         # prefill / TTFT test (random text, cache-proof)
│   ├── conc_test.py            # N-way concurrency decode profile test
│   ├── start_qwen38q6_vision.bat       # llama.cpp Q6_K deployment
│   ├── run_vllm_nvfp4_bench.bat        # vLLM NVFP4 deployment (marlin)
│   ├── run_vllm_fp8_mtp_bench.bat      # vLLM FP8 + MTP speculative decoding
│   └── run_vllm_nvfp4_mtp_bench.bat    # vLLM NVFP4 + MTP speculative decoding
```

## 🔄 Reproduction

```bash
# prefill/TTFT (works against all three servers)
python scripts/prefill_test.py <API_BASE_URL> <model_name> <label>

# N-way concurrency decode profile
python scripts/conc_test.py <API_BASE_URL> <model_name> <N> <max_tokens> <label>
```

---

## 🆚 Cross-Device Comparison (DGX Spark / RTX PRO 5000 / RTX PRO 6000)

> Summarized from the [3-device comparison](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md) (full tiers & methods there).
> ⚠️ Caveat: vLLM versions / MTP settings / OS differ across devices — treat as magnitude reference, not precise comparison.

**Single-stream decode (tok/s)**

| Context | DGX Spark NVFP4+MTP×3 | PRO 5000 FP8 noMTP → +MTP (this repo) | PRO 5000 NVFP4 noMTP → +MTP (this repo) | PRO 5000 Q6_K noMTP → +MTP (this repo) | PRO 6000 NVFP4 noMTP → +MTP(n=2) | PRO 6000 Q6_K |
|:---|---:|---:|---:|---:|---:|---:|
| Short (~1-11K) | ~21 | 37.3 → 43.2 | 49.6 → 63.6 | 39.7 → 61.9 | 58.6 → **100.2** | 55.4 |
| ~148K | — | 26.4 → 15.3 ❌ | 42.1 → **57.8** ✅ | 39.9 → 40.0 | — | — |
| ~200K | 14.2 | 26.4 (no MTP) | 42.1 (no MTP) | 39.9 | **43.7** → 18.2 ❌ | 35.7 |

**Single-stream prefill ~200K (tok/s) / TTFT (s)**

| DGX Spark NVFP4 | PRO 5000 FP8 (this repo) | PRO 5000 Q6_K (this repo) | **PRO 6000 FP8+MTP** | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K |
|---|---:|---:|---:|---:|---:|
| 840 / 244s | 2,114 / 110s | 768 / 302s | 3,869 / 45.8s | **4,447 / 39.9s** | 1,676 / 105.9s |

**Concurrent decode aggregate (tok/s)**

| Concurrency | DGX Spark NVFP4 | PRO 5000 FP8 (this repo) | PRO 5000 NVFP4 (this repo) | **PRO 5000 NVFP4+MTP (this repo)** | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K (4 slots) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.0 | 37.3 | 49.6 | **63.6** | 78.8 | 95.0 | 49.6 |
| 2 | 22.7 | 64.7 | 88.8 | **112.6** | 137.0 | 151.3 | 92.8 |
| 4 | 44.0 | — | — | — | 299.3 | 346.5 | 158.5 |
| 8 | 77.7 | — | — | — | 556.2 | **654.1** | — |

> Note: PRO 5000 measured up to 2-way only (72GB card; 4-way needs lower per-slot context); PRO 6000's 96GB reaches 8-way.

One-liner: **PRO 6000 leads prefill/TTFT/concurrency; the long-context decode champion is this repo's
NVFP4+MTP (n=1, marlin) — 57.8 t/s @148K, the highest measured across the three devices**. Note the MTP
long-context gain flips with backend/n: PRO 6000's n=2 config collapses at 200K (43.7 t/s only without MTP),
and FP8+MTP is a net loss at long context on both devices — measure your own config before use (see §4.4/§4.5).

---

## 🔗 Related Repositories

- [RTX PRO 6000 FP8 vs NVFP4 benchmark](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark) — includes [DGX Spark / PRO 5000 / PRO 6000 3-device comparison](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark/blob/main/docs/Qwen3.8-27B-跨设备横向对比.md)
- [DGX Spark NVFP4 deployment tutorial](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)

---

*Tests & docs: Deep-AI-Evo · Model: Qwen3.8-27B (Apache-2.0) · Single-machine measurements; results may vary by environment*
