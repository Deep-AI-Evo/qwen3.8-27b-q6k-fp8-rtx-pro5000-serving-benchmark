# Windows + Docker + SGLang + DSPARK 部署教程

> Qwen3.8-27B 的 DSPARK 投机解码部署（Windows 10 + Docker Desktop + WSL2）
> 按 [LMSYS Qwen3.8-27B cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B) 实现
> 实测日期：2026-08-17 · 硬件：RTX PRO 5000 72GB（270W 锁功率）

## 📊 性能摘要（实测）

| 指标 | NVFP4 + DSPARK | FP8 + DSPARK |
|:---|---:|---:|
| prefill 2K | 5,388-6,772 t/s | 4,858-5,030 t/s |
| prefill 32K | 5,221-5,249 t/s | 4,306-4,309 t/s |
| prefill 200K | 1,900 t/s | — |
| decode ~11K | **85.2 t/s** | 64.3 t/s |
| decode 200K | **71.4 t/s** | 52.7 t/s |
| 两并发·平均 | 76.8 t/s | 55.7 t/s |
| 两并发·总体 | **147.6 t/s** | 101.8 t/s |

**对比 vLLM NVFP4+MTP（63.6/57.8/112.6）：decode 快 25-34%，prefill 快 6 倍。**

---

## 1️⃣ 前置条件

| 组件 | 要求 | 验证命令 |
|:---|:---|:---|
| Windows | 10/11 x64 | — |
| WSL2 | 已启用，任一发行版 | `wsl --status` |
| Docker Desktop | 4.x+（29.x 实测） | `docker version` |
| NVIDIA 驱动 | ≥ 535（WSL 驱动与宿主一致） | `nvidia-smi`（WSL 内） |
| GPU 透传 | Docker 内可见 GPU | 见下 |

**GPU 透传验证**：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# 应显示你的 GPU（RTX PRO 5000 72GB）
```

---

## 2️⃣ 模型准备（hf-mirror 加速）

```bash
# 目标模型（选一或都下）
# RadixArk NVFP4（20.4 GiB，推荐——速度冠军）
https://hf-mirror.com/RadixArk/Qwen3.8-27B-NVFP4
# 官方 FP8（28.8 GiB）
https://hf-mirror.com/Qwen/Qwen3.8-27B-FP8

# DSPARK 草稿模型（2.53 GiB，必需）
https://hf-mirror.com/RadixArk/Qwen3.8-27B-DSpark
```

下载到本地目录（如 `H:\models\...`），挂载进容器。

---

## 3️⃣ 部署步骤

### 3.1 拉取官方镜像

```bash
docker pull lmsysorg/sglang:qwen38-27b
```

### 3.2 启动 NVFP4 + DSPARK（PowerShell）

```powershell
docker run -d --name sglang-nvfp4-dspark --gpus all -p 12345:12345 `
  -v "H:\models\RadixArk-Qwen3.8-27B-NVFP4:/models/nvfp4" `
  -v "H:\models\Qwen3.8-27B-DSpark:/models/dspark" `
  lmsysorg/sglang:qwen38-27b `
  python3 -m sglang.launch_server --model /models/nvfp4 `
  --speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark `
  --speculative-dspark-block-size 7 --mamba-full-memory-ratio 0.7 `
  --chunked-prefill-size 2048 --reasoning-parser qwen3 --tool-call-parser qwen3_coder `
  --mm-feature-transport cpu `
  --trust-remote-code --host 0.0.0.0 --port 12345
```

FP8 版同理（`--model /models/fp8` + 挂载 FP8 目录）。

> ⚠️ **必须用 PowerShell 执行**，不要用 Git Bash 直接跑 docker run（见踩坑 ①）

### 3.3 等待就绪

```bash
docker logs -f sglang-nvfp4-dspark
# 看到 "Initialized DSpark draft runner ... gamma=7" 和 "The server is fired up and ready to roll!" 即完成
# 加载耗时约 2-3 分钟（模型 20GB + 草稿 2.5GB + CUDA graph 捕获）
```

---

## 4️⃣ 踩坑记录（全部实测）

### ① Git Bash 路径转换破坏 docker 命令
**现象**：`-v "H:\models\...:/models/nvfp4"` 在 Git Bash 下变成 `C:/Program Files/Git/models/nvfp4`
**解决**：用 PowerShell 脚本执行 docker run（`scripts/start_sglang_nvfp4_dspark.ps1`）

### ② WSL2 CUDA IPC 句柄失效（致命）
**现象**：启动后期崩溃 `CUDA error: invalid resource handle`，`Failed to deserialize from cached pooled CUDA IPC handle`（SGLang 多模态特征池跨进程传输）
**解决**：加 `--mm-feature-transport cpu`（特征池走 CPU 传输，绕开 CUDA IPC）
**本质**：WSL2 + CUDA 13.x 的 IPC 句柄跨进程兼容问题，SGLang 的 cuda_ipc_transport 在容器内不可用

### ③ Docker Desktop 端口转发失效
**现象**：`docker ps` 显示 `0.0.0.0:12345->12345/tcp` 绑定正确，但 Windows 侧 `curl localhost:12345` 不通（容器内 200）
**解决**：重启 Docker Desktop / `wsl --shutdown` 可能修复；**不修复时直接在容器内测速**：
```bash
docker cp conc_test.py sglang-nvfp4-dspark:/tmp/
MSYS_NO_PATHCONV=1 docker exec sglang-nvfp4-dspark python3 /tmp/conc_test.py http://localhost:12345 /models/nvfp4 1 256 test
```

### ④ MSYS_NO_PATHCONV 必需
**现象**：`docker exec ... python3 /tmp/script.py` 被 Git Bash 转成 `C:/WINDOWS/TEMP/script.py`
**解决**：所有 docker exec 参数前加 `MSYS_NO_PATHCONV=1`

### ⑤ 显存管理
- 一个容器约 40+ GiB（NVFP4 20.4 + 草稿 2.5 + KV 池 + CUDA graph）
- **两个容器（NVFP4 + FP8）同时跑会超 72GB**，一次只跑一个
- 270W 功率限制不影响速度（实测峰值功耗 244W < 270W）

### ⑥ Docker Desktop 未启动
**现象**：`failed to connect to docker API at npipe`
**解决**：启动 `C:\Users\15744\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe`（注意安装路径可能在 LOCALAPPDATA 而非 Program Files）

### ⑦ 端口占用
- 建议容器用 12345/12347/12348 等独立端口，避免与本地 vLLM/llama.cpp 冲突

---

## 5️⃣ 测速方法（容器内）

```bash
# 复制测试脚本（仓库 scripts/ 下有）
docker cp scripts/conc_test.py sglang-nvfp4-dspark:/tmp/
docker cp scripts/prefill_test.py sglang-nvfp4-dspark:/tmp/

# 单流 decode（模型名用 /v1/models 返回的 id，通常 /models/nvfp4）
MSYS_NO_PATHCONV=1 docker exec sglang-nvfp4-dspark \
  python3 /tmp/conc_test.py http://localhost:12345 /models/nvfp4 1 256 single

# 两并发
MSYS_NO_PATHCONV=1 docker exec sglang-nvfp4-dspark \
  python3 /tmp/conc_test.py http://localhost:12345 /models/nvfp4 2 256 2way

# prefill/TTFT
MSYS_NO_PATHCONV=1 docker exec sglang-nvfp4-dspark \
  python3 /tmp/prefill_test.py http://localhost:12345 /models/nvfp4 prefill
```

---

## 6️⃣ 相关文件

| 文件 | 用途 |
|:---|:---|
| `scripts/start_sglang_nvfp4_dspark.ps1` | NVFP4 + DSPARK 启动 |
| `scripts/start_sglang_fp8_dspark.ps1` | FP8 + DSPARK 启动 |
| `scripts/conc_test.py` | 并发 decode 测速 |
| `scripts/prefill_test.py` | prefill/TTFT 测速 |

## 7️⃣ 参考

- [LMSYS Qwen3.8-27B cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)（草稿模型，SpecForge 训练）
- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
